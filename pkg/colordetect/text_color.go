package colordetect

import (
	"image"
	"math"
	"sort"

	"github.com/multippt/gopaddleocr/pkg/ocr/utils"
)

// ---------------------------------------------------------------------------
// ComputeTextColor
//
// Returns (textColor [R,G,B], fgMask [][]bool, cropRGBA *image.RGBA).
// fgMask and cropRGBA are in AABB-crop coordinates and are reused by
// computeWordColors to avoid duplicate work.
// ---------------------------------------------------------------------------

func ComputeTextColor(img image.Image, quad [4][2]int) ([]int, [][]bool, *image.RGBA) {
	xs := [4]int{quad[0][0], quad[1][0], quad[2][0], quad[3][0]}
	ys := [4]int{quad[0][1], quad[1][1], quad[2][1], quad[3][1]}
	x0, y0 := utils.MinInt4(xs), utils.MinInt4(ys)
	x1, y1 := utils.MaxInt4(xs), utils.MaxInt4(ys)

	bounds := img.Bounds()
	x0 = utils.ClampInt(x0, bounds.Min.X, bounds.Max.X-1)
	y0 = utils.ClampInt(y0, bounds.Min.Y, bounds.Max.Y-1)
	x1 = utils.ClampInt(x1, bounds.Min.X+1, bounds.Max.X)
	y1 = utils.ClampInt(y1, bounds.Min.Y+1, bounds.Max.Y)

	w, h := x1-x0, y1-y0
	if w <= 0 || h <= 0 {
		return nil, nil, nil
	}

	// Crop AABB and build polygon mask.
	arr := image.NewRGBA(image.Rect(0, 0, w, h))
	for py := 0; py < h; py++ {
		for px := 0; px < w; px++ {
			r32, g32, b32, a32 := img.At(x0+px, y0+py).RGBA()
			arr.SetRGBA(px, py, utils.ColorRGBA(r32, g32, b32, a32))
		}
	}

	// Local quad coordinates.
	localQ := [4][2]float64{
		{float64(quad[0][0] - x0), float64(quad[0][1] - y0)},
		{float64(quad[1][0] - x0), float64(quad[1][1] - y0)},
		{float64(quad[2][0] - x0), float64(quad[2][1] - y0)},
		{float64(quad[3][0] - x0), float64(quad[3][1] - y0)},
	}
	polyPts := [][2]float64{localQ[0], localQ[1], localQ[2], localQ[3]}

	// Polygon mask (true = inside quad).
	mask := make([][]bool, h)
	for py := 0; py < h; py++ {
		mask[py] = make([]bool, w)
		for px := 0; px < w; px++ {
			mask[py][px] = utils.PointInPolyF(float64(px), float64(py), polyPts)
		}
	}

	// Collect masked pixel grayscale values and RGB.
	var pixels [][3]uint8
	var grays []uint8
	for py := 0; py < h; py++ {
		for px := 0; px < w; px++ {
			if !mask[py][px] {
				continue
			}
			pix := arr.RGBAAt(px, py)
			pixels = append(pixels, [3]uint8{pix.R, pix.G, pix.B})
			gv := uint8((int(pix.R)*299 + int(pix.G)*587 + int(pix.B)*114) / 1000)
			grays = append(grays, gv)
		}
	}
	if len(pixels) < 4 {
		return nil, nil, nil
	}

	// Otsu threshold on grayscale values.
	t := otsuThreshold(grays)

	// Split into dark/light clusters.
	var darkPx, lightPx [][3]uint8
	darkMaskFlat := make([]bool, len(grays))
	for i, g := range grays {
		if int(g) <= t {
			darkPx = append(darkPx, pixels[i])
			darkMaskFlat[i] = true
		} else {
			lightPx = append(lightPx, pixels[i])
		}
	}

	if len(darkPx) == 0 || len(lightPx) == 0 {
		tc := medianRGB(pixels)
		fg := maskFromFlat(mask, h, w, makeAllTrue(len(grays)))
		return []int{int(tc[0]), int(tc[1]), int(tc[2])}, fg, arr
	}

	// Determine which cluster is text:
	// Strategy 1 — outside-polygon pixels as background reference.
	var outsidePx [][3]uint8
	for py := 0; py < h; py++ {
		for px := 0; px < w; px++ {
			if !mask[py][px] {
				pix := arr.RGBAAt(px, py)
				outsidePx = append(outsidePx, [3]uint8{pix.R, pix.G, pix.B})
			}
		}
	}

	var textPx [][3]uint8
	var fgFlat []bool

	const minBgPixels = 20
	if len(outsidePx) >= minBgPixels {
		bgMedian := medianRGB(outsidePx)
		darkMedian := medianRGB(darkPx)
		lightMedian := medianRGB(lightPx)
		dDark := colorDistSq(darkMedian, bgMedian)
		dLight := colorDistSq(lightMedian, bgMedian)
		if dDark < dLight {
			textPx = lightPx
			fgFlat = invertBool(darkMaskFlat)
		} else {
			textPx = darkPx
			fgFlat = darkMaskFlat
		}
	} else {
		// Strategy 2 — minority rule.
		if len(darkPx) <= len(lightPx) {
			textPx = darkPx
			fgFlat = darkMaskFlat
		} else {
			textPx = lightPx
			fgFlat = invertBool(darkMaskFlat)
		}
	}

	tc := medianRGB(textPx)
	fgMask := maskFromFlat(mask, h, w, fgFlat)
	return []int{int(tc[0]), int(tc[1]), int(tc[2])}, fgMask, arr
}

// ---------------------------------------------------------------------------
// ComputeWordColors
//
// Segments a single quad into color-uniform word regions using column
// (or row for vertical) occupancy gaps in the foreground mask.
// ---------------------------------------------------------------------------

type WordColorEntry struct {
	Start int   `json:"start"`
	End   int   `json:"end"`
	Color []int `json:"color"`
}

func ComputeWordColors(text string, quad [4][2]int, fgMask [][]bool, arr *image.RGBA, isVert bool) []WordColorEntry {
	if fgMask == nil || arr == nil || text == "" {
		return nil
	}
	nChars := len([]rune(text))
	h := len(fgMask)
	if h == 0 {
		return nil
	}
	w := len(fgMask[0])

	var colHasFg []bool
	var spanAxis int
	if !isVert {
		colHasFg = make([]bool, w)
		for py := 0; py < h; py++ {
			for px := 0; px < w; px++ {
				if fgMask[py][px] {
					colHasFg[px] = true
				}
			}
		}
		spanAxis = w
	} else {
		colHasFg = make([]bool, h)
		for py := 0; py < h; py++ {
			for px := 0; px < w; px++ {
				if fgMask[py][px] {
					colHasFg[py] = true
				}
			}
		}
		spanAxis = h
	}

	// Find contiguous foreground runs.
	type segment struct{ start, end int }
	var segs []segment
	inSeg := false
	segStart := 0
	for i, has := range colHasFg {
		if has && !inSeg {
			segStart = i
			inSeg = true
		} else if !has && inSeg {
			segs = append(segs, segment{segStart, i})
			inSeg = false
		}
	}
	if inSeg {
		segs = append(segs, segment{segStart, spanAxis})
	}
	if len(segs) <= 1 {
		return nil
	}

	var wordColors []WordColorEntry
	for _, seg := range segs {
		var segPx [][3]uint8
		for py := 0; py < h; py++ {
			for px := 0; px < w; px++ {
				inBand := false
				if !isVert {
					inBand = px >= seg.start && px < seg.end
				} else {
					inBand = py >= seg.start && py < seg.end
				}
				if inBand && fgMask[py][px] {
					pix := arr.RGBAAt(px, py)
					segPx = append(segPx, [3]uint8{pix.R, pix.G, pix.B})
				}
			}
		}
		if len(segPx) == 0 {
			continue
		}
		segColor := medianRGB(segPx)
		charStart := int(math.Round(float64(nChars) * float64(seg.start) / float64(spanAxis)))
		charEnd := int(math.Round(float64(nChars) * float64(seg.end) / float64(spanAxis)))
		charStart = utils.ClampInt(charStart, 0, nChars)
		charEnd = utils.ClampInt(charEnd, 0, nChars)
		if len(wordColors) > 0 {
			if charStart < wordColors[len(wordColors)-1].End {
				charStart = wordColors[len(wordColors)-1].End
			}
		}
		if charStart >= nChars {
			continue
		}
		if charEnd <= charStart {
			charEnd = charStart + 1
		}
		if charEnd > nChars {
			charEnd = nChars
		}
		wordColors = append(wordColors, WordColorEntry{
			Start: charStart,
			End:   charEnd,
			Color: []int{int(segColor[0]), int(segColor[1]), int(segColor[2])},
		})
	}

	if len(wordColors) <= 1 {
		return nil
	}

	wordColors = suppressColorOutliers(wordColors)
	wordColors = mergeSimilarColorSpans(wordColors)
	if len(wordColors) <= 1 {
		return nil
	}
	return wordColors
}

// ---------------------------------------------------------------------------
// Otsu threshold (pure Go, matches Python _otsu())
// ---------------------------------------------------------------------------

func otsuThreshold(grays []uint8) int {
	if len(grays) == 0 {
		return 128
	}
	hist := [256]int{}
	for _, v := range grays {
		hist[v]++
	}
	total := len(grays)
	sumTotal := 0.0
	for i, cnt := range hist {
		sumTotal += float64(i) * float64(cnt)
	}
	sumBg := 0.0
	wBg := 0
	bestThresh := 0
	bestVar := 0.0
	for t := 0; t < 256; t++ {
		wBg += hist[t]
		if wBg == 0 {
			continue
		}
		wFg := total - wBg
		if wFg == 0 {
			break
		}
		sumBg += float64(t) * float64(hist[t])
		meanBg := sumBg / float64(wBg)
		meanFg := (sumTotal - sumBg) / float64(wFg)
		varBetween := float64(wBg) * float64(wFg) * (meanBg - meanFg) * (meanBg - meanFg)
		if varBetween > bestVar {
			bestVar = varBetween
			bestThresh = t
		}
	}
	return bestThresh
}

// ---------------------------------------------------------------------------
// Outlier suppression and span merging (ports of Python helpers)
// ---------------------------------------------------------------------------

func suppressColorOutliers(wcs []WordColorEntry) []WordColorEntry {
	if len(wcs) < 3 {
		return wcs
	}
	colors := make([][3]float64, len(wcs))
	for i, wc := range wcs {
		colors[i] = [3]float64{float64(wc.Color[0]), float64(wc.Color[1]), float64(wc.Color[2])}
	}
	medC := medianColor(colors)

	distsSq := make([]float64, len(wcs))
	for i, c := range colors {
		distsSq[i] = colorDistSqF(c, medC)
	}
	medDistSq := medianFloat(distsSq)

	const absThresh = 6000.0
	const ratioThresh = 3.0
	const neighbourSim = 4000.0

	flags := make([]bool, len(wcs))
	for i, d := range distsSq {
		flags[i] = d > absThresh && (medDistSq == 0 || d > ratioThresh*medDistSq)
	}

	entries := make([]WordColorEntry, len(wcs))
	copy(entries, wcs)
	i := 0
	for i < len(entries) {
		if !flags[i] {
			i++
			continue
		}
		left := i - 1
		right := i + 1
		hasLeft := left >= 0
		hasRight := right < len(entries)
		if !hasLeft && !hasRight {
			i++
			continue
		}
		if hasLeft && hasRight {
			lc := [3]float64{float64(entries[left].Color[0]), float64(entries[left].Color[1]), float64(entries[left].Color[2])}
			rc := [3]float64{float64(entries[right].Color[0]), float64(entries[right].Color[1]), float64(entries[right].Color[2])}
			if colorDistSqF(lc, rc) > neighbourSim {
				i++
				continue
			}
		}
		if !hasLeft {
			entries[right].Start = entries[i].Start
		} else if !hasRight {
			entries[left].End = entries[i].End
		} else {
			ic := [3]float64{float64(entries[i].Color[0]), float64(entries[i].Color[1]), float64(entries[i].Color[2])}
			lc := [3]float64{float64(entries[left].Color[0]), float64(entries[left].Color[1]), float64(entries[left].Color[2])}
			rc := [3]float64{float64(entries[right].Color[0]), float64(entries[right].Color[1]), float64(entries[right].Color[2])}
			if colorDistSqF(ic, lc) <= colorDistSqF(ic, rc) {
				entries[left].End = entries[i].End
			} else {
				entries[right].Start = entries[i].Start
			}
		}
		entries = append(entries[:i], entries[i+1:]...)
		flags = append(flags[:i], flags[i+1:]...)
	}
	return entries
}

func mergeSimilarColorSpans(wcs []WordColorEntry) []WordColorEntry {
	if len(wcs) == 0 {
		return wcs
	}
	const mergeThresh = 2000.0
	merged := []WordColorEntry{wcs[0]}
	for _, wc := range wcs[1:] {
		prev := &merged[len(merged)-1]
		pc := [3]float64{float64(prev.Color[0]), float64(prev.Color[1]), float64(prev.Color[2])}
		cc := [3]float64{float64(wc.Color[0]), float64(wc.Color[1]), float64(wc.Color[2])}
		if colorDistSqF(pc, cc) <= mergeThresh {
			prevLen := prev.End - prev.Start
			currLen := wc.End - wc.Start
			total := prevLen + currLen
			if total == 0 {
				continue
			}
			blended := []int{
				int(math.Round((float64(prev.Color[0])*float64(prevLen) + float64(wc.Color[0])*float64(currLen)) / float64(total))),
				int(math.Round((float64(prev.Color[1])*float64(prevLen) + float64(wc.Color[1])*float64(currLen)) / float64(total))),
				int(math.Round((float64(prev.Color[2])*float64(prevLen) + float64(wc.Color[2])*float64(currLen)) / float64(total))),
			}
			prev.End = wc.End
			prev.Color = blended
		} else {
			merged = append(merged, wc)
		}
	}
	return merged
}

// ---------------------------------------------------------------------------
// Color utility helpers
// ---------------------------------------------------------------------------

func colorDistSq(a, b [3]uint8) float64 {
	dr := float64(a[0]) - float64(b[0])
	dg := float64(a[1]) - float64(b[1])
	db := float64(a[2]) - float64(b[2])
	return dr*dr + dg*dg + db*db
}

func colorDistSqF(a, b [3]float64) float64 {
	dr, dg, db := a[0]-b[0], a[1]-b[1], a[2]-b[2]
	return dr*dr + dg*dg + db*db
}

func medianRGB(px [][3]uint8) [3]uint8 {
	if len(px) == 0 {
		return [3]uint8{}
	}
	ch := [3][]int{{}, {}, {}}
	for _, p := range px {
		ch[0] = append(ch[0], int(p[0]))
		ch[1] = append(ch[1], int(p[1]))
		ch[2] = append(ch[2], int(p[2]))
	}
	return [3]uint8{
		uint8(medianInt(ch[0])),
		uint8(medianInt(ch[1])),
		uint8(medianInt(ch[2])),
	}
}

func medianInt(vals []int) int {
	if len(vals) == 0 {
		return 0
	}
	sorted := make([]int, len(vals))
	copy(sorted, vals)
	sort.Ints(sorted)
	return sorted[len(sorted)/2]
}

func medianFloat(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	sorted := make([]float64, len(vals))
	copy(sorted, vals)
	sort.Float64s(sorted)
	return sorted[len(sorted)/2]
}

func medianColor(colors [][3]float64) [3]float64 {
	if len(colors) == 0 {
		return [3]float64{}
	}
	ch := [3][]float64{{}, {}, {}}
	for _, c := range colors {
		ch[0] = append(ch[0], c[0])
		ch[1] = append(ch[1], c[1])
		ch[2] = append(ch[2], c[2])
	}
	return [3]float64{medianFloat(ch[0]), medianFloat(ch[1]), medianFloat(ch[2])}
}

// maskFromFlat converts the flat per-polygon-pixel bool slice back to a 2D mask
// over the full crop (same shape as fgMask).
func maskFromFlat(polygonMask [][]bool, h, w int, flat []bool) [][]bool {
	fg := make([][]bool, h)
	for i := range fg {
		fg[i] = make([]bool, w)
	}
	idx := 0
	for py := 0; py < h; py++ {
		for px := 0; px < w; px++ {
			if polygonMask[py][px] {
				if idx < len(flat) {
					fg[py][px] = flat[idx]
				}
				idx++
			}
		}
	}
	return fg
}

func makeAllTrue(n int) []bool {
	b := make([]bool, n)
	for i := range b {
		b[i] = true
	}
	return b
}

func invertBool(b []bool) []bool {
	out := make([]bool, len(b))
	for i, v := range b {
		out[i] = !v
	}
	return out
}
