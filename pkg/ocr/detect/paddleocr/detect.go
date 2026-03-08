package paddleocr

import (
	"fmt"
	"image"
	"math"

	"github.com/multippt/gopaddleocr/pkg/ocr/common"
	"github.com/multippt/gopaddleocr/pkg/ocr/utils"
	ort "github.com/yalue/onnxruntime_go"
)

const ModelName = "paddleocr-det"

type ModelConfig struct {
	LimitSideLength int
	Std             [3]float32
	Mean            [3]float32
	Thresh          float32
	BoxThresh       float64
	UnclipRatio     float64
	MinArea         int

	common.BaseModelConfig
}

// ---------------------------------------------------------------------------
// Model wraps the DB detection ONNX session
// ---------------------------------------------------------------------------

type Model struct {
	*common.OnnxModel
}

func NewModel() *Model {
	m := &Model{}
	m.OnnxModel = common.NewOnnxModel(m)
	return m
}

func (m *Model) GetName() string { return ModelName }

func (m *Model) GetDefaultConfig() common.ModelConfig {
	return &ModelConfig{
		LimitSideLength: 1280,
		Mean:            [3]float32{0.485, 0.456, 0.406},
		Std:             [3]float32{0.229, 0.224, 0.225},
		Thresh:          0.3,
		BoxThresh:       0.6,
		UnclipRatio:     2.0,
		MinArea:         16,
		BaseModelConfig: common.BaseModelConfig{
			OnnxConfig: common.Config{
				ModelPath: "ch_PP-OCRv5_server_det.onnx",
			},
		},
	}
}

// Detect performs detection and returns ordered boxes in original image space.
func (m *Model) Detect(img image.Image) ([]utils.Box, error) {
	config, ok := m.GetDefaultConfig().(*ModelConfig)
	if !ok {
		return nil, common.ErrInvalidConfig
	}

	bounds := img.Bounds()
	origW := bounds.Max.X - bounds.Min.X
	origH := bounds.Max.Y - bounds.Min.Y

	// Resize
	scale := 1.0
	if origW >= origH && origW > config.LimitSideLength {
		scale = float64(config.LimitSideLength) / float64(origW)
	} else if origH > origW && origH > config.LimitSideLength {
		scale = float64(config.LimitSideLength) / float64(origH)
	}
	resW := int(math.Round(float64(origW) * scale))
	resH := int(math.Round(float64(origH) * scale))
	if resW < 1 {
		resW = 1
	}
	if resH < 1 {
		resH = 1
	}

	// Pad to multiple of 32
	padW := ((resW + 31) / 32) * 32
	padH := ((resH + 31) / 32) * 32

	// Build NCHW float32 tensor (padded area filled with normalised zero).
	data := make([]float32, 3*padH*padW)
	zeroR := (0 - config.Mean[0]) / config.Std[0]
	zeroG := (0 - config.Mean[1]) / config.Std[1]
	zeroB := (0 - config.Mean[2]) / config.Std[2]
	// Fill padding with normalised 0.
	for c := 0; c < 3; c++ {
		var zv float32
		switch c {
		case 0:
			zv = zeroR
		case 1:
			zv = zeroG
		case 2:
			zv = zeroB
		}
		base := c * padH * padW
		for i := 0; i < padH*padW; i++ {
			data[base+i] = zv
		}
	}

	scaleX := float64(origW) / float64(resW)
	scaleY := float64(origH) / float64(resH)

	for py := 0; py < resH; py++ {
		// Sample from original image via nearest-neighbour for speed.
		sy := int(float64(py)*scaleY + 0.5)
		if sy >= origH {
			sy = origH - 1
		}
		sy += bounds.Min.Y
		for px := 0; px < resW; px++ {
			sx := int(float64(px)*scaleX + 0.5)
			if sx >= origW {
				sx = origW - 1
			}
			sx += bounds.Min.X
			r32, g32, b32, _ := img.At(sx, sy).RGBA()
			rv := float32(r32) / 65535.0
			gv := float32(g32) / 65535.0
			bv := float32(b32) / 65535.0
			idx := py*padW + px
			data[0*padH*padW+idx] = (rv - config.Mean[0]) / config.Std[0]
			data[1*padH*padW+idx] = (gv - config.Mean[1]) / config.Std[1]
			data[2*padH*padW+idx] = (bv - config.Mean[2]) / config.Std[2]
		}
	}

	// Inference
	shape := ort.NewShape(1, 3, int64(padH), int64(padW))
	inTensor, err := ort.NewTensor(shape, data)
	if err != nil {
		return nil, fmt.Errorf("det input tensor: %w", err)
	}
	defer func() {
		_ = inTensor.Destroy()
	}()

	outputs := make([]ort.Value, 1)
	if err := m.GetSession().Run([]ort.Value{inTensor}, outputs); err != nil {
		return nil, fmt.Errorf("det inference: %w", err)
	}
	outTensor, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		_ = outputs[0].Destroy()
		return nil, fmt.Errorf("unexpected det output type")
	}
	defer func() {
		_ = outTensor.Destroy()
	}()

	probData := outTensor.GetData() // (1,1,padH,padW) flattened

	// Postprocess
	boxes := m.postprocess(config, probData, padH, padW, resH, resW, origH, origW)
	return boxes, nil
}

// ---------------------------------------------------------------------------
// Postprocessing
// ---------------------------------------------------------------------------

func (m *Model) postprocess(
	config *ModelConfig, probData []float32, padH, padW, resH, resW, origH, origW int,
) []utils.Box {
	// Build binary mask.
	mask := make([]bool, padH*padW)
	for i, v := range probData {
		if i >= padH*padW {
			break
		}
		mask[i] = v > config.Thresh
	}

	// Connected components.
	components := connectedComponents(mask, padH, padW)

	// Scale factors: probability-map coords → original image coords.
	sx := float64(origW) / float64(resW)
	sy := float64(origH) / float64(resH)

	var boxes []utils.Box

	for _, comp := range components {
		if len(comp) < config.MinArea {
			continue
		}

		pts := make([]fPoint, len(comp))
		for i, idx := range comp {
			y := idx / padW
			x := idx % padW
			pts[i] = fPoint{float64(x), float64(y)}
		}

		hull := convexHull(pts)
		if len(hull) < 3 {
			continue
		}

		rect := minAreaRect(hull)
		rectPoly := [][2]float64{
			{rect[0].X, rect[0].Y},
			{rect[1].X, rect[1].Y},
			{rect[2].X, rect[2].Y},
			{rect[3].X, rect[3].Y},
		}

		// Box score is computed on the pre-unclip rectangle (matches PaddleOCR).
		if boxScore(probData, padW, rectPoly) < config.BoxThresh {
			continue
		}

		// Expand polygon, then get the min-area rect of the expanded shape.
		unclipped := unclipPoly(rectPoly, config.UnclipRatio)
		if len(unclipped) < 4 {
			continue
		}
		uHull := make([]fPoint, len(unclipped))
		for i, p := range unclipped {
			uHull[i] = fPoint{p[0], p[1]}
		}
		finalRect := minAreaRect(uHull)

		// Order and scale 4 corners to the original image space.
		corners := [4][2]float64{
			{finalRect[0].X, finalRect[0].Y},
			{finalRect[1].X, finalRect[1].Y},
			{finalRect[2].X, finalRect[2].Y},
			{finalRect[3].X, finalRect[3].Y},
		}
		ordered := utils.OrderPoints4(corners)
		var quad [4][2]int
		for i, p := range ordered {
			quad[i][0] = utils.ClampInt(int(math.Round(p[0]*sx)), 0, origW-1)
			quad[i][1] = utils.ClampInt(int(math.Round(p[1]*sy)), 0, origH-1)
		}
		boxes = append(boxes, utils.Box{Quad: quad, Score: boxScore(probData, padW, rectPoly), ClassID: -1, Order: -1})
	}

	return boxes
}

// ---------------------------------------------------------------------------
// Connected component labeling (union-find, 4-connectivity)
// ---------------------------------------------------------------------------

func connectedComponents(mask []bool, H, W int) [][]int {
	parent := make([]int, H*W)
	for i := range parent {
		parent[i] = i
	}

	var find func(int) int
	find = func(x int) int {
		for parent[x] != x {
			parent[x] = parent[parent[x]]
			x = parent[x]
		}
		return x
	}
	union := func(a, b int) {
		ra, rb := find(a), find(b)
		if ra != rb {
			parent[ra] = rb
		}
	}

	for y := 0; y < H; y++ {
		for x := 0; x < W; x++ {
			idx := y*W + x
			if !mask[idx] {
				continue
			}
			if x > 0 && mask[idx-1] {
				union(idx, idx-1)
			}
			if y > 0 && mask[idx-W] {
				union(idx, idx-W)
			}
		}
	}

	groups := make(map[int][]int)
	for y := 0; y < H; y++ {
		for x := 0; x < W; x++ {
			idx := y*W + x
			if !mask[idx] {
				continue
			}
			root := find(idx)
			groups[root] = append(groups[root], idx)
		}
	}

	result := make([][]int, 0, len(groups))
	for _, g := range groups {
		result = append(result, g)
	}
	return result
}

// ---------------------------------------------------------------------------
// Convex hull — Jarvis march / gift wrapping O(nh)
// ---------------------------------------------------------------------------

type fPoint struct{ X, Y float64 }

func cross2D(O, A, B fPoint) float64 {
	return (A.X-O.X)*(B.Y-O.Y) - (A.Y-O.Y)*(B.X-O.X)
}

func dist2(a, b fPoint) float64 {
	dx, dy := a.X-b.X, a.Y-b.Y
	return dx*dx + dy*dy
}

func convexHull(pts []fPoint) []fPoint {
	n := len(pts)
	if n < 3 {
		return pts
	}
	// Deduplicate.
	uniq := pts[:0:0]
	seen := make(map[[2]int64]bool)
	for _, p := range pts {
		key := [2]int64{int64(p.X * 10), int64(p.Y * 10)}
		if !seen[key] {
			seen[key] = true
			uniq = append(uniq, p)
		}
	}
	pts = uniq
	n = len(pts)
	if n < 3 {
		return pts
	}

	// Find starting point (leftmost, break ties by lowest Y).
	start := 0
	for i := 1; i < n; i++ {
		if pts[i].X < pts[start].X || (pts[i].X == pts[start].X && pts[i].Y < pts[start].Y) {
			start = i
		}
	}

	hull := []fPoint{pts[start]}
	cur := start
	for {
		next := 0
		if next == cur {
			next = 1
		}
		for i := 0; i < n; i++ {
			c := cross2D(pts[cur], pts[next], pts[i])
			if c < 0 || (c == 0 && dist2(pts[cur], pts[i]) > dist2(pts[cur], pts[next])) {
				next = i
			}
		}
		if next == start {
			break
		}
		hull = append(hull, pts[next])
		cur = next
		if len(hull) > n {
			break // safety
		}
	}
	return hull
}

// ---------------------------------------------------------------------------
// Minimum area rectangle (rotating calipers on convex hull)
// ---------------------------------------------------------------------------

func minAreaRect(hull []fPoint) [4]fPoint {
	n := len(hull)
	if n == 0 {
		return [4]fPoint{}
	}
	if n == 1 {
		return [4]fPoint{hull[0], hull[0], hull[0], hull[0]}
	}
	if n == 2 {
		return [4]fPoint{hull[0], hull[1], hull[1], hull[0]}
	}

	bestArea := math.MaxFloat64
	var bestRect [4]fPoint

	for i := 0; i < n; i++ {
		p1 := hull[i]
		p2 := hull[(i+1)%n]
		dx := p2.X - p1.X
		dy := p2.Y - p1.Y
		edgeLen := math.Sqrt(dx*dx + dy*dy)
		if edgeLen < 1e-10 {
			continue
		}
		// Unit vectors along edge and normal.
		ux, uy := dx/edgeLen, dy/edgeLen
		nx, ny := -uy, ux

		minU, maxU := math.MaxFloat64, -math.MaxFloat64
		minN, maxN := math.MaxFloat64, -math.MaxFloat64
		for _, p := range hull {
			u := (p.X-p1.X)*ux + (p.Y-p1.Y)*uy
			nv := (p.X-p1.X)*nx + (p.Y-p1.Y)*ny
			if u < minU {
				minU = u
			}
			if u > maxU {
				maxU = u
			}
			if nv < minN {
				minN = nv
			}
			if nv > maxN {
				maxN = nv
			}
		}

		area := (maxU - minU) * (maxN - minN)
		if area < bestArea {
			bestArea = area
			bestRect = [4]fPoint{
				{p1.X + minU*ux + minN*nx, p1.Y + minU*uy + minN*ny},
				{p1.X + maxU*ux + minN*nx, p1.Y + maxU*uy + minN*ny},
				{p1.X + maxU*ux + maxN*nx, p1.Y + maxU*uy + maxN*ny},
				{p1.X + minU*ux + maxN*nx, p1.Y + minU*uy + maxN*ny},
			}
		}
	}
	return bestRect
}

// ---------------------------------------------------------------------------
// Polygon unclip: expand polygon area by unclip_ratio using edge offsetting
// ---------------------------------------------------------------------------

func unclipPoly(poly [][2]float64, ratio float64) [][2]float64 {
	n := len(poly)
	if n < 3 {
		return poly
	}

	// Compute area (shoelace) and perimeter.
	area := 0.0
	perim := 0.0
	for i := 0; i < n; i++ {
		j := (i + 1) % n
		area += poly[i][0]*poly[j][1] - poly[j][0]*poly[i][1]
		dx := poly[j][0] - poly[i][0]
		dy := poly[j][1] - poly[i][1]
		perim += math.Sqrt(dx*dx + dy*dy)
	}
	if perim < 1e-10 {
		return poly
	}
	area = math.Abs(area) / 2.0
	distance := area * ratio / perim

	// Compute signed area to determine orientation.
	// In image coords (y-down), CW polygons have positive signed area.
	// The normals computed below point outward for CW; for CCW, invert.
	signed := 0.0
	for i := 0; i < n; i++ {
		j := (i + 1) % n
		signed += poly[i][0]*poly[j][1] - poly[j][0]*poly[i][1]
	}
	if signed < 0 {
		distance = -distance
	}

	// Compute outward normal for each edge.
	normals := make([][2]float64, n)
	for i := 0; i < n; i++ {
		j := (i + 1) % n
		dx := poly[j][0] - poly[i][0]
		dy := poly[j][1] - poly[i][1]
		l := math.Sqrt(dx*dx + dy*dy)
		if l < 1e-10 {
			normals[i] = [2]float64{0, 0}
			continue
		}
		normals[i] = [2]float64{dy / l, -dx / l}
	}

	// Offset each edge and find intersection of adjacent offset edges.
	result := make([][2]float64, n)
	for i := 0; i < n; i++ {
		j := (i + 1) % n
		// Offset edge i: p[i]→p[j], and prev edge (i-1)→p[i].
		prev := (i - 1 + n) % n
		nx1, ny1 := normals[prev][0]*distance, normals[prev][1]*distance
		nx2, ny2 := normals[i][0]*distance, normals[i][1]*distance

		p1s := [2]float64{poly[prev][0] + nx1, poly[prev][1] + ny1}
		p1e := [2]float64{poly[i][0] + nx1, poly[i][1] + ny1}
		p2s := [2]float64{poly[i][0] + nx2, poly[i][1] + ny2}
		p2e := [2]float64{poly[j][0] + nx2, poly[j][1] + ny2}

		result[i] = lineLineIntersect(p1s, p1e, p2s, p2e)
	}
	return result
}

func lineLineIntersect(p1, p2, p3, p4 [2]float64) [2]float64 {
	x1, y1 := p1[0], p1[1]
	x2, y2 := p2[0], p2[1]
	x3, y3 := p3[0], p3[1]
	x4, y4 := p4[0], p4[1]
	denom := (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
	if math.Abs(denom) < 1e-10 {
		return [2]float64{(p2[0] + p3[0]) / 2, (p2[1] + p3[1]) / 2}
	}
	t := ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
	return [2]float64{x1 + t*(x2-x1), y1 + t*(y2-y1)}
}

// ---------------------------------------------------------------------------
// Box score: mean probability inside polygon
// ---------------------------------------------------------------------------

func boxScore(probMap []float32, W int, poly [][2]float64) float64 {
	if len(poly) == 0 {
		return 0
	}
	// AABB of polygon.
	minX, minY := poly[0][0], poly[0][1]
	maxX, maxY := poly[0][0], poly[0][1]
	for _, p := range poly[1:] {
		if p[0] < minX {
			minX = p[0]
		}
		if p[0] > maxX {
			maxX = p[0]
		}
		if p[1] < minY {
			minY = p[1]
		}
		if p[1] > maxY {
			maxY = p[1]
		}
	}
	H := len(probMap) / W
	x0 := utils.ClampInt(int(math.Floor(minX)), 0, W-1)
	y0 := utils.ClampInt(int(math.Floor(minY)), 0, H-1)
	x1 := utils.ClampInt(int(math.Ceil(maxX)), 0, W-1)
	y1 := utils.ClampInt(int(math.Ceil(maxY)), 0, H-1)

	var sum float64
	var cnt int
	for y := y0; y <= y1; y++ {
		for x := x0; x <= x1; x++ {
			if utils.PointInPolyF(float64(x), float64(y), poly) {
				sum += float64(probMap[y*W+x])
				cnt++
			}
		}
	}
	if cnt == 0 {
		return 0
	}
	return sum / float64(cnt)
}
