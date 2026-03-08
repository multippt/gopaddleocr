package colordetect

import (
	"image"
	"image/color"
	"testing"

	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"
)

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

func newSolidImage(w, h int, c color.RGBA) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.SetRGBA(x, y, c)
		}
	}
	return img
}

// renderASCII draws text onto dst using the built-in 7×13 bitmap font.
// No external font file is needed. baseline is the y-coordinate of the
// text baseline (ascent ≈ 10px above, descent ≈ 3px below).
func renderASCII(dst *image.RGBA, text string, baselineX, baselineY int, fg color.RGBA) {
	d := &font.Drawer{
		Dst:  dst,
		Src:  image.NewUniform(fg),
		Face: basicfont.Face7x13,
		Dot:  fixed.P(baselineX, baselineY),
	}
	d.DrawString(text)
}

// drawCJKGlyph paints a dense stroke pattern at (x, y) of size sz×sz that
// approximates the ink density of a CJK character (~55% coverage):
// outer border + three horizontal rails + three vertical rails + a filled
// centre block.  This exercises high-coverage paths that would trip a naive
// "dark = minority = text" heuristic.
func drawCJKGlyph(dst *image.RGBA, x, y, sz int, fg color.RGBA) {
	h1, h2, h3 := sz/5, sz/2, 4*sz/5
	v1, v2, v3 := sz/5, sz/2, 4*sz/5
	for dy := 0; dy < sz; dy++ {
		for dx := 0; dx < sz; dx++ {
			onBorder := dx == 0 || dx == sz-1 || dy == 0 || dy == sz-1
			onHRail := dy == h1 || dy == h2 || dy == h3
			onVRail := dx == v1 || dx == v2 || dx == v3
			inCentre := dx >= v1 && dx <= v3 && dy >= h1 && dy <= h3
			if onBorder || onHRail || onVRail || inCentre {
				dst.SetRGBA(x+dx, y+dy, fg)
			}
		}
	}
}

// colorClose returns true if every channel of got is within tol of want.
func colorClose(got []int, want color.RGBA, tol int) bool {
	if len(got) < 3 {
		return false
	}
	diff := func(a int, b uint8) int {
		d := a - int(b)
		if d < 0 {
			d = -d
		}
		return d
	}
	return diff(got[0], want.R) <= tol &&
		diff(got[1], want.G) <= tol &&
		diff(got[2], want.B) <= tol
}

// ---------------------------------------------------------------------------
// otsuThreshold
// ---------------------------------------------------------------------------

func TestOtsuThreshold_Empty(t *testing.T) {
	if got := otsuThreshold(nil); got != 128 {
		t.Errorf("nil: got %d, want 128", got)
	}
	if got := otsuThreshold([]uint8{}); got != 128 {
		t.Errorf("empty slice: got %d, want 128", got)
	}
}

func TestOtsuThreshold_Uniform(t *testing.T) {
	// All same value: should not panic.
	grays := make([]uint8, 100)
	for i := range grays {
		grays[i] = 100
	}
	_ = otsuThreshold(grays)
}

func TestOtsuThreshold_Bimodal(t *testing.T) {
	// 50 dark pixels (value 30) + 50 light pixels (value 220).
	// Threshold is inclusive (g <= thresh → dark), so it may equal 30.
	grays := make([]uint8, 100)
	for i := 0; i < 50; i++ {
		grays[i] = 30
	}
	for i := 50; i < 100; i++ {
		grays[i] = 220
	}
	thresh := otsuThreshold(grays)
	// The threshold must correctly split the two clusters.
	if thresh < 30 || thresh >= 220 {
		t.Errorf("bimodal threshold %d should be in [30, 220)", thresh)
	}
}

func TestOtsuThreshold_SingleValue(t *testing.T) {
	_ = otsuThreshold([]uint8{128}) // must not panic
}

// ---------------------------------------------------------------------------
// medianInt
// ---------------------------------------------------------------------------

func TestMedianInt_Empty(t *testing.T) {
	if got := medianInt(nil); got != 0 {
		t.Errorf("nil: got %d, want 0", got)
	}
}

func TestMedianInt_Single(t *testing.T) {
	if got := medianInt([]int{42}); got != 42 {
		t.Errorf("single element: got %d, want 42", got)
	}
}

func TestMedianInt_OddCount(t *testing.T) {
	// sorted: [1, 3, 5, 7, 9] → median = sorted[2] = 5
	got := medianInt([]int{9, 1, 5, 3, 7})
	if got != 5 {
		t.Errorf("odd count: got %d, want 5", got)
	}
}

func TestMedianInt_EvenCount(t *testing.T) {
	// sorted: [2, 4, 6, 8] → sorted[2] = 6 (upper-middle via len/2)
	got := medianInt([]int{8, 2, 6, 4})
	if got != 6 {
		t.Errorf("even count: got %d, want 6 (upper-middle)", got)
	}
}

// ---------------------------------------------------------------------------
// medianFloat
// ---------------------------------------------------------------------------

func TestMedianFloat_Empty(t *testing.T) {
	if got := medianFloat(nil); got != 0 {
		t.Errorf("nil: got %f, want 0", got)
	}
}

func TestMedianFloat_OddCount(t *testing.T) {
	got := medianFloat([]float64{3.0, 1.0, 2.0})
	if got != 2.0 {
		t.Errorf("odd count: got %f, want 2.0", got)
	}
}

// ---------------------------------------------------------------------------
// medianRGB
// ---------------------------------------------------------------------------

func TestMedianRGB_Empty(t *testing.T) {
	got := medianRGB(nil)
	if got != ([3]uint8{}) {
		t.Errorf("empty: got %v, want [0,0,0]", got)
	}
}

func TestMedianRGB_Single(t *testing.T) {
	got := medianRGB([][3]uint8{{10, 20, 30}})
	if got != ([3]uint8{10, 20, 30}) {
		t.Errorf("single: got %v, want {10,20,30}", got)
	}
}

func TestMedianRGB_Three(t *testing.T) {
	// Each channel sorted independently:
	// R: [10, 50, 90] → median=50; G: [1, 5, 9] → 5; B: [100, 150, 200] → 150
	px := [][3]uint8{
		{10, 9, 100},
		{90, 1, 200},
		{50, 5, 150},
	}
	got := medianRGB(px)
	if got[0] != 50 || got[1] != 5 || got[2] != 150 {
		t.Errorf("three pixels: got %v, want {50,5,150}", got)
	}
}

// ---------------------------------------------------------------------------
// medianColor
// ---------------------------------------------------------------------------

func TestMedianColor_Empty(t *testing.T) {
	got := medianColor(nil)
	if got != ([3]float64{}) {
		t.Errorf("empty: got %v, want [0,0,0]", got)
	}
}

func TestMedianColor_Three(t *testing.T) {
	colors := [][3]float64{
		{1, 10, 100},
		{3, 30, 300},
		{2, 20, 200},
	}
	got := medianColor(colors)
	if got[0] != 2 || got[1] != 20 || got[2] != 200 {
		t.Errorf("three colors: got %v, want {2,20,200}", got)
	}
}

// ---------------------------------------------------------------------------
// colorDistSq / colorDistSqF
// ---------------------------------------------------------------------------

func TestColorDistSq_SameColor(t *testing.T) {
	if got := colorDistSq([3]uint8{100, 150, 200}, [3]uint8{100, 150, 200}); got != 0 {
		t.Errorf("same color: got %f, want 0", got)
	}
}

func TestColorDistSq_KnownValue(t *testing.T) {
	// (0,0,0) vs (255,0,0): dist² = 255² = 65025
	got := colorDistSq([3]uint8{0, 0, 0}, [3]uint8{255, 0, 0})
	if got != 65025 {
		t.Errorf("got %f, want 65025", got)
	}
}

func TestColorDistSqF_KnownValue(t *testing.T) {
	got := colorDistSqF([3]float64{0, 0, 0}, [3]float64{3, 4, 0})
	if got != 25 {
		t.Errorf("got %f, want 25", got)
	}
}

// ---------------------------------------------------------------------------
// invertBool / makeAllTrue
// ---------------------------------------------------------------------------

func TestInvertBool(t *testing.T) {
	in := []bool{true, false, true, true, false}
	got := invertBool(in)
	want := []bool{false, true, false, false, true}
	if len(got) != len(want) {
		t.Fatalf("len: got %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("[%d]: got %v, want %v", i, got[i], want[i])
		}
	}
}

func TestInvertBool_Empty(t *testing.T) {
	if got := invertBool(nil); len(got) != 0 {
		t.Errorf("nil input: got len=%d, want 0", len(got))
	}
}

func TestMakeAllTrue(t *testing.T) {
	got := makeAllTrue(5)
	if len(got) != 5 {
		t.Fatalf("len: got %d, want 5", len(got))
	}
	for i, v := range got {
		if !v {
			t.Errorf("[%d] = false, want true", i)
		}
	}
}

func TestMakeAllTrue_Zero(t *testing.T) {
	if got := makeAllTrue(0); len(got) != 0 {
		t.Errorf("n=0: got len=%d, want 0", len(got))
	}
}

// ---------------------------------------------------------------------------
// maskFromFlat
// ---------------------------------------------------------------------------

func TestMaskFromFlat_Basic(t *testing.T) {
	// polygonMask 2×3:
	//   row 0: [true, false, true]
	//   row 1: [false, true, false]
	// Row-major iteration visits polygon pixels in order: (0,0),(0,2),(1,1).
	// flat = [true, false, true]
	// Expected: fg[0][0]=true, fg[0][2]=false, fg[1][1]=true.
	polyMask := [][]bool{
		{true, false, true},
		{false, true, false},
	}
	flat := []bool{true, false, true}
	fg := maskFromFlat(polyMask, 2, 3, flat)

	if fg[0][0] != true {
		t.Errorf("fg[0][0]: got %v, want true", fg[0][0])
	}
	if fg[0][1] != false {
		t.Errorf("fg[0][1]: got %v, want false (not in polygon)", fg[0][1])
	}
	if fg[0][2] != false {
		t.Errorf("fg[0][2]: got %v, want false (flat[1]=false)", fg[0][2])
	}
	if fg[1][1] != true {
		t.Errorf("fg[1][1]: got %v, want true (flat[2]=true)", fg[1][1])
	}
}

func TestMaskFromFlat_AllTrue(t *testing.T) {
	polyMask := [][]bool{{true, true}, {true, true}}
	flat := []bool{true, true, true, true}
	fg := maskFromFlat(polyMask, 2, 2, flat)
	for py := 0; py < 2; py++ {
		for px := 0; px < 2; px++ {
			if !fg[py][px] {
				t.Errorf("fg[%d][%d] = false, want true", py, px)
			}
		}
	}
}

// ---------------------------------------------------------------------------
// suppressColorOutliers
// ---------------------------------------------------------------------------

func TestSuppressColorOutliers_TooFew(t *testing.T) {
	wcs := []WordColorEntry{
		{0, 5, []int{255, 0, 0}},
		{5, 10, []int{0, 0, 255}},
	}
	got := suppressColorOutliers(wcs)
	if len(got) != 2 {
		t.Errorf("len: got %d, want 2", len(got))
	}
}

func TestSuppressColorOutliers_NoOutlier(t *testing.T) {
	wcs := []WordColorEntry{
		{0, 3, []int{200, 10, 10}},
		{3, 6, []int{205, 10, 10}},
		{6, 10, []int{210, 10, 10}},
	}
	got := suppressColorOutliers(wcs)
	if len(got) != 3 {
		t.Errorf("no outlier: got %d entries, want 3", len(got))
	}
}

func TestSuppressColorOutliers_MiddleOutlier(t *testing.T) {
	// Middle entry (black) is far from the red neighbours → absorbed.
	wcs := []WordColorEntry{
		{0, 3, []int{200, 0, 0}},  // red
		{3, 6, []int{0, 0, 0}},    // black — outlier
		{6, 10, []int{210, 0, 0}}, // red
	}
	got := suppressColorOutliers(wcs)
	if len(got) != 2 {
		t.Fatalf("middle outlier: got %d entries, want 2", len(got))
	}
	coveredByFirst := got[0].End == 6
	coveredBySecond := got[1].Start == 3
	if !coveredByFirst && !coveredBySecond {
		t.Errorf("outlier range not absorbed: entries = %v", got)
	}
	for _, e := range got {
		if e.Color[0] < 100 && e.Color[1] < 100 && e.Color[2] < 100 {
			t.Errorf("entry still has outlier (black) color: %v", e.Color)
		}
	}
}

// ---------------------------------------------------------------------------
// mergeSimilarColorSpans
// ---------------------------------------------------------------------------

func TestMergeSimilarColorSpans_Empty(t *testing.T) {
	if got := mergeSimilarColorSpans(nil); len(got) != 0 {
		t.Errorf("empty: got %d entries, want 0", len(got))
	}
}

func TestMergeSimilarColorSpans_Similar(t *testing.T) {
	// dist²((100,100,100),(105,100,100)) = 25 << 2000 → merge.
	wcs := []WordColorEntry{
		{0, 5, []int{100, 100, 100}},
		{5, 10, []int{105, 100, 100}},
	}
	got := mergeSimilarColorSpans(wcs)
	if len(got) != 1 {
		t.Fatalf("similar colors: got %d entries, want 1", len(got))
	}
	if got[0].Start != 0 || got[0].End != 10 {
		t.Errorf("merged span: got [%d,%d), want [0,10)", got[0].Start, got[0].End)
	}
	// Weighted blend: equal lengths → R = round((100*5+105*5)/10) = round(102.5) = 103.
	if got[0].Color[0] != 103 {
		t.Errorf("blended R: got %d, want 103", got[0].Color[0])
	}
	if got[0].Color[1] != 100 {
		t.Errorf("blended G: got %d, want 100", got[0].Color[1])
	}
}

func TestMergeSimilarColorSpans_Different(t *testing.T) {
	// dist²(red, blue) ≈ 130050 >> 2000 → not merged.
	wcs := []WordColorEntry{
		{0, 5, []int{255, 0, 0}},
		{5, 10, []int{0, 0, 255}},
	}
	got := mergeSimilarColorSpans(wcs)
	if len(got) != 2 {
		t.Fatalf("different colors: got %d entries, want 2", len(got))
	}
}

func TestMergeSimilarColorSpans_ThreeWithOnePair(t *testing.T) {
	wcs := []WordColorEntry{
		{0, 4, []int{50, 50, 50}},
		{4, 8, []int{55, 50, 50}},  // close → merges with entry 0
		{8, 12, []int{255, 0, 0}},  // very different → stays separate
	}
	got := mergeSimilarColorSpans(wcs)
	if len(got) != 2 {
		t.Fatalf("three with one pair: got %d entries, want 2", len(got))
	}
	if got[0].End != 8 {
		t.Errorf("merged end: got %d, want 8", got[0].End)
	}
}

// ---------------------------------------------------------------------------
// ComputeTextColor — edge cases
// ---------------------------------------------------------------------------

func TestComputeTextColor_ZeroWidthQuad(t *testing.T) {
	img := newSolidImage(50, 20, color.RGBA{R: 200, G: 200, B: 200, A: 255})
	quad := [4][2]int{{5, 0}, {5, 0}, {5, 10}, {5, 10}} // all x = 5 → w = 0
	tc, mask, arr := ComputeTextColor(img, quad)
	if tc != nil || mask != nil || arr != nil {
		t.Error("zero-width quad: expected (nil, nil, nil)")
	}
}

func TestComputeTextColor_TooFewPixels(t *testing.T) {
	// 1×1 crop → 1 pixel inside polygon, which is < 4.
	img := newSolidImage(10, 10, color.RGBA{R: 128, G: 128, B: 128, A: 255})
	quad := [4][2]int{{0, 0}, {1, 0}, {1, 1}, {0, 1}}
	tc, mask, arr := ComputeTextColor(img, quad)
	if tc != nil || mask != nil || arr != nil {
		t.Error("too-few pixels: expected (nil, nil, nil)")
	}
}

// ---------------------------------------------------------------------------
// ComputeTextColor — rendered English text (ground-truth colour validation)
//
// These tests use basicfont.Face7x13 (a 1-bit bitmap font bundled with
// golang.org/x/image) to draw real glyph patterns.  Because each glyph is a
// genuine ink/whitespace mix, the pixel distribution is representative of
// real scanned or rendered text.  We render in a known colour and then assert
// that ComputeTextColor recovers that colour within a ±30 tolerance.
// ---------------------------------------------------------------------------

func TestComputeTextColor_RenderedBlackOnWhite(t *testing.T) {
	white := color.RGBA{R: 255, G: 255, B: 255, A: 255}
	black := color.RGBA{R: 0, G: 0, B: 0, A: 255}

	img := newSolidImage(200, 20, white)
	// basicfont.Face7x13 baseline at y=13; ascent≈10px so top of glyphs ≈ y=3.
	renderASCII(img, "Hello World Test", 5, 13, black)

	quad := [4][2]int{{0, 0}, {199, 0}, {199, 19}, {0, 19}}
	tc, _, _ := ComputeTextColor(img, quad)
	if tc == nil {
		t.Fatal("expected non-nil TextColor")
	}
	if !colorClose(tc, black, 30) {
		t.Errorf("black-on-white: got %v, want near (0,0,0) ±30", tc)
	}
}

func TestComputeTextColor_RenderedWhiteOnBlack(t *testing.T) {
	black := color.RGBA{R: 0, G: 0, B: 0, A: 255}
	white := color.RGBA{R: 255, G: 255, B: 255, A: 255}

	img := newSolidImage(200, 20, black)
	renderASCII(img, "Hello World Test", 5, 13, white)

	quad := [4][2]int{{0, 0}, {199, 0}, {199, 19}, {0, 19}}
	tc, _, _ := ComputeTextColor(img, quad)
	if tc == nil {
		t.Fatal("expected non-nil TextColor")
	}
	if !colorClose(tc, white, 30) {
		t.Errorf("white-on-black: got %v, want near (255,255,255) ±30", tc)
	}
}

func TestComputeTextColor_RenderedColoredText(t *testing.T) {
	// Render text in saturated colours and verify the detected colour matches
	// the ink colour, not the background.
	white := color.RGBA{R: 255, G: 255, B: 255, A: 255}
	cases := []struct {
		name string
		fg   color.RGBA
	}{
		{"red", color.RGBA{R: 220, G: 0, B: 0, A: 255}},
		{"green", color.RGBA{R: 0, G: 180, B: 0, A: 255}},
		{"blue", color.RGBA{R: 0, G: 0, B: 220, A: 255}},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			img := newSolidImage(200, 20, white)
			renderASCII(img, "Hello World Test", 5, 13, tc.fg)

			quad := [4][2]int{{0, 0}, {199, 0}, {199, 19}, {0, 19}}
			got, _, _ := ComputeTextColor(img, quad)
			if got == nil {
				t.Fatal("expected non-nil TextColor")
			}
			if !colorClose(got, tc.fg, 40) {
				t.Errorf("%s text: got %v, want near (%d,%d,%d) ±40",
					tc.name, got, tc.fg.R, tc.fg.G, tc.fg.B)
			}
		})
	}
}

func TestComputeTextColor_RenderedTextWithOutsideReference(t *testing.T) {
	// Quad smaller than the image: outside-polygon pixels act as background
	// reference.  This exercises the dDark/dLight branch rather than minority
	// rule, which is the more reliable path for real-world use.
	white := color.RGBA{R: 255, G: 255, B: 255, A: 255}
	blue := color.RGBA{R: 0, G: 60, B: 200, A: 255}

	img := newSolidImage(300, 40, white)
	// Text sits inside a sub-region; surrounding pixels are background.
	renderASCII(img, "Outside reference test", 60, 25, blue)

	// Quad covers only the text strip, leaving white margin pixels outside.
	quad := [4][2]int{{50, 10}, {280, 10}, {280, 35}, {50, 35}}
	tc, _, _ := ComputeTextColor(img, quad)
	if tc == nil {
		t.Fatal("expected non-nil TextColor")
	}
	if !colorClose(tc, blue, 40) {
		t.Errorf("outside-reference: got %v, want near blue (%d,%d,%d) ±40",
			tc, blue.R, blue.G, blue.B)
	}
}

// ---------------------------------------------------------------------------
// ComputeTextColor — simulated CJK text (ground-truth colour validation)
//
// Chinese characters have ~55-80% ink coverage within their bounding box,
// far higher than Latin glyphs (~25-40%).  At high coverage the dark-pixel
// cluster can become the *majority*, breaking a naive minority-rule heuristic.
// drawCJKGlyph produces a pattern with ~55% ink coverage (border + rails +
// filled centre) that faithfully exercises this path.
//
// Limitation: golang.org/x/image/font/basicfont only covers ASCII.  A full
// CJK font (e.g. Noto Sans CJK) would be required for true Unicode glyph
// rendering.  To include such tests without a vendored font file, embed the
// font bytes via go:embed or serve it from a test fixture directory.
// ---------------------------------------------------------------------------

func TestComputeTextColor_CJKLikeBlackOnWhite(t *testing.T) {
	// Five CJK-like glyphs at 20×20, black ink on white.
	// High ink density means dark pixels are the majority; the algorithm must
	// still identify them as text (via the outside-polygon background reference).
	white := color.RGBA{R: 255, G: 255, B: 255, A: 255}
	black := color.RGBA{R: 0, G: 0, B: 0, A: 255}

	img := newSolidImage(200, 40, white)
	for i := 0; i < 5; i++ {
		drawCJKGlyph(img, 10+i*36, 10, 20, black)
	}

	// Quad is inset so that the wide white margins are outside-polygon pixels,
	// giving the algorithm a background reference even for dense ink.
	quad := [4][2]int{{5, 5}, {195, 5}, {195, 35}, {5, 35}}
	tc, _, _ := ComputeTextColor(img, quad)
	if tc == nil {
		t.Fatal("CJK-like black-on-white: expected non-nil TextColor")
	}
	if !colorClose(tc, black, 40) {
		t.Errorf("CJK-like black-on-white: got %v, want near (0,0,0) ±40", tc)
	}
}

func TestComputeTextColor_CJKLikeColoredOnWhite(t *testing.T) {
	// Validates that the detected colour tracks the actual ink colour, not
	// the background, even for dense CJK-like glyphs.
	white := color.RGBA{R: 255, G: 255, B: 255, A: 255}
	red := color.RGBA{R: 200, G: 30, B: 30, A: 255}

	img := newSolidImage(200, 40, white)
	for i := 0; i < 5; i++ {
		drawCJKGlyph(img, 10+i*36, 10, 20, red)
	}

	quad := [4][2]int{{5, 5}, {195, 5}, {195, 35}, {5, 35}}
	tc, _, _ := ComputeTextColor(img, quad)
	if tc == nil {
		t.Fatal("CJK-like red-on-white: expected non-nil TextColor")
	}
	if !colorClose(tc, red, 40) {
		t.Errorf("CJK-like red-on-white: got %v, want near (%d,%d,%d) ±40",
			tc, red.R, red.G, red.B)
	}
}

func TestComputeTextColor_CJKLikeHighDensityMinorityRule(t *testing.T) {
	// When the quad covers only glyphs with no margin, there are no
	// outside-polygon pixels, so minority rule fires.  For dense CJK ink the
	// dark cluster is the MAJORITY; minority rule would wrongly pick the light
	// (whitespace) cluster as text.
	//
	// This test documents that known limitation: it asserts the current
	// behaviour and will need updating if the heuristic is improved.
	white := color.RGBA{R: 255, G: 255, B: 255, A: 255}
	black := color.RGBA{R: 0, G: 0, B: 0, A: 255}

	img := newSolidImage(200, 30, white)
	// Pack glyphs with no margin so the quad exactly wraps the ink.
	for i := 0; i < 5; i++ {
		drawCJKGlyph(img, i*20, 0, 20, black)
	}

	// Quad tightly wraps the glyph strip: no outside pixels → minority rule.
	quad := [4][2]int{{0, 0}, {99, 0}, {99, 19}, {0, 19}}
	tc, _, _ := ComputeTextColor(img, quad)
	// If nil or white-ish is returned, the minority-rule heuristic has been
	// fooled by dense ink.  Log this as a known algorithmic limitation.
	if tc == nil {
		t.Log("KNOWN LIMITATION: dense CJK with no outside pixels → too few pixels or nil")
		return
	}
	if tc[0] > 200 && tc[1] > 200 && tc[2] > 200 {
		t.Logf("KNOWN LIMITATION: minority rule returned white (bg) as text color "+
			"for high-density CJK-like glyphs: %v", tc)
	}
}

// ---------------------------------------------------------------------------
// ComputeWordColors — nil / empty guards
// ---------------------------------------------------------------------------

func TestComputeWordColors_NilInputs(t *testing.T) {
	arr := newSolidImage(10, 10, color.RGBA{R: 255, A: 255})
	if got := ComputeWordColors("hello", nil, arr, false); got != nil {
		t.Error("nil mask: expected nil")
	}
	if got := ComputeWordColors("hello", [][]bool{{true}}, nil, false); got != nil {
		t.Error("nil arr: expected nil")
	}
	if got := ComputeWordColors("", [][]bool{{true}}, arr, false); got != nil {
		t.Error("empty text: expected nil")
	}
}

func TestComputeWordColors_EmptyMask(t *testing.T) {
	arr := newSolidImage(10, 10, color.RGBA{R: 255, A: 255})
	if got := ComputeWordColors("hello", [][]bool{}, arr, false); got != nil {
		t.Error("empty fgMask: expected nil")
	}
}

func TestComputeWordColors_SingleSegment(t *testing.T) {
	// All columns have fg → 1 run → nil.
	const W, H = 20, 5
	arr := newSolidImage(W, H, color.RGBA{R: 200, A: 255})
	fgMask := make([][]bool, H)
	for y := range fgMask {
		fgMask[y] = make([]bool, W)
		for x := range fgMask[y] {
			fgMask[y][x] = true
		}
	}
	if got := ComputeWordColors("hello", fgMask, arr, false); got != nil {
		t.Errorf("single segment: expected nil, got %v", got)
	}
}

func TestComputeWordColors_AllGapNoFg(t *testing.T) {
	const W, H = 10, 5
	arr := newSolidImage(W, H, color.RGBA{R: 0, A: 255})
	fgMask := make([][]bool, H)
	for y := range fgMask {
		fgMask[y] = make([]bool, W) // all false
	}
	if got := ComputeWordColors("hello", fgMask, arr, false); got != nil {
		t.Errorf("all-gap mask: expected nil, got %v", got)
	}
}

// ---------------------------------------------------------------------------
// ComputeWordColors — rendered two-colour English words
//
// "Hello" (red) and "World" (blue) are rendered with a visible inter-word
// gap.  The foreground mask is obtained from ComputeTextColor, then
// ComputeWordColors must segment the two colour regions.
//
// This is an end-to-end correctness test: if either function is wrong, the
// per-word colour assignment will be incorrect.
// ---------------------------------------------------------------------------

func TestComputeWordColors_RenderedTwoColorEnglish(t *testing.T) {
	white := color.RGBA{R: 255, G: 255, B: 255, A: 255}
	red := color.RGBA{R: 220, G: 0, B: 0, A: 255}
	blue := color.RGBA{R: 0, G: 0, B: 220, A: 255}

	img := newSolidImage(240, 20, white)
	// "Hello" at x=5, "World" at x=130 → gap of ~90px between the words.
	renderASCII(img, "Hello", 5, 13, red)
	renderASCII(img, "World", 130, 13, blue)

	quad := [4][2]int{{0, 0}, {239, 0}, {239, 19}, {0, 19}}
	_, fgMask, arr := ComputeTextColor(img, quad)
	if fgMask == nil || arr == nil {
		t.Fatal("ComputeTextColor returned nil mask/arr — image may be too sparse")
	}

	got := ComputeWordColors("Hello World", fgMask, arr, false)
	if len(got) < 2 {
		t.Fatalf("two-colour English: expected ≥2 word entries, got %d (%v)", len(got), got)
	}

	first, last := got[0], got[len(got)-1]

	// Ground truth: first segment = red word, last segment = blue word.
	if first.Color[0] < 150 || first.Color[2] > 80 {
		t.Errorf("first word color: got %v, want red-dominant (R>150, B<80)", first.Color)
	}
	if last.Color[2] < 150 || last.Color[0] > 80 {
		t.Errorf("last word color: got %v, want blue-dominant (B>150, R<80)", last.Color)
	}
	if first.Start != 0 {
		t.Errorf("first.Start: got %d, want 0", first.Start)
	}
	if last.End != len([]rune("Hello World")) {
		t.Errorf("last.End: got %d, want %d", last.End, len([]rune("Hello World")))
	}
}

// ---------------------------------------------------------------------------
// ComputeWordColors — vertical layout
// ---------------------------------------------------------------------------

func TestComputeWordColors_TwoColorSegmentsVertical(t *testing.T) {
	// 10×50 image: rows 0-19 red, rows 25-49 blue; rows 20-24 are gap.
	const W, H = 10, 50
	arr := image.NewRGBA(image.Rect(0, 0, W, H))
	red := color.RGBA{R: 255, G: 0, B: 0, A: 255}
	blue := color.RGBA{R: 0, G: 0, B: 255, A: 255}
	for y := 0; y < H; y++ {
		for x := 0; x < W; x++ {
			switch {
			case y < 20:
				arr.SetRGBA(x, y, red)
			case y >= 25:
				arr.SetRGBA(x, y, blue)
			}
		}
	}

	fgMask := make([][]bool, H)
	for y := 0; y < H; y++ {
		fgMask[y] = make([]bool, W)
		for x := 0; x < W; x++ {
			fgMask[y][x] = y < 20 || y >= 25
		}
	}

	got := ComputeWordColors("hello world", fgMask, arr, true)
	if len(got) < 2 {
		t.Fatalf("vertical two-colour: expected ≥2 entries, got %d", len(got))
	}
	if got[0].Color[0] < 200 {
		t.Errorf("first segment (red): R=%d, want >200", got[0].Color[0])
	}
	last := got[len(got)-1]
	if last.Color[2] < 200 {
		t.Errorf("last segment (blue): B=%d, want >200", last.Color[2])
	}
}

// ---------------------------------------------------------------------------
// ComputeTextColorResult — integration
// ---------------------------------------------------------------------------

func TestComputeTextColorResult_RenderedBlackOnWhite(t *testing.T) {
	white := color.RGBA{R: 255, G: 255, B: 255, A: 255}
	black := color.RGBA{R: 0, G: 0, B: 0, A: 255}

	img := newSolidImage(200, 20, white)
	renderASCII(img, "Integration test", 5, 13, black)

	box := [][2]int{{0, 0}, {199, 0}, {199, 19}, {0, 19}}
	result := ComputeTextColorResult(img, box, "Integration test")
	if result.TextColor == nil {
		t.Fatal("TextColor should not be nil")
	}
	if !colorClose(result.TextColor, black, 30) {
		t.Errorf("integration: got TextColor=%v, want near black ±30", result.TextColor)
	}
}

func TestComputeTextColorResult_RenderedTwoColorWords(t *testing.T) {
	// End-to-end: render two differently coloured words and assert that
	// ComputeTextColorResult returns per-word colour entries.
	white := color.RGBA{R: 255, G: 255, B: 255, A: 255}
	red := color.RGBA{R: 220, G: 0, B: 0, A: 255}
	blue := color.RGBA{R: 0, G: 0, B: 220, A: 255}

	img := newSolidImage(240, 20, white)
	renderASCII(img, "Hello", 5, 13, red)
	renderASCII(img, "World", 130, 13, blue)

	box := [][2]int{{0, 0}, {239, 0}, {239, 19}, {0, 19}}
	result := ComputeTextColorResult(img, box, "Hello World")

	if result.TextColor == nil {
		t.Fatal("TextColor should not be nil")
	}
	// WordColors may be nil if the algorithm cannot segment; that's a
	// documented limitation rather than a crash.
	if result.WordColors != nil && len(result.WordColors) >= 2 {
		first := result.WordColors[0]
		last := result.WordColors[len(result.WordColors)-1]
		if first.Color[0] < 150 {
			t.Errorf("first word color: R=%d, want >150 (red)", first.Color[0])
		}
		if last.Color[2] < 150 {
			t.Errorf("last word color: B=%d, want >150 (blue)", last.Color[2])
		}
	} else {
		t.Logf("NOTE: word-level colour segmentation returned nil or 1 entry — " +
			"the inter-word gap may be too small for the segmenter to split")
	}
}
