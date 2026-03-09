package utils

import (
	"image"
	"image/color"
	"math"
	"testing"
)

// ---------------------------------------------------------------------------
// rotate90CCW tests
// ---------------------------------------------------------------------------

// TestRotate90CCW_Dims checks that a W×H image becomes H×W after rotation.
func TestRotate90CCW_Dims(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 30, 120)) // portrait 30×120
	out := Rotate90CCW(img)
	b := out.Bounds()
	if b.Dx() != 120 || b.Dy() != 30 {
		t.Errorf("expected 120×30, got %d×%d", b.Dx(), b.Dy())
	}
}

// TestRotate90CCW_Pixels verifies the CCW pixel mapping with a tiny 2×3 grid.
//
// Input (W=2, H=3):
//
//	col: 0  1
//	     A  B   (row 0)
//	     C  D   (row 1)
//	     E  F   (row 2)
//
// Expected output after 90° CCW (W=3, H=2):
//
//	col: 0  1  2
//	     B  D  F   (row 0)
//	     A  C  E   (row 1)
//
// Derivation: output(ox, oy) = input(ix = W-1-oy, iy = ox)
func TestRotate90CCW_Pixels(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 2, 3))
	pixels := [3][2]color.RGBA{
		{{R: 10, A: 255}, {R: 20, A: 255}}, // row 0: A=10, B=20
		{{R: 30, A: 255}, {R: 40, A: 255}}, // row 1: C=30, D=40
		{{R: 50, A: 255}, {R: 60, A: 255}}, // row 2: E=50, F=60
	}
	for y := 0; y < 3; y++ {
		for x := 0; x < 2; x++ {
			img.SetRGBA(x, y, pixels[y][x])
		}
	}

	out := Rotate90CCW(img)

	// Output row 0: B D F  (ox=0..2, oy=0)
	wantRow0 := []uint8{20, 40, 60}
	// Output row 1: A C E  (ox=0..2, oy=1)
	wantRow1 := []uint8{10, 30, 50}

	for ox, want := range wantRow0 {
		got := out.RGBAAt(ox, 0).R
		if got != want {
			t.Errorf("row0 col%d: got R=%d, want R=%d", ox, got, want)
		}
	}
	for ox, want := range wantRow1 {
		got := out.RGBAAt(ox, 1).R
		if got != want {
			t.Errorf("row1 col%d: got R=%d, want R=%d", ox, got, want)
		}
	}
}

// TestRotate90CCW_TopBecomesLeft confirms the key semantic property:
// a column of pixels at input x=0 (left edge) appears at output y=H-1 (bottom)
// reading left-to-right == top-to-bottom order of the input.
//
// This matters for vertical text: top character → leftmost in output.
func TestRotate90CCW_TopBecomesLeft(t *testing.T) {
	const W, H = 1, 4
	img := image.NewRGBA(image.Rect(0, 0, W, H))
	// Single column, 4 rows: values 10, 20, 30, 40 top to bottom.
	for y := 0; y < H; y++ {
		img.SetRGBA(0, y, color.RGBA{R: uint8((y + 1) * 10), A: 255})
	}

	out := Rotate90CCW(img) // output is 4×1

	// After CCW on a 1-wide column, output row 0 should read: 10 20 30 40 (left→right).
	for ox := 0; ox < H; ox++ {
		want := uint8((ox + 1) * 10)
		got := out.RGBAAt(ox, 0).R
		if got != want {
			t.Errorf("output col %d: got R=%d, want R=%d", ox, got, want)
		}
	}
}

// ---------------------------------------------------------------------------
// Vertical text warp+rotate dimension tests
// ---------------------------------------------------------------------------

// TestVerticalTextWarpDims verifies that the warp→rotate pipeline for a
// vertical quad produces an image of dimensions (targetH × recHeight) without
// requiring any ONNX models.
func TestVerticalTextWarpDims(t *testing.T) {
	// Synthesise a plain grey source image large enough to hold the quad.
	src := image.NewRGBA(image.Rect(0, 0, 100, 300))
	for y := 0; y < 300; y++ {
		for x := 0; x < 100; x++ {
			src.SetRGBA(x, y, color.RGBA{R: 180, G: 180, B: 180, A: 255})
		}
	}

	// A vertical quad: 30px wide, 200px tall (srcH/srcW ≈ 6.67).
	quad := [4][2]float64{
		{10, 20},  // TL
		{40, 20},  // TR
		{40, 220}, // BR
		{10, 220}, // BL
	}

	srcW := PointDistance(quad[0], quad[1]) // ≈ 30
	srcH := PointDistance(quad[0], quad[3]) // ≈ 200

	if srcH <= srcW {
		t.Fatal("test quad should be vertical (H > W)")
	}

	recHeight := 48

	targetH := int(math.Round(float64(recHeight) * srcH / srcW))
	portrait := PerspectiveWarp(src, quad, recHeight, targetH)
	if portrait == nil {
		t.Fatal("perspectiveWarp returned nil for portrait crop")
	}
	// Portrait dimensions: recHeight wide, targetH tall.
	pb := portrait.Bounds()
	if pb.Dx() != recHeight {
		t.Errorf("portrait width: got %d, want %d (recHeight)", pb.Dx(), recHeight)
	}
	if pb.Dy() != targetH {
		t.Errorf("portrait height: got %d, want %d (targetH)", pb.Dy(), targetH)
	}

	rotated := Rotate90CCW(portrait)
	rb := rotated.Bounds()
	// After CCW: width = former height, height = former width.
	if rb.Dx() != targetH {
		t.Errorf("rotated width: got %d, want %d (targetH)", rb.Dx(), targetH)
	}
	if rb.Dy() != recHeight {
		t.Errorf("rotated height: got %d, want %d (recHeight)", rb.Dy(), recHeight)
	}
}

// TestHorizontalTextWarpDims verifies horizontal text still goes through the
// original path and produces the expected dimensions.
func TestHorizontalTextWarpDims(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 400, 100))
	// Horizontal quad: 200px wide, 30px tall.
	quad := [4][2]float64{
		{50, 30},  // TL
		{250, 30}, // TR
		{250, 60}, // BR
		{50, 60},  // BL
	}

	srcW := PointDistance(quad[0], quad[1]) // ≈ 200
	srcH := PointDistance(quad[0], quad[3]) // ≈ 30

	if srcH > srcW {
		t.Fatal("test quad should be horizontal (W > H)")
	}

	recHeight := 48

	targetW := int(math.Round(float64(recHeight) * srcW / srcH))
	warped := PerspectiveWarp(src, quad, targetW, recHeight)
	if warped == nil {
		t.Fatal("perspectiveWarp returned nil for horizontal crop")
	}
	wb := warped.Bounds()
	if wb.Dx() != targetW {
		t.Errorf("warped width: got %d, want %d", wb.Dx(), targetW)
	}
	if wb.Dy() != recHeight {
		t.Errorf("warped height: got %d, want %d", wb.Dy(), recHeight)
	}
}

// ---------------------------------------------------------------------------
// orderPoints4 tests
// ---------------------------------------------------------------------------

func TestOrderPoints4_Axis(t *testing.T) {
	// Axis-aligned rectangle corners in shuffled order.
	pts := [4][2]float64{
		{10, 0},  // TR
		{0, 10},  // BL
		{10, 10}, // BR
		{0, 0},   // TL
	}
	got := OrderPoints4(pts)
	tl := got[0]
	tr := got[1]
	br := got[2]
	bl := got[3]

	if tl != ([2]float64{0, 0}) {
		t.Errorf("TL wrong: %v", tl)
	}
	if tr != ([2]float64{10, 0}) {
		t.Errorf("TR wrong: %v", tr)
	}
	if br != ([2]float64{10, 10}) {
		t.Errorf("BR wrong: %v", br)
	}
	if bl != ([2]float64{0, 10}) {
		t.Errorf("BL wrong: %v", bl)
	}
}

func TestOrderPoints4_Rotated(t *testing.T) {
	// Slightly rotated rectangle — typical for detected text boxes.
	pts := [4][2]float64{
		{2, 0},
		{12, 1},
		{11, 5},
		{1, 4},
	}
	got := OrderPoints4(pts)
	// TL should be leftmost+topmost: (2,0) has min(x+y)=2
	// TR should be rightmost+topmost: (12,1) has max(x-y)=11
	// BR should be rightmost+bottommost: (11,5) has max(x+y)=16
	// BL should be leftmost+bottommost: (1,4) has min(x-y)=-3
	if got[0] != ([2]float64{2, 0}) {
		t.Errorf("TL wrong: %v", got[0])
	}
	if got[1] != ([2]float64{12, 1}) {
		t.Errorf("TR wrong: %v", got[1])
	}
	if got[2] != ([2]float64{11, 5}) {
		t.Errorf("BR wrong: %v", got[2])
	}
	if got[3] != ([2]float64{1, 4}) {
		t.Errorf("BL wrong: %v", got[3])
	}
}

// ---------------------------------------------------------------------------
// ColorRGBA
// ---------------------------------------------------------------------------

func TestColorRGBA(t *testing.T) {
	// color.RGBA.RGBA() returns 16-bit values (8-bit shifted left by 8).
	// ColorRGBA reverses that: uint8(v >> 8).
	c := ColorRGBA(0xFF00, 0x8000, 0x0100, 0xFF00)
	if c.R != 0xFF {
		t.Errorf("R: got %d, want %d", c.R, 0xFF)
	}
	if c.G != 0x80 {
		t.Errorf("G: got %d, want %d", c.G, 0x80)
	}
	if c.B != 0x01 {
		t.Errorf("B: got %d, want %d", c.B, 0x01)
	}
	if c.A != 0xFF {
		t.Errorf("A: got %d, want %d", c.A, 0xFF)
	}
	// Zero inputs → zero outputs.
	z := ColorRGBA(0, 0, 0, 0)
	if z != (color.RGBA{}) {
		t.Errorf("zero: got %v, want zero", z)
	}
}

// ---------------------------------------------------------------------------
// BilinearSample
// ---------------------------------------------------------------------------

func TestBilinearSample_Uniform(t *testing.T) {
	// A solid-colour image: any sample point should return that colour.
	img := image.NewRGBA(image.Rect(0, 0, 4, 4))
	fill := color.RGBA{R: 100, G: 150, B: 200, A: 255}
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			img.SetRGBA(x, y, fill)
		}
	}
	for _, pt := range [][2]float64{{0, 0}, {1.5, 1.5}, {3, 3}, {0.5, 2.7}} {
		got := BilinearSample(img, pt[0], pt[1])
		if got.R != fill.R || got.G != fill.G || got.B != fill.B {
			t.Errorf("sample(%v): got %v, want %v", pt, got, fill)
		}
	}
}

func TestBilinearSample_Interpolation(t *testing.T) {
	// 1×2 image (width=2, height=1): left pixel R=0, right pixel R=254.
	// Sample at x=0.5 should give R≈127.
	img := image.NewRGBA(image.Rect(0, 0, 2, 1))
	img.SetRGBA(0, 0, color.RGBA{R: 0, A: 255})
	img.SetRGBA(1, 0, color.RGBA{R: 254, A: 255})
	got := BilinearSample(img, 0.5, 0)
	// lerp: (0*0.5 + 65278*0.5) / 257 ≈ 127
	if got.R < 125 || got.R > 129 {
		t.Errorf("mid-interpolation R: got %d, want ~127", got.R)
	}
}

// ---------------------------------------------------------------------------
// BilinearResize
// ---------------------------------------------------------------------------

func TestBilinearResize_Dims(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 100, 80))
	out := BilinearResize(src, 50, 40)
	b := out.Bounds()
	if b.Dx() != 50 || b.Dy() != 40 {
		t.Errorf("expected 50×40, got %d×%d", b.Dx(), b.Dy())
	}
}

func TestBilinearResize_Uniform(t *testing.T) {
	// Resizing a solid-colour image should preserve the colour.
	src := image.NewRGBA(image.Rect(0, 0, 10, 10))
	fill := color.RGBA{R: 80, G: 160, B: 240, A: 255}
	for y := 0; y < 10; y++ {
		for x := 0; x < 10; x++ {
			src.SetRGBA(x, y, fill)
		}
	}
	out := BilinearResize(src, 5, 5)
	for y := 0; y < 5; y++ {
		for x := 0; x < 5; x++ {
			got := out.RGBAAt(x, y)
			if got.R != fill.R || got.G != fill.G || got.B != fill.B {
				t.Errorf("pixel(%d,%d): got %v, want %v", x, y, got, fill)
			}
		}
	}
}

// ---------------------------------------------------------------------------
// ImageToNCHW / ImageToNCHWFromImage
// ---------------------------------------------------------------------------

func TestImageToNCHW(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 2, 2))
	img.SetRGBA(0, 0, color.RGBA{R: 255, G: 0, B: 0, A: 255})
	img.SetRGBA(1, 0, color.RGBA{R: 0, G: 255, B: 0, A: 255})
	img.SetRGBA(0, 1, color.RGBA{R: 0, G: 0, B: 255, A: 255})
	img.SetRGBA(1, 1, color.RGBA{R: 128, G: 128, B: 128, A: 255})

	mean := [3]float64{0, 0, 0}
	std := [3]float64{1, 1, 1}
	data := ImageToNCHW(img, 2, 2, mean, std)

	if len(data) != 3*2*2 {
		t.Fatalf("len: got %d, want %d", len(data), 3*2*2)
	}
	// R channel pixel(0,0): (255/255 - 0) / 1 = 1.0
	if math.Abs(float64(data[0])-1.0) > 0.01 {
		t.Errorf("R[0,0]: got %f, want 1.0", data[0])
	}
	// G channel pixel(0,0): 0.0
	if math.Abs(float64(data[4])) > 0.01 {
		t.Errorf("G[0,0]: got %f, want 0.0", data[4])
	}
	// B channel pixel(0,0): 0.0
	if math.Abs(float64(data[8])) > 0.01 {
		t.Errorf("B[0,0]: got %f, want 0.0", data[8])
	}
}

func TestImageToNCHWFromImage(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 3, 3))
	data := ImageToNCHWFromImage(img, 3, 3, [3]float64{0, 0, 0}, [3]float64{1, 1, 1})
	if len(data) != 3*3*3 {
		t.Fatalf("len: got %d, want %d", len(data), 3*3*3)
	}
}

// ---------------------------------------------------------------------------
// PointDistance
// ---------------------------------------------------------------------------

func TestPointDistance(t *testing.T) {
	tests := []struct {
		a, b [2]float64
		want float64
	}{
		{[2]float64{0, 0}, [2]float64{3, 4}, 5},
		{[2]float64{0, 0}, [2]float64{0, 0}, 0},
		{[2]float64{1, 1}, [2]float64{1, 1}, 0},
		{[2]float64{-3, 0}, [2]float64{0, 4}, 5},
	}
	for _, tc := range tests {
		got := PointDistance(tc.a, tc.b)
		if math.Abs(got-tc.want) > 1e-9 {
			t.Errorf("PointDistance(%v, %v) = %f, want %f", tc.a, tc.b, got, tc.want)
		}
	}
}

// ---------------------------------------------------------------------------
// PointInPolyF / PointInQuad4
// ---------------------------------------------------------------------------

func TestPointInPolyF(t *testing.T) {
	// Axis-aligned square [0,10]×[0,10].
	square := [][2]float64{{0, 0}, {10, 0}, {10, 10}, {0, 10}}

	if !PointInPolyF(5, 5, square) {
		t.Error("centre (5,5) should be inside square")
	}
	if PointInPolyF(15, 5, square) {
		t.Error("(15,5) should be outside square")
	}
	if PointInPolyF(-1, 5, square) {
		t.Error("(-1,5) should be outside square")
	}

	// Triangle: (0,0), (10,0), (5,10).
	tri := [][2]float64{{0, 0}, {10, 0}, {5, 10}}
	if !PointInPolyF(5, 5, tri) {
		t.Error("(5,5) should be inside triangle")
	}
	if PointInPolyF(0, 9, tri) {
		t.Error("(0,9) should be outside triangle")
	}
}

func TestPointInQuad4(t *testing.T) {
	q := [4][2]float64{{0, 0}, {10, 0}, {10, 10}, {0, 10}}
	if !PointInQuad4(5, 5, q) {
		t.Error("centre should be inside quad")
	}
	if PointInQuad4(20, 20, q) {
		t.Error("(20,20) should be outside quad")
	}
}

// ---------------------------------------------------------------------------
// FloatQuad
// ---------------------------------------------------------------------------

func TestFloatQuad(t *testing.T) {
	q := [4][2]int{{1, 2}, {3, 4}, {5, 6}, {7, 8}}
	got := FloatQuad(q)
	for i := 0; i < 4; i++ {
		if got[i][0] != float64(q[i][0]) || got[i][1] != float64(q[i][1]) {
			t.Errorf("point %d: got %v, want %v", i, got[i], q[i])
		}
	}
}

// ---------------------------------------------------------------------------
// PerspectiveWarp edge cases
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// PerspectiveWarp pixel-level tests
// ---------------------------------------------------------------------------

func TestPerspectiveWarp_AxisAligned_PixelValues(t *testing.T) {
	// 10×4 image with R=x*20, G=y*50.
	src := image.NewRGBA(image.Rect(0, 0, 10, 4))
	for y := 0; y < 4; y++ {
		for x := 0; x < 10; x++ {
			src.SetRGBA(x, y, color.RGBA{R: uint8(x * 20), G: uint8(y * 50), A: 255})
		}
	}
	// Axis-aligned quad covering the entire image.
	quad := [4][2]float64{{0, 0}, {9, 0}, {9, 3}, {0, 3}}
	out := PerspectiveWarp(src, quad, 10, 4)
	if out == nil {
		t.Fatal("PerspectiveWarp returned nil")
	}
	b := out.Bounds()
	if b.Dx() != 10 || b.Dy() != 4 {
		t.Fatalf("expected 10×4, got %d×%d", b.Dx(), b.Dy())
	}
	// Corner (0,0): R≈0, G≈0.
	c00 := out.RGBAAt(0, 0)
	if math.Abs(float64(c00.R)-0) > 2 || math.Abs(float64(c00.G)-0) > 2 {
		t.Errorf("(0,0): got R=%d G=%d, want R≈0 G≈0", c00.R, c00.G)
	}
	// Corner (9,0): R≈180, G≈0.
	c90 := out.RGBAAt(9, 0)
	if math.Abs(float64(c90.R)-180) > 2 || math.Abs(float64(c90.G)-0) > 2 {
		t.Errorf("(9,0): got R=%d G=%d, want R≈180 G≈0", c90.R, c90.G)
	}
	// Corner (0,3): R≈0, G≈150.
	c03 := out.RGBAAt(0, 3)
	if math.Abs(float64(c03.R)-0) > 2 || math.Abs(float64(c03.G)-150) > 2 {
		t.Errorf("(0,3): got R=%d G=%d, want R≈0 G≈150", c03.R, c03.G)
	}
}

func TestPerspectiveWarp_TrapezoidalQuad(t *testing.T) {
	// 100×100 solid green image.
	src := image.NewRGBA(image.Rect(0, 0, 100, 100))
	for y := 0; y < 100; y++ {
		for x := 0; x < 100; x++ {
			src.SetRGBA(x, y, color.RGBA{R: 0, G: 200, B: 0, A: 255})
		}
	}
	quad := [4][2]float64{{10, 10}, {80, 20}, {75, 70}, {15, 65}}
	out := PerspectiveWarp(src, quad, 60, 50)
	if out == nil {
		t.Fatal("PerspectiveWarp returned nil for trapezoidal quad")
	}
	b := out.Bounds()
	if b.Dx() != 60 || b.Dy() != 50 {
		t.Errorf("expected 60×50, got %d×%d", b.Dx(), b.Dy())
	}
	// Centre pixel should come from inside the green region (G > 0).
	mid := out.RGBAAt(30, 25)
	if mid.G == 0 {
		t.Errorf("centre pixel G=%d, expected non-zero (interior green pixel)", mid.G)
	}
}

func TestBilinearSample_EdgePixel(t *testing.T) {
	// 3×3 image; corner (2,2) has a distinct red value.
	img := image.NewRGBA(image.Rect(0, 0, 3, 3))
	img.SetRGBA(2, 2, color.RGBA{R: 200, A: 255})
	// Sample very close to (2,2) but within bounds.
	got := BilinearSample(img, 2.9, 2.9)
	// Must not panic; R should be dominated by the corner value.
	if got.R < 100 {
		t.Errorf("edge pixel R=%d, want ≥100 (corner value dominates at x=2.9,y=2.9)", got.R)
	}
}

func TestImageToNCHW_NormalizationValues(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 1, 1))
	img.SetRGBA(0, 0, color.RGBA{R: 127, G: 0, B: 255, A: 255})
	mean := [3]float64{0.5, 0.5, 0.5}
	std := [3]float64{0.5, 0.5, 0.5}
	data := ImageToNCHW(img, 1, 1, mean, std)
	if len(data) != 3 {
		t.Fatalf("expected 3 values, got %d", len(data))
	}
	// R: (127/255 - 0.5)/0.5 ≈ -0.004
	if math.Abs(float64(data[0])-(-0.004)) > 0.01 {
		t.Errorf("R: got %f, want ≈-0.004", data[0])
	}
	// G: (0/255 - 0.5)/0.5 = -1.0
	if math.Abs(float64(data[1])-(-1.0)) > 0.01 {
		t.Errorf("G: got %f, want ≈-1.0", data[1])
	}
	// B: (255/255 - 0.5)/0.5 = 1.0
	if math.Abs(float64(data[2])-1.0) > 0.01 {
		t.Errorf("B: got %f, want ≈1.0", data[2])
	}
}

func TestPerspectiveWarp_InvalidDims(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 100, 100))
	quad := [4][2]float64{{10, 10}, {90, 10}, {90, 90}, {10, 90}}
	if got := PerspectiveWarp(src, quad, 0, 50); got == nil || got.Bounds().Dx() != 1 {
		t.Error("dstW=0 should return a 1-wide image")
	}
	if got := PerspectiveWarp(src, quad, 50, 0); got == nil || got.Bounds().Dy() != 1 {
		t.Error("dstH=0 should return a 1-tall image")
	}
	if got := PerspectiveWarp(src, quad, -1, 50); got == nil || got.Bounds().Dx() != 1 {
		t.Error("dstW<0 should return a 1-wide image")
	}
}
