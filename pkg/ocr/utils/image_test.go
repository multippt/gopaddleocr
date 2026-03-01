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
