package paddleocr

import (
	"math"
	"testing"

	"github.com/multippt/gopaddleocr/pkg/ocr/common"
)

// testDetConfig is used by tests that need a ModelConfig without importing ocr (avoids cycle).
// Values match the defaults previously in DefaultConfig.
var testDetConfig = &ModelConfig{
	LimitSideLength: 1280,
	Mean:            [3]float32{0.485, 0.456, 0.406},
	Std:             [3]float32{0.229, 0.224, 0.225},
	Thresh:          0.3,
	BoxThresh:       0.6,
	UnclipRatio:     2.0,
	MinArea:         16,
	BaseModelConfig: common.BaseModelConfig{
		OnnxConfig: common.Config{
			InputName:  "x",
			OutputName: "fetch_name_0",
		},
	},
}

// ---------------------------------------------------------------------------
// unclipPoly tests
// ---------------------------------------------------------------------------

func TestUnclipPolyExpands(t *testing.T) {
	// A simple 10×4 axis-aligned rectangle (CW in image coords: TL→TR→BR→BL).
	poly := [][2]float64{
		{0, 0},
		{10, 0},
		{10, 4},
		{0, 4},
	}
	area := 10.0 * 4.0        // 40
	perim := 2 * (10.0 + 4.0) // 28
	ratio := 1.5
	dist := area * ratio / perim // ≈ 2.143

	got := unclipPoly(poly, ratio)
	if len(got) != 4 {
		t.Fatalf("expected 4 points, got %d", len(got))
	}

	// Find axis-aligned bounding box of result.
	minX, minY := got[0][0], got[0][1]
	maxX, maxY := got[0][0], got[0][1]
	for _, p := range got[1:] {
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

	gotW := maxX - minX
	gotH := maxY - minY
	wantW := 10.0 + 2*dist
	wantH := 4.0 + 2*dist

	if math.Abs(gotW-wantW) > 0.5 {
		t.Errorf("width: got %.3f, want %.3f (dist=%.3f)", gotW, wantW, dist)
	}
	if math.Abs(gotH-wantH) > 0.5 {
		t.Errorf("height: got %.3f, want %.3f (dist=%.3f)", gotH, wantH, dist)
	}
}

func TestUnclipPolyTextRegion(t *testing.T) {
	// Simulate a typical text-line probability map region:
	// width=276, height=15 (in padded-map pixels).
	poly := [][2]float64{
		{0, 0},
		{276, 0},
		{276, 15},
		{0, 15},
	}
	got := unclipPoly(poly, testDetConfig.UnclipRatio)

	minY, maxY := got[0][1], got[0][1]
	for _, p := range got[1:] {
		if p[1] < minY {
			minY = p[1]
		}
		if p[1] > maxY {
			maxY = p[1]
		}
	}
	gotH := maxY - minY
	// Expected: 15 + 2*(276*15*1.5 / (2*(276+15))) ≈ 15 + 2*10.67 ≈ 36.3
	if gotH < 30 {
		t.Errorf("height %.2f too small — polygon likely shrunk instead of expanded", gotH)
	}
}

// ---------------------------------------------------------------------------
// minAreaRect tests
// ---------------------------------------------------------------------------

func TestMinAreaRect_Square(t *testing.T) {
	hull := []fPoint{{0, 0}, {4, 0}, {4, 4}, {0, 4}}
	rect := minAreaRect(hull)

	// AABB of returned corners should span (0,0)→(4,4).
	minX, minY := rect[0].X, rect[0].Y
	maxX, maxY := rect[0].X, rect[0].Y
	for _, p := range rect[1:] {
		if p.X < minX {
			minX = p.X
		}
		if p.X > maxX {
			maxX = p.X
		}
		if p.Y < minY {
			minY = p.Y
		}
		if p.Y > maxY {
			maxY = p.Y
		}
	}
	if math.Abs(maxX-minX-4) > 0.5 {
		t.Errorf("width %.2f != 4", maxX-minX)
	}
	if math.Abs(maxY-minY-4) > 0.5 {
		t.Errorf("height %.2f != 4", maxY-minY)
	}
}

func TestMinAreaRect_Rectangle(t *testing.T) {
	hull := []fPoint{{0, 0}, {20, 0}, {20, 5}, {0, 5}}
	rect := minAreaRect(hull)

	minX, minY := rect[0].X, rect[0].Y
	maxX, maxY := rect[0].X, rect[0].Y
	for _, p := range rect[1:] {
		if p.X < minX {
			minX = p.X
		}
		if p.X > maxX {
			maxX = p.X
		}
		if p.Y < minY {
			minY = p.Y
		}
		if p.Y > maxY {
			maxY = p.Y
		}
	}
	if math.Abs(maxX-minX-20) > 0.5 {
		t.Errorf("width %.2f != 20", maxX-minX)
	}
	if math.Abs(maxY-minY-5) > 0.5 {
		t.Errorf("height %.2f != 5", maxY-minY)
	}
}

// ---------------------------------------------------------------------------
// Full pipeline smoke test: synthesise a probability map and verify boxes
// ---------------------------------------------------------------------------

func TestDetPostprocess_SimpleRect(t *testing.T) {
	// Create a 64×64 padded probability map with one text-like region:
	// a horizontal band at rows 20-30 (height=10), cols 5-55 (width=50).
	padH, padW := 64, 64
	resH, resW := 64, 64
	origH, origW := 64, 64

	prob := make([]float32, padH*padW)
	for y := 20; y <= 30; y++ {
		for x := 5; x <= 55; x++ {
			prob[y*padW+x] = 0.8 // well above detThresh
		}
	}

	m := &Model{config: testDetConfig}
	boxes := m.postprocess(prob, padH, padW, resH, resW, origH, origW)
	if len(boxes) == 0 {
		t.Fatal("expected at least one box, got none")
	}

	q := boxes[0].Quad
	// Find bounding box of returned quad in original image coords.
	minX, minY := q[0][0], q[0][1]
	maxX, maxY := q[0][0], q[0][1]
	for _, p := range q[1:] {
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
	gotW := maxX - minX
	gotH := maxY - minY

	t.Logf("quad: %v", q)
	t.Logf("AABB: x=[%d,%d] y=[%d,%d] w=%d h=%d", minX, maxX, minY, maxY, gotW, gotH)

	// After unclipping, height should be larger than the raw region (10px).
	if gotH <= 10 {
		t.Errorf("height %d should be > 10 (unclip should expand)", gotH)
	}
	// Width should also be expanded.
	if gotW <= 50 {
		t.Errorf("width %d should be > 50 (unclip should expand)", gotW)
	}
}
