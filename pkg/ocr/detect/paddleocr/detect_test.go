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
	BaseModelConfig: common.BaseModelConfig{},
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
// connectedComponents tests
// ---------------------------------------------------------------------------

func TestConnectedComponents_Empty(t *testing.T) {
	mask := make([]bool, 5*5) // all false
	comps := connectedComponents(mask, 5, 5)
	if len(comps) != 0 {
		t.Errorf("expected 0 components, got %d", len(comps))
	}
}

func TestConnectedComponents_Single(t *testing.T) {
	// 10×10 mask with a single 3×3 blob at (2,2)→(4,4).
	W, H := 10, 10
	mask := make([]bool, H*W)
	for y := 2; y <= 4; y++ {
		for x := 2; x <= 4; x++ {
			mask[y*W+x] = true
		}
	}
	comps := connectedComponents(mask, H, W)
	if len(comps) != 1 {
		t.Fatalf("expected 1 component, got %d", len(comps))
	}
	if len(comps[0]) != 9 {
		t.Errorf("component pixel count=%d want 9", len(comps[0]))
	}
}

func TestConnectedComponents_Two(t *testing.T) {
	// Two separated 3×3 blobs with a gap of at least 1 pixel.
	W, H := 20, 10
	mask := make([]bool, H*W)
	// Blob 1: cols 1–3, rows 1–3
	for y := 1; y <= 3; y++ {
		for x := 1; x <= 3; x++ {
			mask[y*W+x] = true
		}
	}
	// Blob 2: cols 10–12, rows 1–3 (gap of 6 columns)
	for y := 1; y <= 3; y++ {
		for x := 10; x <= 12; x++ {
			mask[y*W+x] = true
		}
	}
	comps := connectedComponents(mask, H, W)
	if len(comps) != 2 {
		t.Errorf("expected 2 components, got %d", len(comps))
	}
	for i, c := range comps {
		if len(c) != 9 {
			t.Errorf("component %d: pixel count=%d want 9", i, len(c))
		}
	}
}

// ---------------------------------------------------------------------------
// convexHull tests
// ---------------------------------------------------------------------------

func TestConvexHull_Triangle(t *testing.T) {
	pts := []fPoint{{0, 0}, {5, 0}, {2.5, 5}}
	hull := convexHull(pts)
	if len(hull) != 3 {
		t.Errorf("expected 3-point hull for triangle, got %d", len(hull))
	}
}

func TestConvexHull_Collinear(t *testing.T) {
	// All points on y=0 — hull should not panic.
	pts := []fPoint{{0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}}
	_ = convexHull(pts) // must not panic
}

// ---------------------------------------------------------------------------
// boxScore tests
// ---------------------------------------------------------------------------

func TestBoxScore_FullRect(t *testing.T) {
	// 20×20 prob map filled with 0.8; polygon covers a 10×10 interior region.
	W := 20
	prob := make([]float32, 20*W)
	for i := range prob {
		prob[i] = 0.8
	}
	poly := [][2]float64{{2, 2}, {12, 2}, {12, 12}, {2, 12}}
	score := boxScore(prob, W, poly)
	if score < 0.75 || score > 0.85 {
		t.Errorf("boxScore=%f want ≈0.8", score)
	}
}

func TestBoxScore_EmptyPoly(t *testing.T) {
	prob := make([]float32, 20*20)
	score := boxScore(prob, 20, nil)
	if score != 0 {
		t.Errorf("boxScore for empty poly=%f want 0", score)
	}
}

// ---------------------------------------------------------------------------
// lineLineIntersect tests
// ---------------------------------------------------------------------------

func TestLineLineIntersect_Perpendicular(t *testing.T) {
	// Horizontal line y=3: (0,3)→(10,3); Vertical line x=5: (5,0)→(5,10).
	p := lineLineIntersect([2]float64{0, 3}, [2]float64{10, 3}, [2]float64{5, 0}, [2]float64{5, 10})
	if math.Abs(p[0]-5) > 0.01 || math.Abs(p[1]-3) > 0.01 {
		t.Errorf("intersection=%v want [5,3]", p)
	}
}

func TestLineLineIntersect_Parallel(t *testing.T) {
	// Two parallel horizontal lines — should return midpoint, no panic.
	p := lineLineIntersect([2]float64{0, 0}, [2]float64{10, 0}, [2]float64{0, 5}, [2]float64{10, 5})
	// Midpoint of p2 and p3 end-points (fallback): ((10+0)/2, (0+5)/2) = (5, 2.5)
	_ = p // just must not panic
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

func TestDetPostprocess_TinyRegionFiltered(t *testing.T) {
	// A 3×3 blob (9 pixels) is below MinArea=16 and must be filtered out.
	padH, padW := 64, 64
	prob := make([]float32, padH*padW)
	for y := 10; y <= 12; y++ {
		for x := 10; x <= 12; x++ {
			prob[y*padW+x] = 0.8
		}
	}
	m := &Model{config: testDetConfig}
	boxes := m.postprocess(prob, padH, padW, padH, padW, padH, padW)
	if len(boxes) != 0 {
		t.Errorf("expected 0 boxes for tiny region, got %d", len(boxes))
	}
}

func TestDetPostprocess_MultipleRegions(t *testing.T) {
	// Two separated horizontal bands → 2 boxes.
	padH, padW := 128, 128
	prob := make([]float32, padH*padW)
	// Band 1: rows 10–18, cols 5–100
	for y := 10; y <= 18; y++ {
		for x := 5; x <= 100; x++ {
			prob[y*padW+x] = 0.9
		}
	}
	// Band 2: rows 50–58, cols 5–100 (28 row gap — no 4-connectivity)
	for y := 50; y <= 58; y++ {
		for x := 5; x <= 100; x++ {
			prob[y*padW+x] = 0.9
		}
	}
	m := &Model{config: testDetConfig}
	boxes := m.postprocess(prob, padH, padW, padH, padW, padH, padW)
	if len(boxes) != 2 {
		t.Errorf("expected 2 boxes for two separated bands, got %d", len(boxes))
	}
}

func TestDetPostprocess_BelowBoxThresh(t *testing.T) {
	// Region pixels above Thresh (0.3) but mean below BoxThresh (0.6) → 0 boxes.
	padH, padW := 64, 64
	prob := make([]float32, padH*padW)
	// 6×6 = 36 pixels (> MinArea=16); prob=0.4 → above Thresh but below BoxThresh.
	for y := 10; y <= 15; y++ {
		for x := 10; x <= 15; x++ {
			prob[y*padW+x] = 0.4
		}
	}
	m := &Model{config: testDetConfig}
	boxes := m.postprocess(prob, padH, padW, padH, padW, padH, padW)
	if len(boxes) != 0 {
		t.Errorf("expected 0 boxes when boxScore < BoxThresh, got %d", len(boxes))
	}
}
