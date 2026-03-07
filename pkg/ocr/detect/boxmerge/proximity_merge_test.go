package boxmerge

import (
	"errors"
	"image"
	"testing"

	"github.com/multippt/gopaddleocr/pkg/ocr/utils"
)

// ---------------------------------------------------------------------------
// ProximityMergeStrategy — clusterBoxes
// ---------------------------------------------------------------------------

func TestProximity_EmptyInput(t *testing.T) {
	s := &ProximityMergeStrategy{config: testConfig}
	if result := s.clusterBoxes(nil); result != nil {
		t.Fatalf("expected nil, got %v", result)
	}
}

func TestProximity_SingleBox(t *testing.T) {
	s := &ProximityMergeStrategy{config: testConfig}
	box := makeBox(0, 0, 100, 10)
	result := s.clusterBoxes([]utils.Box{box})
	if len(result) != 1 {
		t.Fatalf("expected 1 parent, got %d", len(result))
	}
	if len(result[0].Children) != 1 {
		t.Fatalf("expected 1 child, got %d", len(result[0].Children))
	}
	if result[0].Quad != box.Quad {
		t.Errorf("parent quad = %v, want %v", result[0].Quad, box.Quad)
	}
}

func TestProximity_MergeCloseBoxes(t *testing.T) {
	s := &ProximityMergeStrategy{config: testConfig}
	// Gap in Y = 12 - 10 = 2 <= MaxMergeDistance(10) → merge
	boxes := []utils.Box{
		makeBox(0, 0, 100, 10),
		makeBox(0, 12, 100, 10),
	}
	result := s.clusterBoxes(boxes)
	if len(result) != 1 {
		t.Fatalf("expected 1 parent, got %d", len(result))
	}
	if len(result[0].Children) != 2 {
		t.Fatalf("expected 2 children, got %d", len(result[0].Children))
	}
}

func TestProximity_NoMergeFarBoxes(t *testing.T) {
	s := &ProximityMergeStrategy{config: testConfig}
	// Gap in Y = 25 - 10 = 15 > MaxMergeDistance(10) → no merge
	boxes := []utils.Box{
		makeBox(0, 0, 100, 10),
		makeBox(0, 25, 100, 10),
	}
	result := s.clusterBoxes(boxes)
	if len(result) != 2 {
		t.Fatalf("expected 2 parents, got %d", len(result))
	}
}

func TestProximity_NoMergeOrientMismatch(t *testing.T) {
	s := &ProximityMergeStrategy{config: testConfig}
	boxes := []utils.Box{
		makeBox(0, 0, 100, 10),     // horizontal (w > h)
		makeVertBox(5, 2, 10, 100), // vertical (h > w)
	}
	result := s.clusterBoxes(boxes)
	if len(result) != 2 {
		t.Fatalf("expected 2 parents (no cross-orient merge), got %d", len(result))
	}
}

func TestProximity_NoMergeSizeRatio(t *testing.T) {
	s := &ProximityMergeStrategy{config: testConfig}
	// textSize1=10, textSize2=17 → ratio 1.7 > MaxSizeRatio(1.5)
	boxes := []utils.Box{
		makeBox(0, 0, 100, 10),
		makeBox(0, 12, 100, 17),
	}
	result := s.clusterBoxes(boxes)
	if len(result) != 2 {
		t.Fatalf("expected 2 parents (size ratio too large), got %d", len(result))
	}
}

func TestProximity_HorizontalReadingOrder(t *testing.T) {
	s := &ProximityMergeStrategy{config: testConfig}
	// topLeft–topRight gap = dx 10 ≤ 10 → merge
	// topLeft–bottomLeft gap = dy 2  ≤ 10 → merge
	// All three end up in one cluster via union-find transitivity.
	topLeft    := makeBox(0, 0, 100, 10)   // cy=5, cx_sum=100
	topRight   := makeBox(110, 0, 100, 10) // cy=5, cx_sum=320
	bottomLeft := makeBox(0, 12, 100, 10)  // cy=17, cx_sum=100
	result := s.clusterBoxes([]utils.Box{topRight, topLeft, bottomLeft})
	if len(result) != 1 {
		t.Fatalf("expected 1 cluster, got %d", len(result))
	}
	ch := result[0].Children
	if len(ch) != 3 {
		t.Fatalf("expected 3 children, got %d", len(ch))
	}
	// Sorted: topLeft (cy=5, cx_sum=100), topRight (cy=5, cx_sum=320), bottomLeft (cy=17)
	if ch[0].Quad != topLeft.Quad {
		t.Errorf("children[0] should be topLeft")
	}
	if ch[1].Quad != topRight.Quad {
		t.Errorf("children[1] should be topRight")
	}
	if ch[2].Quad != bottomLeft.Quad {
		t.Errorf("children[2] should be bottomLeft")
	}
}

func TestProximity_VerticalReadingOrder(t *testing.T) {
	s := &ProximityMergeStrategy{config: testConfig}
	// leftColTop–rightColTop gap = dx 10 ≤ 10 → merge
	// leftColTop–leftColBot  gap = dy 5  ≤ 10 → merge
	// All three in one cluster; CJK order: right col first, then left top, left bottom.
	rightColTop := makeVertBox(15, 0, 5, 100)  // cx=17
	leftColTop  := makeVertBox(0, 0, 5, 100)   // cx=2, cy_sum=100
	leftColBot  := makeVertBox(0, 105, 5, 100) // cx=2, cy_sum=310
	result := s.clusterBoxes([]utils.Box{leftColBot, leftColTop, rightColTop})
	if len(result) != 1 {
		t.Fatalf("expected 1 cluster, got %d", len(result))
	}
	ch := result[0].Children
	if len(ch) != 3 {
		t.Fatalf("expected 3 children, got %d", len(ch))
	}
	if ch[0].Quad != rightColTop.Quad {
		t.Errorf("children[0] should be rightColTop (highest cx)")
	}
	if ch[1].Quad != leftColTop.Quad {
		t.Errorf("children[1] should be leftColTop (same cx, lower cy_sum)")
	}
	if ch[2].Quad != leftColBot.Quad {
		t.Errorf("children[2] should be leftColBot")
	}
}

func TestProximity_ParentAABB(t *testing.T) {
	s := &ProximityMergeStrategy{config: testConfig}
	// box1 AABB [0,0,10,10], box2 AABB [20,0,50,10]; gap dx=10 → merge
	// Merged parent AABB should be [0,0,50,10].
	boxes := []utils.Box{
		makeBox(0, 0, 10, 10),
		makeBox(20, 0, 30, 10),
	}
	result := s.clusterBoxes(boxes)
	if len(result) != 1 {
		t.Fatalf("expected 1 parent, got %d", len(result))
	}
	aabb := result[0].Quad.AABB()
	if aabb[0] != 0 || aabb[1] != 0 || aabb[2] != 50 || aabb[3] != 10 {
		t.Errorf("parent AABB = %v, want [0 0 50 10]", aabb)
	}
}

func TestProximity_ParentsSortedByReadingOrder(t *testing.T) {
	s := &ProximityMergeStrategy{config: testConfig}
	// Two groups far apart: gap = 50 - 10 = 40 > 10 → no merge.
	upper := makeBox(0, 0, 100, 10)
	lower := makeBox(0, 50, 100, 10)
	result := s.clusterBoxes([]utils.Box{lower, upper})
	if len(result) != 2 {
		t.Fatalf("expected 2 parents, got %d", len(result))
	}
	if result[0].Quad != upper.Quad {
		t.Errorf("result[0] should be upper group (lower centerY)")
	}
}

// ---------------------------------------------------------------------------
// ProximityMergeStrategy — Detect
// ---------------------------------------------------------------------------

func TestProximity_DetectError(t *testing.T) {
	sentinel := errors.New("boom")
	s := &ProximityMergeStrategy{
		config:   testConfig,
		detector: &mockDetector{err: sentinel},
	}
	_, err := s.Detect(image.NewGray(image.Rect(0, 0, 100, 100)))
	if err == nil || !errors.Is(err, sentinel) {
		t.Fatalf("expected wrapped sentinel error, got %v", err)
	}
}

func TestProximity_DetectEmpty(t *testing.T) {
	s := &ProximityMergeStrategy{
		config:   testConfig,
		detector: &mockDetector{},
	}
	result, err := s.Detect(image.NewGray(image.Rect(0, 0, 100, 100)))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != nil {
		t.Fatalf("expected nil result, got %v", result)
	}
}

func TestProximity_DetectHappyPath(t *testing.T) {
	boxes := []utils.Box{
		makeBox(0, 0, 100, 10),
		makeBox(0, 12, 100, 10),
	}
	s := &ProximityMergeStrategy{
		config:   testConfig,
		detector: &mockDetector{boxes: boxes},
	}
	result, err := s.Detect(image.NewGray(image.Rect(0, 0, 200, 200)))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result) != 1 {
		t.Fatalf("expected 1 parent, got %d", len(result))
	}
	if len(result[0].Children) != 2 {
		t.Fatalf("expected 2 children, got %d", len(result[0].Children))
	}
}
