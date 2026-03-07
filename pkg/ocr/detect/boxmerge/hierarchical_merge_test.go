package boxmerge

import (
	"errors"
	"image"
	"strings"
	"testing"

	"github.com/multippt/gopaddleocr/pkg/ocr/utils"
)

// ---------------------------------------------------------------------------
// HierarchicalMergeStrategy — assignChildrenToParents
// ---------------------------------------------------------------------------

func newHierarchical() *HierarchicalMergeStrategy {
	return &HierarchicalMergeStrategy{
		config:         testConfig,
		parentDetector: &mockDetector{},
		childDetector:  &mockDetector{},
	}
}

func TestHierarchical_ChildFullyInsideParent(t *testing.T) {
	s := newHierarchical()
	parent := makeBox(0, 0, 100, 100)
	child := makeBox(10, 10, 80, 80) // AABB [10,10,90,90] fully inside [0,0,100,100]
	result := s.assignChildrenToParents([]utils.Box{child}, []utils.Box{parent})
	if len(result) != 1 {
		t.Fatalf("expected 1 parent, got %d", len(result))
	}
	if len(result[0].Children) != 1 {
		t.Fatalf("expected 1 child, got %d", len(result[0].Children))
	}
}

func TestHierarchical_ChildAboveThreshold(t *testing.T) {
	s := newHierarchical()
	// child AABB [15,0,115,100], area=10000
	// intersection with parent [0,0,100,100] = [15,0,100,100] = 85*100=8500
	// ratio = 0.85 >= MinOverlapRatio(0.8) → assigned
	parent := makeBox(0, 0, 100, 100)
	child := makeBox(15, 0, 100, 100)
	result := s.assignChildrenToParents([]utils.Box{child}, []utils.Box{parent})
	if len(result) != 1 || len(result[0].Children) != 1 {
		t.Fatalf("expected child assigned to parent")
	}
}

func TestHierarchical_ChildBelowThreshold(t *testing.T) {
	s := newHierarchical()
	// child AABB [50,0,150,100], area=10000
	// intersection with parent [0,0,100,100] = [50,0,100,100] = 50*100=5000
	// ratio = 0.5 < MinOverlapRatio(0.8) → not assigned; child becomes orphan
	parent := makeBox(0, 0, 100, 100)
	child := makeBox(50, 0, 100, 100)
	result := s.assignChildrenToParents([]utils.Box{child}, []utils.Box{parent})
	if len(result) != 1 {
		t.Fatalf("expected 1 orphan, got %d", len(result))
	}
	if result[0].ClassID != -1 || result[0].Order != -1 {
		t.Errorf("orphan should have ClassID=-1 and Order=-1, got ClassID=%d Order=%d",
			result[0].ClassID, result[0].Order)
	}
}

func TestHierarchical_NoParents(t *testing.T) {
	s := newHierarchical()
	children := []utils.Box{
		makeBox(0, 0, 100, 10),
		makeBox(0, 20, 100, 10),
	}
	result := s.assignChildrenToParents(children, nil)
	if len(result) != 2 {
		t.Fatalf("expected 2 orphans, got %d", len(result))
	}
}

func TestHierarchical_DegenerateChild(t *testing.T) {
	s := newHierarchical()
	degenerate := utils.Box{Quad: utils.Quad{{0, 0}, {0, 0}, {0, 0}, {0, 0}}}
	result := s.assignChildrenToParents([]utils.Box{degenerate}, nil)
	if len(result) != 0 {
		t.Fatalf("degenerate child should be skipped; expected 0 results, got %d", len(result))
	}
}

func TestHierarchical_OrientationMismatch(t *testing.T) {
	s := newHierarchical()
	parent := makeBox(0, 0, 100, 50)     // horizontal
	child := makeVertBox(10, 10, 30, 80) // vertical (h=80 > w=30)
	result := s.assignChildrenToParents([]utils.Box{child}, []utils.Box{parent})
	// parent excluded (no children); child becomes orphan
	if len(result) != 1 {
		t.Fatalf("expected 1 orphan, got %d", len(result))
	}
	if result[0].ClassID != -1 {
		t.Errorf("orphan ClassID = %d, want -1", result[0].ClassID)
	}
}

func TestHierarchical_MultipleParents_BestWins(t *testing.T) {
	s := newHierarchical()
	// child AABB [10,0,110,100], area=10000
	// parent1 [0,0,100,100]: intersection=[10,0,100,100]=90*100=9000, ratio=0.9
	// parent2 [0,0,200,100]: intersection=[10,0,110,100]=100*100=10000, ratio=1.0
	// parent2 has larger overlap → wins
	parent1 := makeBox(0, 0, 100, 100)
	parent2 := makeBox(0, 0, 200, 100)
	child := makeBox(10, 0, 100, 100)
	result := s.assignChildrenToParents([]utils.Box{child}, []utils.Box{parent1, parent2})
	if len(result) != 1 {
		t.Fatalf("expected 1 parent in result, got %d", len(result))
	}
	if result[0].Quad != parent2.Quad {
		t.Errorf("expected parent2 (best overlap) to win")
	}
}

func TestHierarchical_ParentWithNoChildrenExcluded(t *testing.T) {
	s := newHierarchical()
	parent := makeBox(0, 0, 100, 100)
	child := makeBox(200, 200, 100, 100) // far outside parent → no overlap
	result := s.assignChildrenToParents([]utils.Box{child}, []utils.Box{parent})
	// parent excluded; child becomes orphan
	if len(result) != 1 {
		t.Fatalf("expected 1 result (orphan), got %d", len(result))
	}
	if result[0].ClassID != -1 {
		t.Errorf("orphan ClassID = %d, want -1", result[0].ClassID)
	}
}

func TestHierarchical_MultipleChildrenOneParent(t *testing.T) {
	s := newHierarchical()
	parent := makeBox(0, 0, 200, 100)
	child1 := makeBox(10, 10, 50, 50)
	child2 := makeBox(70, 10, 50, 50)
	child3 := makeBox(130, 10, 50, 50)
	result := s.assignChildrenToParents(
		[]utils.Box{child1, child2, child3},
		[]utils.Box{parent},
	)
	if len(result) != 1 {
		t.Fatalf("expected 1 parent, got %d", len(result))
	}
	if len(result[0].Children) != 3 {
		t.Fatalf("expected 3 children, got %d", len(result[0].Children))
	}
}

func TestHierarchical_OrphanChildBecomesParent(t *testing.T) {
	s := newHierarchical()
	child := makeBox(0, 0, 100, 10)
	result := s.assignChildrenToParents([]utils.Box{child}, nil)
	if len(result) != 1 {
		t.Fatalf("expected 1 orphan parent, got %d", len(result))
	}
	orphan := result[0]
	if orphan.ClassID != -1 || orphan.Order != -1 {
		t.Errorf("orphan ClassID=%d Order=%d, want -1/-1", orphan.ClassID, orphan.Order)
	}
	if len(orphan.Children) != 1 || orphan.Children[0].Quad != child.Quad {
		t.Errorf("orphan should contain the child itself")
	}
}

// ---------------------------------------------------------------------------
// HierarchicalMergeStrategy — Detect
// ---------------------------------------------------------------------------

func TestHierarchical_ChildDetectError(t *testing.T) {
	sentinel := errors.New("child boom")
	s := &HierarchicalMergeStrategy{
		config:         testConfig,
		parentDetector: &mockDetector{},
		childDetector:  &mockDetector{err: sentinel},
	}
	_, err := s.Detect(image.NewGray(image.Rect(0, 0, 100, 100)))
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !errors.Is(err, sentinel) {
		t.Errorf("expected sentinel in error chain, got %v", err)
	}
	if !strings.Contains(err.Error(), "child detect") {
		t.Errorf("error %q should contain \"child detect\"", err.Error())
	}
}

func TestHierarchical_ParentDetectError(t *testing.T) {
	sentinel := errors.New("parent boom")
	s := &HierarchicalMergeStrategy{
		config:         testConfig,
		parentDetector: &mockDetector{err: sentinel},
		childDetector:  &mockDetector{},
	}
	_, err := s.Detect(image.NewGray(image.Rect(0, 0, 100, 100)))
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !errors.Is(err, sentinel) {
		t.Errorf("expected sentinel in error chain, got %v", err)
	}
	if !strings.Contains(err.Error(), "parent detect") {
		t.Errorf("error %q should contain \"parent detect\"", err.Error())
	}
}

func TestHierarchical_NilParentFallsBackToProximity(t *testing.T) {
	result := NewHierarchicalMergeStrategy(testConfig, nil, &mockDetector{})
	if _, ok := result.(*ProximityMergeStrategy); !ok {
		t.Errorf("expected *ProximityMergeStrategy, got %T", result)
	}
}

func TestHierarchical_NilChildFallsBackToProximity(t *testing.T) {
	result := NewHierarchicalMergeStrategy(testConfig, &mockDetector{}, nil)
	if _, ok := result.(*ProximityMergeStrategy); !ok {
		t.Errorf("expected *ProximityMergeStrategy, got %T", result)
	}
}

func TestHierarchical_DetectHappyPath(t *testing.T) {
	parentBox := makeBox(0, 0, 200, 100)
	childBox := makeBox(10, 10, 100, 50) // inside parent; ratio=1.0
	s := &HierarchicalMergeStrategy{
		config:         testConfig,
		parentDetector: &mockDetector{boxes: []utils.Box{parentBox}},
		childDetector:  &mockDetector{boxes: []utils.Box{childBox}},
	}
	result, err := s.Detect(image.NewGray(image.Rect(0, 0, 300, 200)))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result) != 1 {
		t.Fatalf("expected 1 parent, got %d", len(result))
	}
	if len(result[0].Children) != 1 {
		t.Fatalf("expected 1 child, got %d", len(result[0].Children))
	}
}
