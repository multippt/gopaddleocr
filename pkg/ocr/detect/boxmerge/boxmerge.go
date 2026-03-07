package boxmerge

import (
	"fmt"
	"image"
	"math"
	"sort"
	"sync"

	"github.com/multippt/gopaddleocr/pkg/ocr/detect"
	"github.com/multippt/gopaddleocr/pkg/ocr/onnx"
)

const ModelName = "box-merge"

// Strategy controls how child boxes are grouped into parents.
type Strategy int

const (
	// DocLayout uses PP-DocLayout for parent boxes, then assigns child boxes by overlap.
	DocLayout Strategy = iota
	// Statistical clusters child boxes by proximity, text size, and orientation.
	Statistical
)

// Config configures the BoxMerge detector (strategy parameters only).
type Config struct {
	onnx.BaseModelConfig

	Strategy Strategy

	// DocLayout settings
	MinOverlapRatio float64 // Minimum fraction of child area covered by parent (default 0.8).

	// Statistical settings
	MaxMergeDistance float64 // Max gap (pixels) between boxes to consider merging (default 20).
	MaxSizeRatio     float64 // Max ratio of text sizes to merge (default 1.5).
}

// Model implements detect.Detector by merging child boxes into parent groups.
type Model struct {
	config         *Config
	childDetector  detect.Detector
	parentDetector detect.Detector
}

func NewModel(childDetector, parentDetector detect.Detector) (*Model, error) {
	if childDetector == nil {
		return nil, fmt.Errorf("boxmerge: ChildDetector is required")
	}
	return &Model{
		config: &Config{
			MinOverlapRatio:  0.8,
			MaxMergeDistance: 20,
			MaxSizeRatio:     1.5,
		},
		childDetector:  childDetector,
		parentDetector: parentDetector,
	}, nil
}

func (m *Model) Init(_ onnx.ModelConfig) error { return nil }

func (m *Model) Close() error {
	var first error
	if m.childDetector != nil {
		if err := m.childDetector.Close(); err != nil && first == nil {
			first = err
		}
	}
	if m.parentDetector != nil {
		if err := m.parentDetector.Close(); err != nil && first == nil {
			first = err
		}
	}
	return first
}

// Detect runs the configured merge strategy and returns parent boxes with children.
func (m *Model) Detect(img image.Image) ([]detect.Box, error) {
	switch m.config.Strategy {
	case DocLayout:
		return m.detectDocLayout(img)
	case Statistical:
		return m.detectStatistical(img)
	default:
		return nil, fmt.Errorf("boxmerge: unknown strategy %d", m.config.Strategy)
	}
}

// ---------------------------------------------------------------------------
// DocLayout strategy
// ---------------------------------------------------------------------------

func (m *Model) detectDocLayout(img image.Image) ([]detect.Box, error) {
	// Run child and parent detectors in parallel.
	var (
		childBoxes, parentBoxes []detect.Box
		childErr, parentErr     error
		wg                      sync.WaitGroup
	)
	wg.Add(2)
	go func() {
		defer wg.Done()
		childBoxes, childErr = m.childDetector.Detect(img)
	}()
	go func() {
		defer wg.Done()
		parentBoxes, parentErr = m.parentDetector.Detect(img)
	}()
	wg.Wait()

	if childErr != nil {
		return nil, fmt.Errorf("boxmerge child detect: %w", childErr)
	}
	if parentErr != nil {
		return nil, fmt.Errorf("boxmerge parent detect: %w", parentErr)
	}

	return m.assignChildrenToParents(childBoxes, parentBoxes), nil
}

// assignChildrenToParents places each child into the parent with largest overlap
// (if >= MinOverlapRatio of child area). Unmatched children become their own parents.
func (m *Model) assignChildrenToParents(children, parents []detect.Box) []detect.Box {
	// Track which children are assigned.
	assigned := make([]bool, len(children))
	// For each parent, collect assigned children.
	parentChildren := make([][]detect.Box, len(parents))

	for ci, child := range children {
		childAABB := detect.BoxAABB(child.Quad)
		childArea := aabbArea(childAABB)
		if childArea <= 0 {
			assigned[ci] = true // degenerate box, skip
			continue
		}
		childOrient := boxOrientation(child.Quad)

		bestParent := -1
		bestOverlap := 0.0
		for pi, parent := range parents {
			parentOrient := boxOrientation(parent.Quad)
			if childOrient != parentOrient {
				continue
			}
			parentAABB := detect.BoxAABB(parent.Quad)
			overlap := aabbIntersectionArea(childAABB, parentAABB)
			ratio := overlap / childArea
			if ratio >= m.config.MinOverlapRatio && overlap > bestOverlap {
				bestOverlap = overlap
				bestParent = pi
			}
		}
		if bestParent >= 0 {
			assigned[ci] = true
			parentChildren[bestParent] = append(parentChildren[bestParent], child)
		}
	}

	// Build result: parents with their children.
	var result []detect.Box
	for pi, parent := range parents {
		if len(parentChildren[pi]) == 0 {
			continue // skip parents with no children
		}
		parent.Children = parentChildren[pi]
		result = append(result, parent)
	}

	// Orphan children become self-parents.
	for ci, child := range children {
		if !assigned[ci] {
			orphanParent := detect.Box{
				Quad:     child.Quad,
				Score:    child.Score,
				ClassID:  -1,
				Order:    -1,
				Children: []detect.Box{child},
			}
			result = append(result, orphanParent)
		}
	}

	return result
}

// ---------------------------------------------------------------------------
// Statistical strategy
// ---------------------------------------------------------------------------

func (m *Model) detectStatistical(img image.Image) ([]detect.Box, error) {
	children, err := m.childDetector.Detect(img)
	if err != nil {
		return nil, fmt.Errorf("boxmerge child detect: %w", err)
	}
	if len(children) == 0 {
		return nil, nil
	}

	return m.clusterBoxes(children), nil
}

// clusterBoxes groups child boxes by proximity, text size, and orientation.
func (m *Model) clusterBoxes(boxes []detect.Box) []detect.Box {
	n := len(boxes)
	if n == 0 {
		return nil
	}

	// Union-Find for clustering.
	parent := make([]int, n)
	for i := range parent {
		parent[i] = i
	}
	var find func(int) int
	find = func(x int) int {
		if parent[x] != x {
			parent[x] = find(parent[x])
		}
		return parent[x]
	}
	union := func(a, b int) {
		ra, rb := find(a), find(b)
		if ra != rb {
			parent[ra] = rb
		}
	}

	// Precompute AABBs and text sizes.
	type boxInfo struct {
		aabb     [4]int // minX, minY, maxX, maxY
		textSize float64
		orient   int // 0=horizontal, 1=vertical
	}
	infos := make([]boxInfo, n)
	for i, b := range boxes {
		aabb := detect.BoxAABB(b.Quad)
		w := float64(aabb[2] - aabb[0])
		h := float64(aabb[3] - aabb[1])
		// Text size is the smaller dimension (line height for horizontal, line width for vertical).
		orient := 0
		textSize := h
		if h > w {
			orient = 1
			textSize = w
		}
		infos[i] = boxInfo{aabb: aabb, textSize: textSize, orient: orient}
	}

	// Pairwise check for mergeable boxes.
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if infos[i].orient != infos[j].orient {
				continue
			}
			// Check text size ratio.
			sizeRatio := infos[i].textSize / infos[j].textSize
			if sizeRatio < 1 {
				sizeRatio = 1 / sizeRatio
			}
			if sizeRatio > m.config.MaxSizeRatio {
				continue
			}
			// Check distance (gap between AABBs).
			dist := aabbGap(infos[i].aabb, infos[j].aabb)
			if dist <= m.config.MaxMergeDistance {
				union(i, j)
			}
		}
	}

	// Group by cluster root.
	groups := make(map[int][]int)
	for i := 0; i < n; i++ {
		root := find(i)
		groups[root] = append(groups[root], i)
	}

	// Build parent boxes from clusters.
	var result []detect.Box
	for _, indices := range groups {
		children := make([]detect.Box, len(indices))
		for i, idx := range indices {
			children[i] = boxes[idx]
		}

		// Parent quad is the AABB of all children.
		parentAABB := infos[indices[0]].aabb
		for _, idx := range indices[1:] {
			a := infos[idx].aabb
			if a[0] < parentAABB[0] {
				parentAABB[0] = a[0]
			}
			if a[1] < parentAABB[1] {
				parentAABB[1] = a[1]
			}
			if a[2] > parentAABB[2] {
				parentAABB[2] = a[2]
			}
			if a[3] > parentAABB[3] {
				parentAABB[3] = a[3]
			}
		}

		// Sort children by reading order (top-to-bottom for horizontal, left-to-right for vertical).
		sort.Slice(children, func(a, b int) bool {
			aabb1 := detect.BoxAABB(children[a].Quad)
			aabb2 := detect.BoxAABB(children[b].Quad)
			if infos[indices[0]].orient == 0 {
				// Horizontal: sort by Y center, then X.
				cy1 := (aabb1[1] + aabb1[3]) / 2
				cy2 := (aabb2[1] + aabb2[3]) / 2
				if cy1 != cy2 {
					return cy1 < cy2
				}
				return (aabb1[0] + aabb1[2]) < (aabb2[0] + aabb2[2])
			}
			// Vertical: sort by X center, then Y.
			cx1 := (aabb1[0] + aabb1[2]) / 2
			cx2 := (aabb2[0] + aabb2[2]) / 2
			if cx1 != cx2 {
				return cx1 < cx2
			}
			return (aabb1[1] + aabb1[3]) < (aabb2[1] + aabb2[3])
		})

		parentQuad := [4][2]int{
			{parentAABB[0], parentAABB[1]}, // TL
			{parentAABB[2], parentAABB[1]}, // TR
			{parentAABB[2], parentAABB[3]}, // BR
			{parentAABB[0], parentAABB[3]}, // BL
		}
		result = append(result, detect.Box{
			Quad:     parentQuad,
			Score:    children[0].Score,
			ClassID:  -1,
			Order:    -1,
			Children: children,
		})
	}

	// Sort parents by reading order (top-left first).
	sort.Slice(result, func(i, j int) bool {
		ai := detect.BoxAABB(result[i].Quad)
		aj := detect.BoxAABB(result[j].Quad)
		cy1 := (ai[1] + ai[3]) / 2
		cy2 := (aj[1] + aj[3]) / 2
		if cy1 != cy2 {
			return cy1 < cy2
		}
		return (ai[0] + ai[2]) < (aj[0] + aj[2])
	})

	return result
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

func aabbArea(a [4]int) float64 {
	w := a[2] - a[0]
	h := a[3] - a[1]
	if w <= 0 || h <= 0 {
		return 0
	}
	return float64(w) * float64(h)
}

func aabbIntersectionArea(a, b [4]int) float64 {
	x1 := a[0]
	if b[0] > x1 {
		x1 = b[0]
	}
	y1 := a[1]
	if b[1] > y1 {
		y1 = b[1]
	}
	x2 := a[2]
	if b[2] < x2 {
		x2 = b[2]
	}
	y2 := a[3]
	if b[3] < y2 {
		y2 = b[3]
	}
	w := x2 - x1
	h := y2 - y1
	if w <= 0 || h <= 0 {
		return 0
	}
	return float64(w) * float64(h)
}

// aabbGap returns the minimum gap between two AABBs (0 if overlapping).
func aabbGap(a, b [4]int) float64 {
	dx := 0.0
	if a[2] < b[0] {
		dx = float64(b[0] - a[2])
	} else if b[2] < a[0] {
		dx = float64(a[0] - b[2])
	}
	dy := 0.0
	if a[3] < b[1] {
		dy = float64(b[1] - a[3])
	} else if b[3] < a[1] {
		dy = float64(a[1] - b[3])
	}
	return math.Sqrt(dx*dx + dy*dy)
}

// boxOrientation returns 0 for horizontal (width >= height) and 1 for vertical.
func boxOrientation(q [4][2]int) int {
	aabb := detect.BoxAABB(q)
	w := aabb[2] - aabb[0]
	h := aabb[3] - aabb[1]
	if h > w {
		return 1
	}
	return 0
}
