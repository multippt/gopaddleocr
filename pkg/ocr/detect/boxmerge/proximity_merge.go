package boxmerge

import (
	"fmt"
	"image"
	"sort"

	"github.com/multippt/gopaddleocr/pkg/ocr/common"
	"github.com/multippt/gopaddleocr/pkg/ocr/detect"
	"github.com/multippt/gopaddleocr/pkg/ocr/utils"
)

// ---------------------------------------------------------------------------
// Proximity merge strategy
// ---------------------------------------------------------------------------

// ProximityMergeStrategy merge boxes based on how close they are to each other.
// We use a clustering algorithm to group boxes based on proximity, text size, and orientation.
// This strategy uses a single detection model that emits line-level boxes.
type ProximityMergeStrategy struct {
	config   *ModelConfig
	detector detect.Detector
}

func NewProximityMergeStrategy(
	config *ModelConfig,
	detector detect.Detector) MergeStrategy {
	return &ProximityMergeStrategy{config: config, detector: detector}
}

func (s *ProximityMergeStrategy) Init(configSrc common.ConfigSource) error {
	return s.detector.Init(configSrc)
}

func (s *ProximityMergeStrategy) Detect(img image.Image) ([]utils.Box, error) {
	children, err := s.detector.Detect(img)
	if err != nil {
		return nil, fmt.Errorf("boxmerge child detect: %w", err)
	}
	if len(children) == 0 {
		return nil, nil
	}

	return s.clusterBoxes(children), nil
}

// clusterBoxes groups child boxes by proximity, text size, and orientation.
func (s *ProximityMergeStrategy) clusterBoxes(boxes []utils.Box) []utils.Box {
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
		aabb     utils.AABB // minX, minY, maxX, maxY
		textSize float64
		orient   int // 0=horizontal, 1=vertical
	}
	infos := make([]boxInfo, n)
	for i, b := range boxes {
		aabb := b.Quad.AABB()
		w := float64(aabb.Width())
		h := float64(aabb.Height())
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
			if sizeRatio > s.config.MaxSizeRatio {
				continue
			}
			// Check distance (gap between AABBs).
			dist := infos[i].aabb.Gap(infos[j].aabb)
			if dist <= s.config.MaxMergeDistance {
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
	var result []utils.Box
	for _, indices := range groups {
		children := make([]utils.Box, len(indices))
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

		// Sort children by reading order (top-to-bottom for horizontal, right-to-left for vertical).
		sort.Slice(children, func(a, b int) bool {
			aabb1 := children[a].Quad.AABB()
			aabb2 := children[b].Quad.AABB()
			if infos[indices[0]].orient == 0 {
				return aabb1.Compare(aabb2)
			}
			return aabb1.CompareCJK(aabb2)
		})

		parentQuad := [4][2]int{
			{parentAABB[0], parentAABB[1]}, // TL
			{parentAABB[2], parentAABB[1]}, // TR
			{parentAABB[2], parentAABB[3]}, // BR
			{parentAABB[0], parentAABB[3]}, // BL
		}
		result = append(result, utils.Box{
			Quad:     parentQuad,
			Score:    children[0].Score,
			ClassID:  -1,
			Order:    -1,
			Children: children,
		})
	}

	// Sort parents by reading order (top-left first).
	sort.Slice(result, func(i, j int) bool {
		return result[i].Quad.AABB().Compare(result[j].Quad.AABB())
	})

	return result
}
