package boxmerge

import (
	"fmt"
	"image"
	"sync"

	"github.com/multippt/gopaddleocr/pkg/ocr/common"
	"github.com/multippt/gopaddleocr/pkg/ocr/detect"
	"github.com/multippt/gopaddleocr/pkg/ocr/utils"
	"golang.org/x/sync/errgroup"
)

// ---------------------------------------------------------------------------
// Hierarchical strategy
// ---------------------------------------------------------------------------

// HierarchicalMergeStrategy merges boxes based on their parent-child relationship.
// This strategy uses two detection models: one for parent boxes (a layout model), and one for child boxes.
// This is to handle cases such as PaddleOCR where the recognition model only works on line-level boxes.
type HierarchicalMergeStrategy struct {
	config         *ModelConfig
	parentDetector detect.Detector
	childDetector  detect.Detector
}

func NewHierarchicalMergeStrategy(
	config *ModelConfig,
	parentDetector, childDetector detect.Detector) MergeStrategy {

	// Fallback if only have one detector.
	if parentDetector == nil || childDetector == nil {
		return NewProximityMergeStrategy(config, parentDetector)
	}

	return &HierarchicalMergeStrategy{
		config:         config,
		parentDetector: parentDetector,
		childDetector:  childDetector,
	}
}

func (s *HierarchicalMergeStrategy) initModelWrapper(model common.Model, configSrc common.ConfigSource) func() error {
	return func() error {
		return model.Init(configSrc)
	}
}

func (s *HierarchicalMergeStrategy) Init(configSrc common.ConfigSource) error {
	var eg errgroup.Group
	eg.Go(s.initModelWrapper(s.parentDetector, configSrc))
	eg.Go(s.initModelWrapper(s.childDetector, configSrc))
	return eg.Wait()
}

func (s *HierarchicalMergeStrategy) Close() error {
	var eg errgroup.Group
	eg.Go(s.parentDetector.Close)
	eg.Go(s.childDetector.Close)
	return eg.Wait()
}

func (s *HierarchicalMergeStrategy) Detect(img image.Image) ([]utils.Box, error) {
	// Run child and parent detectors in parallel.
	var (
		childBoxes, parentBoxes []utils.Box
		childErr, parentErr     error
		wg                      sync.WaitGroup
	)
	wg.Add(2)
	go func() {
		defer wg.Done()
		childBoxes, childErr = s.childDetector.Detect(img)
	}()
	go func() {
		defer wg.Done()
		parentBoxes, parentErr = s.parentDetector.Detect(img)
	}()
	wg.Wait()

	if childErr != nil {
		return nil, fmt.Errorf("boxmerge child detect: %w", childErr)
	}
	if parentErr != nil {
		return nil, fmt.Errorf("boxmerge parent detect: %w", parentErr)
	}

	return s.assignChildrenToParents(childBoxes, parentBoxes), nil
}

// assignChildrenToParents places each child into the parent with largest overlap
// (if >= MinOverlapRatio of child area). Unmatched children become their own parents.
func (s *HierarchicalMergeStrategy) assignChildrenToParents(children, parents []utils.Box) []utils.Box {
	// Track which children are assigned.
	assigned := make([]bool, len(children))
	// For each parent, collect assigned children.
	parentChildren := make([][]utils.Box, len(parents))

	for ci, child := range children {
		childAABB := child.Quad.AABB()
		childArea := childAABB.BoundedArea()
		if childArea <= 0 {
			assigned[ci] = true // degenerate box, skip
			continue
		}
		childIsVertical := child.Quad.IsVertical()

		bestParent := -1
		bestOverlap := 0.0
		for pi, parent := range parents {
			parentIsVertical := parent.Quad.IsVertical()
			if childIsVertical != parentIsVertical {
				continue
			}
			parentAABB := parent.Quad.AABB()
			overlap := childAABB.IntersectionArea(parentAABB)
			ratio := overlap / childArea
			if ratio >= s.config.MinOverlapRatio && overlap > bestOverlap {
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
	var result []utils.Box
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
			orphanParent := utils.Box{
				Quad:     child.Quad,
				Score:    child.Score,
				ClassID:  -1,
				Order:    -1,
				Children: []utils.Box{child},
			}
			result = append(result, orphanParent)
		}
	}

	return result
}
