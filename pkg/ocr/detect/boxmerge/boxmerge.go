package boxmerge

import (
	"fmt"
	"image"

	"github.com/multippt/gopaddleocr/pkg/ocr/common"
	"github.com/multippt/gopaddleocr/pkg/ocr/detect"
	"github.com/multippt/gopaddleocr/pkg/ocr/utils"
	"golang.org/x/sync/errgroup"
)

const ModelName = "box-merge"

// Strategy controls how child boxes are grouped into parents.
type Strategy int

const (
	// Hierarchical uses PP-Hierarchical for parent boxes, then assigns child boxes by overlap.
	Hierarchical Strategy = iota
	// Proximity clusters child boxes by proximity, text size, and orientation.
	Proximity
)

// ModelConfig configures the BoxMerge detector (strategy parameters only).
type ModelConfig struct {
	common.BaseModelConfig

	Strategy Strategy

	// Hierarchical settings
	MinOverlapRatio float64 // Minimum fraction of child area covered by parent (default 0.8).

	// Proximity settings
	MaxMergeDistance float64 // Max gap (pixels) between boxes to consider merging (default 20).
	MaxSizeRatio     float64 // Max ratio of text sizes to merge (default 1.5).
}

// Model implements detect.Detector by merging child boxes into parent groups.
type Model struct {
	config *ModelConfig

	strategy       MergeStrategy
	parentDetector detect.Detector
	childDetector  detect.Detector
}

func NewModel(parentDetector, childDetector detect.Detector) *Model {
	return &Model{
		parentDetector: parentDetector,
		childDetector:  childDetector,
	}
}

func (m *Model) GetName() string { return ModelName }

func (m *Model) GetDefaultConfig() common.ModelConfig {
	return &ModelConfig{
		Strategy:         Proximity,
		MinOverlapRatio:  0.8,
		MaxMergeDistance: 10,
		MaxSizeRatio:     1.5,
		BaseModelConfig:  common.BaseModelConfig{},
	}
}

func (m *Model) Init(configSrc common.ConfigSource) error {
	cfg, ok := configSrc.GetConfig(m.GetName()).(*ModelConfig)
	if !ok {
		cfg = m.GetDefaultConfig().(*ModelConfig)
	}
	m.config = cfg

	switch m.config.Strategy {
	case Hierarchical:
		m.strategy = NewHierarchicalMergeStrategy(m.config, m.parentDetector, m.childDetector)
	case Proximity:
		m.strategy = NewProximityMergeStrategy(m.config, m.childDetector)
	default:
		return fmt.Errorf("boxmerge: unknown strategy %d", m.config.Strategy)
	}

	return m.strategy.Init(configSrc)
}

func (m *Model) Close() error {
	var eg errgroup.Group
	if m.childDetector != nil {
		eg.Go(m.childDetector.Close)
	}
	if m.parentDetector != nil {
		eg.Go(m.parentDetector.Close)
	}
	return eg.Wait()
}

// Detect runs the configured merge strategy and returns parent boxes with children.
func (m *Model) Detect(img image.Image) ([]utils.Box, error) {
	return m.strategy.Detect(img)
}

type MergeStrategy interface {
	Init(common.ConfigSource) error
	Detect(image.Image) ([]utils.Box, error)
}
