package boxmerge

import (
	"image"

	"github.com/multippt/gopaddleocr/pkg/ocr/common"
	"github.com/multippt/gopaddleocr/pkg/ocr/utils"
)

// ---------------------------------------------------------------------------
// Mock infrastructure
// ---------------------------------------------------------------------------

type mockDetector struct {
	boxes []utils.Box
	err   error
}

func (m *mockDetector) GetName() string                           { return "mock" }
func (m *mockDetector) Init(_ common.ConfigSource) error          { return nil }
func (m *mockDetector) GetDefaultConfig() common.ModelConfig      { return &common.BaseModelConfig{} }
func (m *mockDetector) Close() error                              { return nil }
func (m *mockDetector) Detect(_ image.Image) ([]utils.Box, error) { return m.boxes, m.err }

type mockConfigSource struct{}

func (mockConfigSource) GetConfig(_ string) common.ModelConfig { return &common.BaseModelConfig{} }

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// makeBox creates a box with the given origin and dimensions.
// For horizontal boxes pass w > h; for vertical boxes pass h > w.
func makeBox(x, y, w, h int) utils.Box {
	return utils.Box{
		Quad: utils.Quad{
			{x, y},
			{x + w, y},
			{x + w, y + h},
			{x, y + h},
		},
		Score:   1.0,
		ClassID: -1,
		Order:   -1,
	}
}

// makeVertBox is an alias that documents the caller's intent to create a vertical box.
func makeVertBox(x, y, w, h int) utils.Box { return makeBox(x, y, w, h) }

// ---------------------------------------------------------------------------
// Shared test config
// ---------------------------------------------------------------------------

var testConfig = &ModelConfig{
	MinOverlapRatio:  0.8,
	MaxMergeDistance: 10,
	MaxSizeRatio:     1.5,
}
