package stub

import (
	"image"

	"github.com/multippt/gopaddleocr/pkg/ocr/common"
)

const ModelName = "stub-cls"

// Model always returns (false, nil) — no rotation needed.
type Model struct{}

func NewModel() *Model { return &Model{} }

func (Model) GetName() string { return ModelName }

func (Model) Init(_ common.ConfigSource) error { return nil }

func (Model) GetDefaultConfig() common.ModelConfig {
	return &common.BaseModelConfig{}
}

func (Model) Classify(_ image.Image, _ [4][2]int) (bool, error) { return false, nil }
func (Model) Close() error                                      { return nil }
