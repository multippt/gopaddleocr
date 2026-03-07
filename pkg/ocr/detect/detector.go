package detect

import (
	"image"

	"github.com/multippt/gopaddleocr/pkg/ocr/common"
	"github.com/multippt/gopaddleocr/pkg/ocr/utils"
)

type Detector interface {
	common.Model
	Detect(img image.Image) ([]utils.Box, error)
	Close() error
}

// StubDetector returns a single box covering the entire image.
type StubDetector struct{}

func (StubDetector) Detect(img image.Image) ([]utils.Box, error) {
	b := img.Bounds()
	quad := utils.Quad{
		{b.Min.X, b.Min.Y},
		{b.Max.X, b.Min.Y},
		{b.Max.X, b.Max.Y},
		{b.Min.X, b.Max.Y},
	}
	return []utils.Box{{Quad: quad, Score: 1, ClassID: -1, Order: -1}}, nil
}

func (StubDetector) Close() error { return nil }
