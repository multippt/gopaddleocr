package stub

import (
	"image"

	"github.com/multippt/gopaddleocr/pkg/ocr/utils"
)

// Detector returns a single box covering the entire image.
type Detector struct{}

func (Detector) Detect(img image.Image) ([]utils.Box, error) {
	b := img.Bounds()
	quad := utils.Quad{
		{b.Min.X, b.Min.Y},
		{b.Max.X, b.Min.Y},
		{b.Max.X, b.Max.Y},
		{b.Min.X, b.Max.Y},
	}
	return []utils.Box{{Quad: quad, Score: 1, ClassID: -1, Order: -1}}, nil
}

func (Detector) Close() error { return nil }
