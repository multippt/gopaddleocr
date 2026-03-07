package recognize

import (
	"image"

	"github.com/multippt/gopaddleocr/pkg/ocr/common"
)

type Result struct {
	Text  string
	Score float64
}

type Recognizer interface {
	common.Model
	RecognizeLineOnly() bool
	Recognize(img image.Image, quad [4][2]int) (Result, error)
	Close() error
}
