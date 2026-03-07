package classify

import (
	"image"

	"github.com/multippt/gopaddleocr/pkg/ocr/common"
)

type Classifier interface {
	common.Model
	Classify(img image.Image, quad [4][2]int) (rotated bool, err error)
	Close() error
}
