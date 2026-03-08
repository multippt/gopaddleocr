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
