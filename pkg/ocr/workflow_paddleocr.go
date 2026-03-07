package ocr

import (
	classifypaddleocr "github.com/multippt/gopaddleocr/pkg/ocr/classify/paddleocr"
	"github.com/multippt/gopaddleocr/pkg/ocr/common"
	"github.com/multippt/gopaddleocr/pkg/ocr/detect"
	"github.com/multippt/gopaddleocr/pkg/ocr/detect/boxmerge"
	detectpaddleocr "github.com/multippt/gopaddleocr/pkg/ocr/detect/paddleocr"
	ppdoclayoutv3 "github.com/multippt/gopaddleocr/pkg/ocr/detect/pp-doclayoutv3"
	recognizepaddleocr "github.com/multippt/gopaddleocr/pkg/ocr/recognize/paddleocr"
)

func NewPaddleOCRWorkflow(engineConf *Config, configSrc common.ConfigSource) *Workflow {
	var det detect.Detector
	if engineConf.EnableBoxMerge {
		det = boxmerge.NewModel(
			ppdoclayoutv3.NewModel(),
			detectpaddleocr.NewModel(),
		)
	}
	return NewWorkflow(
		det,
		classifypaddleocr.NewModel(),
		recognizepaddleocr.NewModel(),
		configSrc)
}
