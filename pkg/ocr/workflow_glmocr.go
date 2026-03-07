package ocr

import (
	classifystub "github.com/multippt/gopaddleocr/pkg/ocr/classify/stub"
	"github.com/multippt/gopaddleocr/pkg/ocr/common"
	"github.com/multippt/gopaddleocr/pkg/ocr/detect"
	"github.com/multippt/gopaddleocr/pkg/ocr/detect/boxmerge"
	detectpaddleocr "github.com/multippt/gopaddleocr/pkg/ocr/detect/paddleocr"
	ppdoclayoutv3 "github.com/multippt/gopaddleocr/pkg/ocr/detect/pp-doclayoutv3"
	openairec "github.com/multippt/gopaddleocr/pkg/ocr/recognize/openai"
)

func NewGLMOCRWorkflow(engineConf *Config, configSrc common.ConfigSource) *Workflow {
	var det detect.Detector
	if engineConf.EnableBoxMerge {
		det = boxmerge.NewModel(
			ppdoclayoutv3.NewModel(),
			detectpaddleocr.NewModel(),
		)
	}
	return NewWorkflow(
		det,
		classifystub.NewModel(),
		openairec.NewModel(),
		configSrc)
}
