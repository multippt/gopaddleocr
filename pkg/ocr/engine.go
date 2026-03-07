package ocr

import (
	"fmt"
	"image"
	"path/filepath"
	"strings"
	"sync"

	"github.com/multippt/gopaddleocr/pkg/ocr/classify"
	classifypaddleocr "github.com/multippt/gopaddleocr/pkg/ocr/classify/paddleocr"
	"github.com/multippt/gopaddleocr/pkg/ocr/detect"
	"github.com/multippt/gopaddleocr/pkg/ocr/detect/boxmerge"
	detectpaddleocr "github.com/multippt/gopaddleocr/pkg/ocr/detect/paddleocr"
	ppdoclayoutv3 "github.com/multippt/gopaddleocr/pkg/ocr/detect/pp-doclayoutv3"
	"github.com/multippt/gopaddleocr/pkg/ocr/onnx"
	"github.com/multippt/gopaddleocr/pkg/ocr/recognize"
	openairec "github.com/multippt/gopaddleocr/pkg/ocr/recognize/openai"
	recognizepaddleocr "github.com/multippt/gopaddleocr/pkg/ocr/recognize/paddleocr"
)

// Result is a single detected text region.
// When Children is non-empty, this is a merged parent region containing child text lines.
type Result struct {
	Box      [][2]int `json:"box"` // [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
	Text     string   `json:"text"`
	Score    float64  `json:"score"`
	Children []Result `json:"children,omitempty"`
}

// Config holds all configuration for the Engine.
type Config struct {
	Models           map[string]onnx.ModelConfig
	EnableBoxMerge   bool
	RecognitionModel string // "PaddleOCR" (default) or "GLM-OCR"
}

// Workflow holds the three-stage pipeline components.
type Workflow struct {
	detector   detect.Detector
	classifier classify.Classifier
	recognizer recognize.Recognizer
}

// Engine coordinates the detect → classify → recognize pipeline.
type Engine struct {
	once     sync.Once
	loadErr  error
	workflow *Workflow
	config   *Config
}

// NewEngine creates an Engine with the given config.
func NewEngine(cfg *Config) *Engine {
	return &Engine{config: cfg}
}

// NewDefaultConfig returns a Config populated with PaddleOCR defaults.
func NewDefaultConfig() *Config {
	return &Config{
		RecognitionModel: "PaddleOCR",
		Models: map[string]onnx.ModelConfig{
			detectpaddleocr.ModelName: &detectpaddleocr.ModelConfig{
				LimitSideLength: 1280,
				Mean:            [3]float32{0.485, 0.456, 0.406},
				Std:             [3]float32{0.229, 0.224, 0.225},
				Thresh:          0.3,
				BoxThresh:       0.6,
				UnclipRatio:     2.0,
				MinArea:         16,
				BaseModelConfig: onnx.BaseModelConfig{
					OnnxConfig: onnx.Config{
						ModelPath:  filepath.Join("./models", "ch_PP-OCRv5_server_det.onnx"),
						InputName:  "x",
						OutputName: "fetch_name_0",
					},
				},
			},
			ppdoclayoutv3.ModelName: &ppdoclayoutv3.ModelConfig{
				BaseModelConfig: onnx.BaseModelConfig{
					OnnxConfig: onnx.Config{
						ModelPath: filepath.Join("./models", "PP-DocLayout-L.onnx"),
					},
				},
			},
			classifypaddleocr.ModelName: &classifypaddleocr.ModelConfig{
				Height:    48,
				Width:     192,
				Threshold: 0.9,
				Mean:      [3]float64{0.5, 0.5, 0.5},
				Std:       [3]float64{0.5, 0.5, 0.5},
				BaseModelConfig: onnx.BaseModelConfig{
					OnnxConfig: onnx.Config{
						ModelPath:  filepath.Join("./models", "ch_ppocr_mobile_v2.0_cls_infer.onnx"),
						InputName:  "x",
						OutputName: "save_infer_model/scale_0.tmp_1",
					},
				},
			},
			recognizepaddleocr.ModelName: &recognizepaddleocr.ModelConfig{
				DictPath: filepath.Join("./models", "ppocr_keys_v5.txt"),
				Height:   48,
				Mean:     [3]float64{0.5, 0.5, 0.5},
				Std:      [3]float64{0.5, 0.5, 0.5},
				BaseModelConfig: onnx.BaseModelConfig{
					OnnxConfig: onnx.Config{
						ModelPath:  filepath.Join("./models", "ch_PP-OCRv5_rec_server_infer.onnx"),
						InputName:  "x",
						OutputName: "fetch_name_0",
					},
				},
			},
		},
	}
}

// Load initializes the workflow exactly once.
func (e *Engine) Load() error {
	e.once.Do(func() {
		e.loadErr = e.buildWorkflow()
	})
	return e.loadErr
}

func (e *Engine) buildWorkflow() error {
	var (
		det detect.Detector
		cls classify.Classifier
		rec recognize.Recognizer
		err error
	)

	switch e.config.RecognitionModel {
	case "GLM-OCR":
		if e.config.EnableBoxMerge {
			childDet := detectpaddleocr.NewModel()
			if err = childDet.Init(e.config.Models[detectpaddleocr.ModelName]); err != nil {
				return fmt.Errorf("load child detector: %w", err)
			}
			parentDet := ppdoclayoutv3.NewModel()
			if err = parentDet.Init(e.config.Models[ppdoclayoutv3.ModelName]); err != nil {
				return fmt.Errorf("load parent detector: %w", err)
			}
			merged, mergeErr := boxmerge.NewModel(childDet, parentDet)
			if mergeErr != nil {
				return fmt.Errorf("boxmerge: %w", mergeErr)
			}
			det = merged
		} else {
			parentDet := ppdoclayoutv3.NewModel()
			if err = parentDet.Init(e.config.Models[ppdoclayoutv3.ModelName]); err != nil {
				return fmt.Errorf("load detector: %w", err)
			}
			det = parentDet
		}
		cls = classify.StubClassifier{}
		openaiModel := openairec.NewModel()
		if err = openaiModel.Init(e.config.Models[openairec.ModelName]); err != nil {
			return fmt.Errorf("load recognizer: %w", err)
		}
		rec = openaiModel

	default: // "PaddleOCR"
		if e.config.EnableBoxMerge {
			childDet := detectpaddleocr.NewModel()
			if err = childDet.Init(e.config.Models[detectpaddleocr.ModelName]); err != nil {
				return fmt.Errorf("load child detector: %w", err)
			}
			parentDet := ppdoclayoutv3.NewModel()
			if err = parentDet.Init(e.config.Models[ppdoclayoutv3.ModelName]); err != nil {
				return fmt.Errorf("load parent detector: %w", err)
			}
			merged, mergeErr := boxmerge.NewModel(childDet, parentDet)
			if mergeErr != nil {
				return fmt.Errorf("boxmerge: %w", mergeErr)
			}
			det = merged
		} else {
			plainDet := detectpaddleocr.NewModel()
			if err = plainDet.Init(e.config.Models[detectpaddleocr.ModelName]); err != nil {
				return fmt.Errorf("load detector: %w", err)
			}
			det = plainDet
		}
		plainCls := classifypaddleocr.NewModel()
		if err = plainCls.Init(e.config.Models[classifypaddleocr.ModelName]); err != nil {
			return fmt.Errorf("load classifier: %w", err)
		}
		cls = plainCls
		plainRec := recognizepaddleocr.NewModel()
		if err = plainRec.Init(e.config.Models[recognizepaddleocr.ModelName]); err != nil {
			return fmt.Errorf("load recognizer: %w", err)
		}
		rec = plainRec
	}

	e.workflow = &Workflow{detector: det, classifier: cls, recognizer: rec}
	return nil
}

// Close releases all model resources.
func (e *Engine) Close() error {
	if e.workflow == nil {
		return nil
	}
	var errs []error
	if e.workflow.detector != nil {
		if err := e.workflow.detector.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	if e.workflow.classifier != nil {
		if err := e.workflow.classifier.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	if e.workflow.recognizer != nil {
		if err := e.workflow.recognizer.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	if len(errs) > 0 {
		return errs[0]
	}
	return nil
}

// RunOCR runs the full detect → classify → recognize pipeline on img.
func (e *Engine) RunOCR(img image.Image) ([]Result, error) {
	if err := e.Load(); err != nil {
		return nil, err
	}

	boxes, err := e.workflow.detector.Detect(img)
	if err != nil {
		return nil, fmt.Errorf("detection: %w", err)
	}
	if len(boxes) == 0 {
		return nil, nil
	}

	var results []Result
	for _, box := range boxes {
		if len(box.Children) > 0 {
			r := e.recognizeHierarchical(img, box)
			if r != nil {
				results = append(results, *r)
			}
		} else {
			r := e.recognizeFlat(img, box)
			if r != nil {
				results = append(results, *r)
			}
		}
	}

	return results, nil
}

// recognizeFlat runs cls+rec on a single box (no children).
func (e *Engine) recognizeFlat(img image.Image, box detect.Box) *Result {
	quad := box.Quad

	flip, err := e.workflow.classifier.Classify(img, quad)
	if err != nil {
		flip = false
	}
	if flip {
		quad = [4][2]int{quad[2], quad[3], quad[0], quad[1]}
	}

	res, err := e.workflow.recognizer.Recognize(img, quad)
	if err != nil || res.Text == "" {
		return nil
	}

	box.Text = res.Text
	outBox := [][2]int{quad[0], quad[1], quad[2], quad[3]}
	return &Result{Box: outBox, Text: res.Text, Score: res.Score}
}

// recognizeHierarchical handles a parent box with children.
// RecognitionModel=="GLM-OCR": rec on parent box only.
// Default (PaddleOCR): cls+rec on children, aggregate per parent.
func (e *Engine) recognizeHierarchical(img image.Image, parent detect.Box) *Result {
	parentQuad := parent.Quad
	outBox := [][2]int{parentQuad[0], parentQuad[1], parentQuad[2], parentQuad[3]}

	if e.config.RecognitionModel == "GLM-OCR" {
		res, err := e.workflow.recognizer.Recognize(img, parentQuad)
		if err != nil || res.Text == "" {
			return nil
		}
		return &Result{Box: outBox, Text: res.Text, Score: res.Score}
	}

	// PaddleOCR: classify+recognize each child, aggregate text.
	var childResults []Result
	var texts []string
	var scoreSum float64
	for _, child := range parent.Children {
		r := e.recognizeFlat(img, child)
		if r != nil {
			childResults = append(childResults, *r)
			texts = append(texts, r.Text)
			scoreSum += r.Score
		}
	}
	if len(childResults) == 0 {
		return nil
	}

	avgScore := scoreSum / float64(len(childResults))
	joined := strings.Join(texts, " ")
	return &Result{
		Box:      outBox,
		Text:     joined,
		Score:    avgScore,
		Children: childResults,
	}
}

// DetectOnly runs only the detection model and returns boxes.
func (e *Engine) DetectOnly(img image.Image) ([]detect.Box, error) {
	if err := e.Load(); err != nil {
		return nil, err
	}
	return e.workflow.detector.Detect(img)
}
