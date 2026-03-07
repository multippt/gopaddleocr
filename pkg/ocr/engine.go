package ocr

import (
	"errors"
	"fmt"
	"image"
	"strings"
	"sync"

	"golang.org/x/sync/errgroup"

	"github.com/multippt/gopaddleocr/pkg/ocr/classify"
	classifypaddleocr "github.com/multippt/gopaddleocr/pkg/ocr/classify/paddleocr"
	"github.com/multippt/gopaddleocr/pkg/ocr/common"
	"github.com/multippt/gopaddleocr/pkg/ocr/detect"
	"github.com/multippt/gopaddleocr/pkg/ocr/detect/boxmerge"
	detectpaddleocr "github.com/multippt/gopaddleocr/pkg/ocr/detect/paddleocr"
	ppdoclayoutv3 "github.com/multippt/gopaddleocr/pkg/ocr/detect/pp-doclayoutv3"
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
	Models           map[string]common.ModelConfig
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
	once    sync.Once
	loadErr error
	models  map[string]common.Model

	workflow *Workflow
	config   *Config
}

// NewEngine creates an Engine with the given config.
func NewEngine() *Engine {
	e := &Engine{
		config: &Config{
			RecognitionModel: "PaddleOCR",
			EnableBoxMerge:   false,
			Models:           make(map[string]common.ModelConfig),
		},
		models: make(map[string]common.Model),
	}

	// Declare known models.
	parentDetModel := ppdoclayoutv3.NewModel()
	childDetModel := detectpaddleocr.NewModel()
	e.RegisterModel(parentDetModel)
	e.RegisterModel(childDetModel)
	e.RegisterModel(boxmerge.NewModel(parentDetModel, childDetModel))
	e.RegisterModel(classifypaddleocr.NewModel())
	e.RegisterModel(recognizepaddleocr.NewModel())
	e.RegisterModel(openairec.NewModel())

	return e
}

func (e *Engine) RegisterModel(model common.Model) {
	e.models[model.GetName()] = model
	e.config.Models[model.GetName()] = model.GetDefaultConfig()
}

// Close releases all model resources.
func (e *Engine) Close() error {
	var eg errgroup.Group
	for _, model := range e.models {
		eg.Go(model.Close)
	}
	return eg.Wait()
}

// Init initializes the workflow and models.
func (e *Engine) Init() error {
	e.once.Do(func() {
		var eg errgroup.Group
		eg.Go(e.buildWorkflow)
		for _, model := range e.models {
			eg.Go(func() error {
				return model.Init(e.config.Models[model.GetName()])
			})
		}
		e.loadErr = eg.Wait()
	})
	return e.loadErr
}

func (e *Engine) buildWorkflow() error {
	w := &Workflow{}

	switch e.config.RecognitionModel {
	case "GLM-OCR":
		if e.config.EnableBoxMerge {
			w.detector = e.models[boxmerge.ModelName].(detect.Detector)
		} else {
			w.detector = e.models[ppdoclayoutv3.ModelName].(detect.Detector)
		}
		w.classifier = &classify.StubClassifier{}
		w.recognizer = e.models[openairec.ModelName].(recognize.Recognizer)
	case "PaddleOCR":
		if e.config.EnableBoxMerge {
			w.detector = e.models[boxmerge.ModelName].(detect.Detector)
		} else {
			w.detector = e.models[detectpaddleocr.ModelName].(detect.Detector)
		}
		w.classifier = e.models[classifypaddleocr.ModelName].(classify.Classifier)
		w.recognizer = e.models[recognizepaddleocr.ModelName].(recognize.Recognizer)
	default: // "PaddleOCR"
		return errors.New("unknown recognition model")
	}

	e.workflow = w
	return nil
}

// RunOCR runs the full detect → classify → recognize pipeline on img.
func (e *Engine) RunOCR(img image.Image) ([]Result, error) {
	if err := e.Init(); err != nil {
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
	if err := e.Init(); err != nil {
		return nil, err
	}
	return e.workflow.detector.Detect(img)
}
