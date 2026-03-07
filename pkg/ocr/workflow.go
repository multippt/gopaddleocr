package ocr

import (
	"fmt"
	"image"
	"strings"
	"sync"

	"github.com/multippt/gopaddleocr/pkg/ocr/classify"
	"github.com/multippt/gopaddleocr/pkg/ocr/common"
	"github.com/multippt/gopaddleocr/pkg/ocr/detect"
	"github.com/multippt/gopaddleocr/pkg/ocr/recognize"
	"github.com/multippt/gopaddleocr/pkg/ocr/utils"
	"golang.org/x/sync/errgroup"
)

// Workflow holds the three-stage pipeline components.
type Workflow struct {
	name       string
	detector   detect.Detector
	classifier classify.Classifier
	recognizer recognize.Recognizer

	configSrc common.ConfigSource
	initErr   error
	once      sync.Once
}

type WorkflowFactory func(engineConf *Config, configSrc common.ConfigSource) *Workflow

func NewWorkflow(
	detector detect.Detector,
	classifier classify.Classifier,
	recognizer recognize.Recognizer,
	configSrc common.ConfigSource,
) *Workflow {
	return &Workflow{
		detector:   detector,
		classifier: classifier,
		recognizer: recognizer,
		configSrc:  configSrc,
	}
}

func (w *Workflow) initModelWrapper(model common.Model) func() error {
	return func() error {
		return model.Init(w.configSrc)
	}
}

// Init initializes the workflow and knownModels.
func (w *Workflow) Init() error {
	w.once.Do(func() {
		var eg errgroup.Group
		eg.Go(w.initModelWrapper(w.detector))
		eg.Go(w.initModelWrapper(w.classifier))
		eg.Go(w.initModelWrapper(w.recognizer))
		w.initErr = eg.Wait()
	})
	return w.initErr
}

// Close releases all model resources.
func (w *Workflow) Close() error {
	var eg errgroup.Group
	eg.Go(w.detector.Close)
	eg.Go(w.classifier.Close)
	eg.Go(w.recognizer.Close)
	return eg.Wait()
}

// RunOCR runs the detection -> classify -> recognize pipeline on img.
func (w *Workflow) RunOCR(img image.Image) ([]Result, error) {
	if err := w.Init(); err != nil {
		return nil, err
	}

	// Get the bounding boxes for the text.
	boxes, err := w.detector.Detect(img)
	if err != nil {
		return nil, fmt.Errorf("detection: %w", err)
	}
	if len(boxes) == 0 {
		return nil, nil
	}

	var results []Result
	for _, box := range boxes {
		if len(box.Children) > 0 {
			r := w.recognizeHierarchical(img, box)
			if r != nil {
				results = append(results, *r)
			}
		} else {
			// Not using box merge, all elements would be flat hierarchy.
			r := w.recognizeFlat(img, box)
			if r != nil {
				results = append(results, *r)
			}
		}
	}

	return results, nil
}

// recognizeFlat runs cls+rec on a single box (no children).
func (w *Workflow) recognizeFlat(img image.Image, box utils.Box) *Result {
	quad := box.Quad

	flip, err := w.classifier.Classify(img, quad)
	if err != nil {
		flip = false
	}
	if flip {
		quad = [4][2]int{quad[2], quad[3], quad[0], quad[1]}
	}

	res, err := w.recognizer.Recognize(img, quad)
	if err != nil || res.Text == "" {
		return nil
	}

	box.Text = res.Text
	outBox := [][2]int{quad[0], quad[1], quad[2], quad[3]}
	return &Result{Box: outBox, Text: res.Text, Score: res.Score}
}

// recognizeHierarchical handles a parent box with children.
// WorkflowType=="GLM-OCR": rec on parent box only.
// Default (PaddleOCR): cls+rec on children, aggregate per parent.
func (w *Workflow) recognizeHierarchical(img image.Image, parent utils.Box) *Result {
	parentQuad := parent.Quad
	outBox := [][2]int{parentQuad[0], parentQuad[1], parentQuad[2], parentQuad[3]}

	if !w.recognizer.RecognizeLineOnly() {
		// This model can operate on multi-line context directly.
		return w.recognizeFlat(img, parent)
	}

	// PaddleOCR: classify+recognize each child, aggregate text.
	var childResults []Result
	var texts []string
	var scoreSum float64
	for _, child := range parent.Children {
		r := w.recognizeFlat(img, child)
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
func (w *Workflow) DetectOnly(img image.Image) ([]utils.Box, error) {
	if err := w.Init(); err != nil {
		return nil, err
	}
	return w.detector.Detect(img)
}
