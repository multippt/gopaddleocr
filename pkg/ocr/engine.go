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
	detectpaddleocr "github.com/multippt/gopaddleocr/pkg/ocr/detect/paddleocr"
	"github.com/multippt/gopaddleocr/pkg/ocr/onnx"
	"github.com/multippt/gopaddleocr/pkg/ocr/recognize"
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

type Config struct {
	DetectModelConfig    *detectpaddleocr.ModelConfig
	ClassifyModelConfig  *classifypaddleocr.ModelConfig
	RecognizeModelConfig *recognizepaddleocr.ModelConfig

	// Optional overrides — if set, bypass model-path construction.
	Detector   detect.Detector
	Classifier classify.Classifier
	Recognizer recognize.Recognizer

	// RecognizeParent controls behavior when detection returns hierarchical boxes.
	// When false (default, PaddleOCR workflow): cls+rec operate on child boxes,
	// text is aggregated per parent.
	// When true (GLM-OCR workflow): rec operates on parent boxes only, children ignored.
	RecognizeParent bool
}

func NewDefaultConfig() *Config {
	return &Config{
		DetectModelConfig: &detectpaddleocr.ModelConfig{
			ModelPath:       filepath.Join("./models", "ch_PP-OCRv5_server_det.onnx"),
			LimitSideLength: 1280,
			Mean:            [3]float32{0.485, 0.456, 0.406},
			Std:             [3]float32{0.229, 0.224, 0.225},
			Thresh:          0.3,
			BoxThresh:       0.6,
			UnclipRatio:     2.0,
			MinArea:         16,
			OnnxConfig: onnx.Config{
				InputName:  "x",
				OutputName: "fetch_name_0",
			},
		},
		ClassifyModelConfig: &classifypaddleocr.ModelConfig{
			ModelPath: filepath.Join("./models", "ch_ppocr_mobile_v2.0_cls_infer.onnx"),
			Height:    48,
			Width:     192,
			Threshold: 0.9,
			Mean:      [3]float64{0.5, 0.5, 0.5},
			Std:       [3]float64{0.5, 0.5, 0.5},
			OnnxConfig: onnx.Config{
				InputName:  "x",
				OutputName: "save_infer_model/scale_0.tmp_1",
			},
		},
		RecognizeModelConfig: &recognizepaddleocr.ModelConfig{
			ModelPath: filepath.Join("./models", "ch_PP-OCRv5_rec_server_infer.onnx"),
			DictPath:  filepath.Join("./models", "ppocr_keys_v5.txt"),
			Height:    48,
			Mean:      [3]float64{0.5, 0.5, 0.5},
			Std:       [3]float64{0.5, 0.5, 0.5},
			OnnxConfig: onnx.Config{
				InputName:  "x",
				OutputName: "fetch_name_0",
			},
		},
	}
}

var DefaultConfig = NewDefaultConfig()

// Option mutates the engine Config (used by NewPaddleOCREngine).
type Option func(*Config)

// WithConfig sets the entire config. Use for full custom config or tests.
func WithConfig(cfg *Config) Option {
	return func(c *Config) {
		if cfg != nil {
			*c = *cfg
			if cfg.DetectModelConfig != nil {
				detCfg := *cfg.DetectModelConfig
				c.DetectModelConfig = &detCfg
			}
			if cfg.ClassifyModelConfig != nil {
				clsCfg := *cfg.ClassifyModelConfig
				c.ClassifyModelConfig = &clsCfg
			}
			if cfg.RecognizeModelConfig != nil {
				recCfg := *cfg.RecognizeModelConfig
				c.RecognizeModelConfig = &recCfg
			}
		}
	}
}

// WithDetectModelPath sets the detection model path.
func WithDetectModelPath(path string) Option {
	return func(c *Config) {
		if c.DetectModelConfig != nil {
			c.DetectModelConfig.ModelPath = path
		}
	}
}

// WithClassifyModelPath sets the classification model path.
func WithClassifyModelPath(path string) Option {
	return func(c *Config) {
		if c.ClassifyModelConfig != nil {
			c.ClassifyModelConfig.ModelPath = path
		}
	}
}

// WithRecognizeModelPath sets the recognition model path.
func WithRecognizeModelPath(path string) Option {
	return func(c *Config) {
		if c.RecognizeModelConfig != nil {
			c.RecognizeModelConfig.ModelPath = path
		}
	}
}

// WithRecognizeDictPath sets the recognition dictionary path.
func WithRecognizeDictPath(path string) Option {
	return func(c *Config) {
		if c.RecognizeModelConfig != nil {
			c.RecognizeModelConfig.DictPath = path
		}
	}
}

// WithDetector overrides the detector; bypasses DetectModelConfig construction.
func WithDetector(d detect.Detector) Option {
	return func(c *Config) {
		c.Detector = d
	}
}

// WithClassifier overrides the classifier; bypasses ClassifyModelConfig construction.
func WithClassifier(cls classify.Classifier) Option {
	return func(c *Config) {
		c.Classifier = cls
	}
}

// WithRecognizer overrides the recognizer; bypasses RecognizeModelConfig construction.
func WithRecognizer(r recognize.Recognizer) Option {
	return func(c *Config) {
		c.Recognizer = r
	}
}

// WithRecognizeParent sets whether recognition operates on parent boxes (true)
// or child boxes (false). Use true for GLM-OCR workflows, false for PaddleOCR.
func WithRecognizeParent(v bool) Option {
	return func(c *Config) {
		c.RecognizeParent = v
	}
}

// defaultConfigCopy returns a deep copy of DefaultConfig so callers can mutate it via options.
func defaultConfigCopy() *Config {
	cfg := &Config{}
	if DefaultConfig.DetectModelConfig != nil {
		det := *DefaultConfig.DetectModelConfig
		cfg.DetectModelConfig = &det
	}
	if DefaultConfig.ClassifyModelConfig != nil {
		cls := *DefaultConfig.ClassifyModelConfig
		cfg.ClassifyModelConfig = &cls
	}
	if DefaultConfig.RecognizeModelConfig != nil {
		rec := *DefaultConfig.RecognizeModelConfig
		cfg.RecognizeModelConfig = &rec
	}
	return cfg
}

// PaddleOCREngine coordinates the detect → classify → recognize pipeline.
type PaddleOCREngine struct {
	once    sync.Once
	loadErr error

	det detect.Detector
	cls classify.Classifier
	rec recognize.Recognizer

	config *Config
}

func NewPaddleOCREngine(opts ...Option) *PaddleOCREngine {
	cfg := defaultConfigCopy()
	for _, opt := range opts {
		opt(cfg)
	}
	return &PaddleOCREngine{
		config: cfg,
	}
}

// Load loads all three ONNX models exactly once.
// It is safe to call from multiple goroutines; only the first call does work.
func (e *PaddleOCREngine) Load() error {
	e.once.Do(func() {
		e.loadErr = e.doLoad()
	})
	return e.loadErr
}

func (e *PaddleOCREngine) doLoad() error {
	var err error

	if e.config.Detector != nil {
		e.det = e.config.Detector
	} else {
		if e.config.DetectModelConfig == nil {
			return fmt.Errorf("load det model: DetectModelConfig is nil")
		}
		e.det, err = detectpaddleocr.NewModel(e.config.DetectModelConfig)
		if err != nil {
			return fmt.Errorf("load det model: %w", err)
		}
	}

	if e.config.Classifier != nil {
		e.cls = e.config.Classifier
	} else {
		if e.config.ClassifyModelConfig == nil {
			return fmt.Errorf("load cls model: ClassifyModelConfig is nil")
		}
		e.cls, err = classifypaddleocr.NewModel(e.config.ClassifyModelConfig)
		if err != nil {
			return fmt.Errorf("load cls model: %w", err)
		}
	}

	if e.config.Recognizer != nil {
		e.rec = e.config.Recognizer
	} else {
		if e.config.RecognizeModelConfig == nil {
			return fmt.Errorf("load rec model: RecognizeModelConfig is nil")
		}
		e.rec, err = recognizepaddleocr.NewModel(e.config.RecognizeModelConfig)
		if err != nil {
			return fmt.Errorf("load rec model: %w", err)
		}
	}

	return nil
}

// Close releases all model resources.
func (e *PaddleOCREngine) Close() error {
	var errs []error
	if e.det != nil {
		if err := e.det.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	if e.cls != nil {
		if err := e.cls.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	if e.rec != nil {
		if err := e.rec.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	if len(errs) > 0 {
		return errs[0]
	}
	return nil
}

// RunOCR runs the full detect → classify → recognize pipeline on img.
func (e *PaddleOCREngine) RunOCR(img image.Image) ([]Result, error) {
	if err := e.Load(); err != nil {
		return nil, err
	}

	boxes, err := e.det.Detect(img)
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
func (e *PaddleOCREngine) recognizeFlat(img image.Image, box detect.Box) *Result {
	quad := box.Quad

	flip, err := e.cls.Classify(img, quad)
	if err != nil {
		flip = false
	}
	if flip {
		quad = [4][2]int{quad[2], quad[3], quad[0], quad[1]}
	}

	res, err := e.rec.Recognize(img, quad)
	if err != nil || res.Text == "" {
		return nil
	}

	outBox := [][2]int{quad[0], quad[1], quad[2], quad[3]}
	return &Result{Box: outBox, Text: res.Text, Score: res.Score}
}

// recognizeHierarchical handles a parent box with children.
// RecognizeParent=false (PaddleOCR): cls+rec on children, aggregate per parent.
// RecognizeParent=true (GLM-OCR): rec on parent box only.
func (e *PaddleOCREngine) recognizeHierarchical(img image.Image, parent detect.Box) *Result {
	parentQuad := parent.Quad
	outBox := [][2]int{parentQuad[0], parentQuad[1], parentQuad[2], parentQuad[3]}

	if e.config.RecognizeParent {
		// GLM-OCR: recognize parent box directly, ignore children.
		res, err := e.rec.Recognize(img, parentQuad)
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
func (e *PaddleOCREngine) DetectOnly(img image.Image) ([]detect.Box, error) {
	if err := e.Load(); err != nil {
		return nil, err
	}
	return e.det.Detect(img)
}
