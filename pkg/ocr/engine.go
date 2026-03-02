package ocr

import (
	"fmt"
	"image"
	"path/filepath"
	"sync"

	"github.com/multippt/gopaddleocr/pkg/ocr/classify"
	"github.com/multippt/gopaddleocr/pkg/ocr/detect"
	"github.com/multippt/gopaddleocr/pkg/ocr/onnx"
	"github.com/multippt/gopaddleocr/pkg/ocr/recognize"
)

// Result is a single detected text region.
type Result struct {
	Box   [][2]int `json:"box"` // [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
	Text  string   `json:"text"`
	Score float64  `json:"score"`
}

type Config struct {
	DetectModelConfig    *detect.ModelConfig
	ClassifyModelConfig  *classify.ModelConfig
	RecognizeModelConfig *recognize.ModelConfig
}

func NewDefaultConfig() *Config {
	return &Config{
		DetectModelConfig: &detect.ModelConfig{
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
		ClassifyModelConfig: &classify.ModelConfig{
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
		RecognizeModelConfig: &recognize.ModelConfig{
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

	det *detect.Model
	cls *classify.Model
	rec *recognize.Model

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
	if e.config.DetectModelConfig == nil {
		return fmt.Errorf("load det model: DetectModelConfig is nil")
	}
	if e.config.ClassifyModelConfig == nil {
		return fmt.Errorf("load cls model: ClassifyModelConfig is nil")
	}
	if e.config.RecognizeModelConfig == nil {
		return fmt.Errorf("load rec model: RecognizeModelConfig is nil")
	}

	var err error
	e.det, err = detect.NewModel(e.config.DetectModelConfig)
	if err != nil {
		return fmt.Errorf("load det model: %w", err)
	}

	e.cls, err = classify.NewModel(e.config.ClassifyModelConfig)
	if err != nil {
		return fmt.Errorf("load cls model: %w", err)
	}

	e.rec, err = recognize.NewModel(e.config.RecognizeModelConfig)
	if err != nil {
		return fmt.Errorf("load rec model: %w", err)
	}

	return nil
}

// RunOCR runs the full detect → classify → recognize pipeline on img.
func (e *PaddleOCREngine) RunOCR(img image.Image) ([]Result, error) {
	if err := e.Load(); err != nil {
		return nil, err
	}

	quads, err := e.det.Run(img)
	if err != nil {
		return nil, fmt.Errorf("detection: %w", err)
	}
	if len(quads) == 0 {
		return nil, nil
	}

	var results []Result
	for _, quad := range quads {
		// Classification: determine if crop needs 180° flip.
		flip, err := e.cls.Run(img, quad)
		if err != nil {
			flip = false // non-fatal
		}
		if flip {
			// Reverse point order → 180° rotation in perspective warp.
			quad = [4][2]int{quad[2], quad[3], quad[0], quad[1]}
		}

		text, score, err := e.rec.Run(img, quad)
		if err != nil {
			continue // skip unreadable region
		}
		if text == "" {
			continue
		}

		box := [][2]int{quad[0], quad[1], quad[2], quad[3]}
		results = append(results, Result{Box: box, Text: text, Score: score})
	}

	return results, nil
}

// DetectOnly runs only the detection model and returns quads.
func (e *PaddleOCREngine) DetectOnly(img image.Image) ([][4][2]int, error) {
	if err := e.Load(); err != nil {
		return nil, err
	}
	return e.det.Run(img)
}
