package ocr

import (
	"fmt"
	"image"
	"path/filepath"
	"sync"

	"github.com/multippt/gopaddleocr/pkg/ocr/classify"
	"github.com/multippt/gopaddleocr/pkg/ocr/detect"
	"github.com/multippt/gopaddleocr/pkg/ocr/recognize"
)

// Result is a single detected text region.
type Result struct {
	Box   [][2]int `json:"box"` // [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
	Text  string   `json:"text"`
	Score float64  `json:"score"`
}

type Config struct {
	DetectModelPath    string
	ClassifyModelPath  string
	RecognizeModelPath string
	RecognizeDictPath  string
}

var DefaultConfig = &Config{
	DetectModelPath:    filepath.Join("./models", "ch_PP-OCRv5_server_det.onnx"),
	ClassifyModelPath:  filepath.Join("./models", "ch_ppocr_mobile_v2.0_cls_infer.onnx"),
	RecognizeModelPath: filepath.Join("./models", "ch_PP-OCRv5_rec_server_infer.onnx"),
	RecognizeDictPath:  filepath.Join("./models", "ppocr_keys_v5.txt"),
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

func NewPaddleOCREngine() *PaddleOCREngine {
	return &PaddleOCREngine{
		config: DefaultConfig,
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

	e.det, err = detect.NewModel(e.config.DetectModelPath)
	if err != nil {
		return fmt.Errorf("load det model: %w", err)
	}

	e.cls, err = classify.NewModel(e.config.ClassifyModelPath)
	if err != nil {
		return fmt.Errorf("load cls model: %w", err)
	}

	e.rec, err = recognize.NewModel(e.config.RecognizeModelPath, e.config.DetectModelPath)
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
