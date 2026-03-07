package ocr

import (
	"bytes"
	"errors"
	"image"
	"os"
	"runtime"
	"sync"

	"github.com/multippt/gopaddleocr/pkg/ocr/common"
	"github.com/multippt/gopaddleocr/pkg/ocr/utils"
	ort "github.com/yalue/onnxruntime_go"
)

// DefaultORTLibPath is the default path to the ONNX Runtime shared library (set by init from GOOS).
var DefaultORTLibPath string = getDefaultORTLibPath()

func getDefaultORTLibPath() string {
	switch runtime.GOOS {
	case "windows":
		return "./onnxruntime/lib/onnxruntime.dll"
	case "darwin":
		return "./onnxruntime/lib/libonnxruntime.dylib"
	default:
		// linux and other unix-like
		return "./onnxruntime/lib/libonnxruntime.so"
	}
}

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
	Models         map[string]common.ModelConfig
	EnableBoxMerge bool
	WorkflowType   string // "PaddleOCR" (default) or "GLM-OCR"
}

// Engine coordinates the detect → classify → recognize pipeline.
type Engine struct {
	once    sync.Once
	loadErr error

	knownWorkflows map[string]WorkflowFactory
	workflow       *Workflow
	config         *Config
}

// NewEngine creates an Engine with the given config.
func NewEngine(opts ...Option) *Engine {
	e := &Engine{
		config: &Config{
			WorkflowType:   "PaddleOCR",
			EnableBoxMerge: true,
			Models:         make(map[string]common.ModelConfig),
		},
		knownWorkflows: make(map[string]WorkflowFactory),
	}

	e.RegisterWorkflow("PaddleOCR", NewPaddleOCRWorkflow)
	e.RegisterWorkflow("GLM-OCR", NewGLMOCRWorkflow)

	for _, opt := range opts {
		opt(e)
	}

	return e
}

// Close releases all model resources.
func (e *Engine) Close() error {
	var err error
	if e.workflow != nil {
		err = e.workflow.Close()
	}
	if err != nil {
		return err
	}

	if ort.IsInitialized() {
		err = ort.DestroyEnvironment()
		if err != nil {
			return err
		}
	}

	return nil
}

func getEnv(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// Init initializes the workflow and knownModels.
func (e *Engine) Init() error {
	var err error

	ortPath := getEnv("ORT_LIB_PATH", DefaultORTLibPath)
	ort.SetSharedLibraryPath(ortPath)
	if !ort.IsInitialized() {
		if err := ort.InitializeEnvironment(); err != nil {
			return err
		}
	}

	if e.workflow == nil {
		e.workflow, err = e.getWorkflow()
		if err != nil {
			return err
		}
	}
	return e.workflow.Init()
}

func (e *Engine) GetConfig(modelName string) common.ModelConfig {
	return e.config.Models[modelName]
}

func (e *Engine) RegisterWorkflow(name string, factory WorkflowFactory) {
	e.knownWorkflows[name] = factory
}

func (e *Engine) getWorkflow() (*Workflow, error) {
	factory, ok := e.knownWorkflows[e.config.WorkflowType]
	if !ok {
		return nil, errors.New("unknown workflow")
	}
	return factory(e.config, e), nil
}

func (e *Engine) decodeImage(data []byte) (image.Image, error) {
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	return img, err
}

func (e *Engine) RunOCR(data []byte) ([]Result, error) {
	img, err := e.decodeImage(data)
	if err != nil {
		return nil, err
	}
	return e.workflow.RunOCR(img)
}

func (e *Engine) ImageRunOCR(img image.Image) ([]Result, error) {
	return e.workflow.RunOCR(img)
}

func (e *Engine) DetectBoundingBoxes(data []byte) ([]utils.Box, error) {
	img, err := e.decodeImage(data)
	if err != nil {
		return nil, err
	}
	return e.workflow.DetectOnly(img)
}

func (e *Engine) ImageDetectBoundingBoxes(img image.Image) ([]utils.Box, error) {
	return e.workflow.DetectOnly(img)
}
