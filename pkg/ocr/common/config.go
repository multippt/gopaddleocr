package common

import (
	"fmt"
	"os"
	"path/filepath"

	ort "github.com/yalue/onnxruntime_go"
)

const defaultModelPath = "models"

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

// guessProjectRoot tries to find the project root path if running in an inner directory.
func guessProjectRoot() string {
	dir, err := os.Getwd()
	if err != nil {
		return ""
	}

	// Walk upward to the module root (directory containing go.mod).
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			break
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			// Reached fs root without finding go.mod
			return ""
		}
		dir = parent
	}
	return dir
}

func GetModelPath(name string) string {
	modelPath := os.Getenv("MODELS_DIR")
	if modelPath != "" {
		// User provided an explicit path.
		return filepath.Join(modelPath, name)
	}

	// Default
	modelPath = filepath.Join(defaultModelPath, name)
	if fileExists(modelPath) {
		return modelPath
	}

	// Some IDE may set a different CWD if running a unit test.
	modelPath = filepath.Join(guessProjectRoot(), modelPath)
	if fileExists(modelPath) {
		return modelPath
	}
	return name
}

type Model interface {
	GetName() string
	Init(config ConfigSource) error
	GetDefaultConfig() ModelConfig
	Close() error
}

type ConfigSource interface {
	GetConfig(modelName string) ModelConfig
}

type Config struct {
	ModelPath string
	Options   *ort.SessionOptions

	resolvedPath bool
}

func (c *Config) GetModelPath() string {
	if c.resolvedPath {
		return c.ModelPath
	}
	c.ModelPath = GetModelPath(c.ModelPath)
	c.resolvedPath = true
	return c.ModelPath
}

// getInputOutputNames reads the input and output names directly from the ONNX model file.
func (c *Config) getInputOutputNames() ([]string, []string, error) {
	modelPath := c.GetModelPath()

	inputs, outputs, err := ort.GetInputOutputInfoWithOptions(modelPath, c.Options)
	if err != nil {
		return nil, nil, fmt.Errorf("reading model I/O names from %q: %w", modelPath, err)
	}
	inputNames := make([]string, len(inputs))
	for i, info := range inputs {
		inputNames[i] = info.Name
	}
	outputNames := make([]string, len(outputs))
	for i, info := range outputs {
		outputNames[i] = info.Name
	}
	return inputNames, outputNames, nil
}

func (c *Config) GetORTSession() (*ort.DynamicAdvancedSession, error) {
	inputNames, outputNames, err := c.getInputOutputNames()
	if err != nil {
		return nil, err
	}

	return ort.NewDynamicAdvancedSession(c.GetModelPath(),
		inputNames,
		outputNames,
		c.Options)
}

// ModelConfig is implemented by every model-specific config struct.
type ModelConfig interface {
	GetOnnxConfig() *Config
	SetOnnxConfig(conf *Config)
}

// BaseModelConfig provides a default implementation of ModelConfig.
// Embed this into model-specific config structs.
type BaseModelConfig struct {
	OnnxConfig Config
}

func (b *BaseModelConfig) GetOnnxConfig() *Config  { return &b.OnnxConfig }
func (b *BaseModelConfig) SetOnnxConfig(c *Config) { b.OnnxConfig = *c }
