package common

import (
	"fmt"

	ort "github.com/yalue/onnxruntime_go"
)

type Model interface {
	GetName() string
	Init(config ModelConfig) error
	GetDefaultConfig() ModelConfig
	Close() error
}

type Config struct {
	ModelPath string
	Options   *ort.SessionOptions
}

// InputOutputNames reads the input and output names directly from the ONNX
// model file, so they do not need to be specified in config.
func InputOutputNames(modelPath string, opts *ort.SessionOptions) ([]string, []string, error) {
	inputs, outputs, err := ort.GetInputOutputInfoWithOptions(modelPath, opts)
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
