package onnx

import ort "github.com/yalue/onnxruntime_go"

type Config struct {
	ModelPath  string
	InputName  string
	OutputName string
	Options    *ort.SessionOptions
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
