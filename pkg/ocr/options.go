package ocr

import "github.com/multippt/gopaddleocr/pkg/ocr/common"

// Option configures an Engine before initialization.
type Option func(*Engine)

// WithModelConfig sets the model config for the named model.
func WithModelConfig(modelName string, config common.ModelConfig) Option {
	return func(e *Engine) {
		e.config.Models[modelName] = config
	}
}

// WithWorkflowType sets the workflow type ("PaddleOCR" or "GLM-OCR").
func WithWorkflowType(workflowName string) Option {
	return func(e *Engine) {
		e.config.WorkflowType = workflowName
	}
}

// WithBoxMerge enables or disables box merging.
func WithBoxMerge(boxMerge bool) Option {
	return func(e *Engine) {
		e.config.EnableBoxMerge = boxMerge
	}
}
