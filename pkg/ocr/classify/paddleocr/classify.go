package paddleocr

import (
	"fmt"
	"image"

	"github.com/multippt/gopaddleocr/pkg/ocr/onnx"
	"github.com/multippt/gopaddleocr/pkg/ocr/utils"
	ort "github.com/yalue/onnxruntime_go"
)

type ModelConfig struct {
	ModelPath string

	Height    int
	Width     int
	Threshold float32
	Mean      [3]float64
	Std       [3]float64

	OnnxConfig onnx.Config
}

// ---------------------------------------------------------------------------
// Model wraps the direction-classification ONNX session
// ---------------------------------------------------------------------------

type Model struct {
	session *ort.DynamicAdvancedSession
	config  *ModelConfig
}

func NewModel(cfg *ModelConfig) (*Model, error) {
	if cfg == nil {
		return nil, fmt.Errorf("classify: config is required")
	}
	m := &Model{config: cfg}
	session, err := ort.NewDynamicAdvancedSession(cfg.ModelPath,
		[]string{cfg.OnnxConfig.InputName},
		[]string{cfg.OnnxConfig.OutputName},
		cfg.OnnxConfig.Options)
	if err != nil {
		return nil, err
	}
	m.session = session
	return m, nil
}

func (m *Model) Close() error {
	if m.session != nil {
		return m.session.Destroy()
	}
	return nil
}

// Classify returns true if the crop needs 180° rotation.
func (m *Model) Classify(img image.Image, quad [4][2]int) (bool, error) {
	crop := utils.PerspectiveWarp(img, utils.FloatQuad(quad), m.config.Width, m.config.Height)
	if crop == nil {
		return false, nil
	}

	data := utils.ImageToNCHW(crop, m.config.Height, m.config.Width, m.config.Mean, m.config.Std)

	shape := ort.NewShape(1, 3, int64(m.config.Height), int64(m.config.Width))
	inTensor, err := ort.NewTensor(shape, data)
	if err != nil {
		return false, fmt.Errorf("cls input tensor: %w", err)
	}
	defer func() {
		_ = inTensor.Destroy()
	}()

	outputs := make([]ort.Value, 1)
	if err := m.session.Run([]ort.Value{inTensor}, outputs); err != nil {
		return false, fmt.Errorf("cls inference: %w", err)
	}
	outTensor, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		_ = outputs[0].Destroy()
		return false, fmt.Errorf("unexpected cls output type")
	}
	defer func() {
		_ = outTensor.Destroy()
	}()

	logits := outTensor.GetData()
	// logits shape: (1, 2) — class 0 = upright, class 1 = 180° rotated.
	if len(logits) < 2 {
		return false, nil
	}
	return logits[1] > logits[0] && logits[1] > m.config.Threshold, nil
}
