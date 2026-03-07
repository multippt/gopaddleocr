package paddleocr

import (
	"fmt"
	"image"
	"path/filepath"

	"github.com/multippt/gopaddleocr/pkg/ocr/common"
	"github.com/multippt/gopaddleocr/pkg/ocr/utils"
	ort "github.com/yalue/onnxruntime_go"
)

const ModelName = "paddleocr-cls"

type ModelConfig struct {
	Height    int
	Width     int
	Threshold float32
	Mean      [3]float64
	Std       [3]float64

	common.BaseModelConfig
}

// ---------------------------------------------------------------------------
// Model wraps the direction-classification ONNX session
// ---------------------------------------------------------------------------

type Model struct {
	session *ort.DynamicAdvancedSession
	config  *ModelConfig
}

func NewModel() *Model {
	return &Model{}
}

func (m *Model) GetName() string { return ModelName }

func (m *Model) GetDefaultConfig() common.ModelConfig {
	return &ModelConfig{
		Height:    48,
		Width:     192,
		Threshold: 0.9,
		Mean:      [3]float64{0.5, 0.5, 0.5},
		Std:       [3]float64{0.5, 0.5, 0.5},
		BaseModelConfig: common.BaseModelConfig{
			OnnxConfig: common.Config{
				ModelPath: filepath.Join("./models", "ch_ppocr_mobile_v2.0_cls_infer.onnx"),
			},
		},
	}
}

func (m *Model) Init(config common.ModelConfig) error {
	cfg, ok := config.(*ModelConfig)
	if !ok {
		return fmt.Errorf("classify/paddleocr: expected *ModelConfig, got %T", config)
	}
	m.config = cfg
	inputNames, outputNames, err := common.InputOutputNames(cfg.OnnxConfig.ModelPath, cfg.OnnxConfig.Options)
	if err != nil {
		return err
	}
	session, err := ort.NewDynamicAdvancedSession(cfg.OnnxConfig.ModelPath,
		inputNames,
		outputNames,
		cfg.OnnxConfig.Options)
	if err != nil {
		return err
	}
	m.session = session
	return nil
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
