package paddleocr

import (
	"fmt"
	"image"

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
	*common.OnnxModel
}

func NewModel() *Model {
	m := &Model{}
	m.OnnxModel = common.NewOnnxModel(m)
	return m
}

func (m *Model) GetName() string { return ModelName }

func (m *Model) GetDefaultConfig() common.ModelConfig {
	return &ModelConfig{
		Height:    80,
		Width:     160,
		Threshold: 0.9,
		Mean:      [3]float64{0.485, 0.456, 0.406},
		Std:       [3]float64{0.229, 0.224, 0.225},
		BaseModelConfig: common.BaseModelConfig{
			OnnxConfig: common.Config{
				ModelPath: "PP-LCNet_x1_0_textline_ori_infer.onnx",
			},
		},
	}
}

// Classify returns true if the crop needs 180° rotation.
func (m *Model) Classify(img image.Image, quad [4][2]int) (bool, error) {
	config, ok := m.GetDefaultConfig().(*ModelConfig)
	if !ok {
		return false, common.ErrInvalidConfig
	}

	crop := utils.PerspectiveWarp(img, utils.FloatQuad(quad), config.Width, config.Height)
	data := utils.ImageToNCHW(crop, config.Height, config.Width, config.Mean, config.Std)

	shape := ort.NewShape(1, 3, int64(config.Height), int64(config.Width))
	inTensor, err := ort.NewTensor(shape, data)
	if err != nil {
		return false, fmt.Errorf("cls input tensor: %w", err)
	}
	defer func() {
		_ = inTensor.Destroy()
	}()

	outputs := make([]ort.Value, 1)
	if err := m.GetSession().Run([]ort.Value{inTensor}, outputs); err != nil {
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
	return logits[1] > logits[0] && logits[1] > config.Threshold, nil
}
