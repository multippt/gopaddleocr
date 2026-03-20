package paddleocr

import (
	"fmt"
	"image"
	"math"
	"path/filepath"

	"github.com/multippt/gopaddleocr/pkg/ocr/common"
	"github.com/multippt/gopaddleocr/pkg/ocr/recognize"
	"github.com/multippt/gopaddleocr/pkg/ocr/utils"
	"github.com/pkg/errors"
	ort "github.com/yalue/onnxruntime_go"
)

const (
	ModelName    = "paddleocr-rec"
	CharDictPath = "ppocr_keys_v5.txt"
)

type ModelConfig struct {
	Height int
	Mean   [3]float64
	Std    [3]float64

	common.BaseModelConfig
}

// ---------------------------------------------------------------------------
// Model wraps the CTC recognition ONNX session
// ---------------------------------------------------------------------------

type Model struct {
	*common.OnnxModel

	charDict *CharsetDict
}

func NewModel() *Model {
	m := &Model{}
	m.OnnxModel = common.NewOnnxModel(m)
	return m
}

func (m *Model) GetName() string { return ModelName }

func (m *Model) GetDefaultConfig() common.ModelConfig {
	return &ModelConfig{
		Height: 48,
		Mean:   [3]float64{0.5, 0.5, 0.5},
		Std:    [3]float64{0.5, 0.5, 0.5},
		BaseModelConfig: common.BaseModelConfig{
			OnnxConfig: common.Config{
				ModelPath: "PP-OCRv5_server_rec.onnx",
			},
		},
	}
}

func (m *Model) Init(configSrc common.ConfigSource) error {
	err := m.OnnxModel.Init(configSrc)
	if err != nil {
		return err
	}

	modelPath := m.OnnxModel.GetConfig().GetOnnxConfig().GetModelPath()
	dictPath := filepath.Join(filepath.Dir(modelPath), CharDictPath)

	dict, err := NewCharsetDict(modelPath, dictPath)
	if err != nil {
		return errors.Wrap(err, "failed to setup charset dicet")
	}
	m.charDict = dict
	return nil
}

func (m *Model) RecognizeLineOnly() bool {
	// This model only operates on line-level recognitions.
	// We should only traverse child elements obtained for detection.
	return true
}

// Recognize recognizes text in the region defined by quad (already ordered tl→tr→br→bl).
func (m *Model) Recognize(img image.Image, quad [4][2]int) (*recognize.Result, error) {
	config, ok := m.GetDefaultConfig().(*ModelConfig)
	if !ok {
		return nil, common.ErrInvalidConfig
	}

	// Order quad points so src[0]=TL, src[1]=TR, src[2]=BR, src[3]=BL.
	ordered := utils.OrderPoints4(utils.FloatQuad(quad))

	// Compute target width from quad aspect ratio.
	topW := utils.PointDistance(ordered[0], ordered[1])
	bottomW := utils.PointDistance(ordered[3], ordered[2])
	leftH := utils.PointDistance(ordered[0], ordered[3])
	rightH := utils.PointDistance(ordered[1], ordered[2])
	srcW := math.Max(topW, bottomW)
	srcH := math.Max(leftH, rightH)
	if srcH < 1 {
		srcH = 1
	}

	var targetW int
	var warpedImg *image.RGBA
	if srcH > srcW {
		// Vertical text: warp to a portrait crop, then rotate 90° CCW so
		// characters that were stacked top-to-bottom now flow left-to-right.
		targetH := int(math.Round(float64(config.Height) * srcH / srcW))
		if targetH < 1 {
			targetH = 1
		}
		portrait := utils.PerspectiveWarp(img, ordered, config.Height, targetH)
		warpedImg = utils.Rotate90CCW(portrait)
		targetW = targetH // after rotation, width = former portrait height
	} else {
		targetW = int(math.Round(float64(config.Height) * srcW / srcH))
		if targetW < 1 {
			targetW = 1
		}
		warpedImg = utils.PerspectiveWarp(img, ordered, targetW, config.Height)
	}

	// Build NCHW float32 tensor.
	data := utils.ImageToNCHW(warpedImg, config.Height, targetW, config.Mean, config.Std)

	shape := ort.NewShape(1, 3, int64(config.Height), int64(targetW))
	inTensor, err := ort.NewTensor(shape, data)
	if err != nil {
		return nil, fmt.Errorf("rec input tensor: %w", err)
	}
	defer func() {
		_ = inTensor.Destroy()
	}()

	outputs := make([]ort.Value, 1)
	if err := m.GetSession().Run([]ort.Value{inTensor}, outputs); err != nil {
		return nil, fmt.Errorf("rec inference: %w", err)
	}
	outTensor, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		_ = outputs[0].Destroy()
		return nil, fmt.Errorf("unexpected rec output type")
	}
	defer func() {
		_ = outTensor.Destroy()
	}()

	logits := outTensor.GetData()
	outShape := outTensor.GetShape()
	// Expected shape: (1, T, numClasses) or (T, numClasses) depending on model.
	var T, numClasses int
	if len(outShape) == 3 {
		T = int(outShape[1])
		numClasses = int(outShape[2])
	} else if len(outShape) == 2 {
		T = int(outShape[0])
		numClasses = int(outShape[1])
	} else {
		return nil, fmt.Errorf("unexpected rec output shape: %v", outShape)
	}

	text, score := m.charDict.Decode(logits, T, numClasses)
	return &recognize.Result{Text: text, Score: score}, nil
}
