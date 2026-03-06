package paddleocr

import (
	"fmt"
	"image"
	"math"

	"github.com/multippt/gopaddleocr/pkg/ocr/onnx"
	"github.com/multippt/gopaddleocr/pkg/ocr/recognize"
	"github.com/multippt/gopaddleocr/pkg/ocr/utils"
	ort "github.com/yalue/onnxruntime_go"
)

type ModelConfig struct {
	ModelPath string
	DictPath  string

	Height int
	Mean   [3]float64
	Std    [3]float64

	OnnxConfig onnx.Config
}

// ---------------------------------------------------------------------------
// Model wraps the CTC recognition ONNX session
// ---------------------------------------------------------------------------

type Model struct {
	session  *ort.DynamicAdvancedSession
	charDict *CharsetDict
	config   *ModelConfig
}

func NewModel(cfg *ModelConfig) (*Model, error) {
	if cfg == nil {
		return nil, fmt.Errorf("recognize: config is required")
	}
	dict, err := NewCharsetDict(cfg.ModelPath, cfg.DictPath)
	if err != nil {
		return nil, fmt.Errorf("char dict: %w", err)
	}
	m := &Model{config: cfg, charDict: dict}
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

// Recognize recognizes text in the region defined by quad (already ordered tl→tr→br→bl).
func (m *Model) Recognize(img image.Image, quad [4][2]int) (recognize.Result, error) {
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
		targetH := int(math.Round(float64(m.config.Height) * srcH / srcW))
		if targetH < 1 {
			targetH = 1
		}
		portrait := utils.PerspectiveWarp(img, ordered, m.config.Height, targetH)
		if portrait == nil {
			return recognize.Result{}, nil
		}
		warpedImg = utils.Rotate90CCW(portrait)
		targetW = targetH // after rotation, width = former portrait height
	} else {
		targetW = int(math.Round(float64(m.config.Height) * srcW / srcH))
		if targetW < 1 {
			targetW = 1
		}
		warpedImg = utils.PerspectiveWarp(img, ordered, targetW, m.config.Height)
		if warpedImg == nil {
			return recognize.Result{}, nil
		}
	}

	// Build NCHW float32 tensor.
	data := utils.ImageToNCHW(warpedImg, m.config.Height, targetW, m.config.Mean, m.config.Std)

	shape := ort.NewShape(1, 3, int64(m.config.Height), int64(targetW))
	inTensor, err := ort.NewTensor(shape, data)
	if err != nil {
		return recognize.Result{}, fmt.Errorf("rec input tensor: %w", err)
	}
	defer func() {
		_ = inTensor.Destroy()
	}()

	outputs := make([]ort.Value, 1)
	if err := m.session.Run([]ort.Value{inTensor}, outputs); err != nil {
		return recognize.Result{}, fmt.Errorf("rec inference: %w", err)
	}
	outTensor, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		_ = outputs[0].Destroy()
		return recognize.Result{}, fmt.Errorf("unexpected rec output type")
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
		return recognize.Result{}, fmt.Errorf("unexpected rec output shape: %v", outShape)
	}

	text, score := m.charDict.Decode(logits, T, numClasses)
	return recognize.Result{Text: text, Score: score}, nil
}
