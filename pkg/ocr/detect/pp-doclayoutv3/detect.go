package ppdoclayoutv3

import (
	"fmt"
	"image"
	"sort"

	"github.com/multippt/gopaddleocr/pkg/ocr/common"
	"github.com/multippt/gopaddleocr/pkg/ocr/utils"
	ort "github.com/yalue/onnxruntime_go"
)

const ModelName = "pp-doclayoutv3"

type ModelConfig struct {
	InputSize      int     // default 800
	ScoreThreshold float32 // default 0.3
	Mean [3]float32
	Std  [3]float32
	common.BaseModelConfig
}

// ---------------------------------------------------------------------------
// Model wraps the PP-DocLayoutV3 ONNX session
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
		InputSize:      800,
		ScoreThreshold: 0.3,
		Mean: [3]float32{0.485, 0.456, 0.406},
		Std:  [3]float32{0.229, 0.224, 0.225},
		BaseModelConfig: common.BaseModelConfig{
			OnnxConfig: common.Config{
				ModelPath: "PP-DocLayoutV3.onnx",
			},
		},
	}
}

// Detect runs PP-DocLayoutV3 and returns layout boxes sorted by reading order.
func (m *Model) Detect(img image.Image) ([]utils.Box, error) {
	config, ok := m.GetDefaultConfig().(*ModelConfig)
	if !ok {
		return nil, common.ErrInvalidConfig
	}

	bounds := img.Bounds()
	origW := bounds.Max.X - bounds.Min.X
	origH := bounds.Max.Y - bounds.Min.Y
	if origW <= 0 || origH <= 0 {
		return nil, nil
	}

	size := config.InputSize
	scaleH := float32(size) / float32(origH)
	scaleW := float32(size) / float32(origW)

	// Resize to size×size (non-aspect-preserving) and normalize with config mean/std.
	resized := utils.BilinearResize(img, size, size)

	imageData := make([]float32, 3*size*size)
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			pix := resized.RGBAAt(x, y)
			rv := float32(pix.R) / 255.0
			gv := float32(pix.G) / 255.0
			bv := float32(pix.B) / 255.0
			idx := y*size + x
			imageData[0*size*size+idx] = (rv - config.Mean[0]) / config.Std[0]
			imageData[1*size*size+idx] = (gv - config.Mean[1]) / config.Std[1]
			imageData[2*size*size+idx] = (bv - config.Mean[2]) / config.Std[2]
		}
	}

	// im_shape tensor: [1][2]float32 = [[800, 800]]
	imShapeTensor, err := ort.NewTensor(ort.NewShape(1, 2), []float32{float32(size), float32(size)})
	if err != nil {
		return nil, fmt.Errorf("ppdoclayoutv3 im_shape tensor: %w", err)
	}
	defer func() { _ = imShapeTensor.Destroy() }()

	// image tensor: [1][3][size][size]float32
	imageTensor, err := ort.NewTensor(ort.NewShape(1, 3, int64(size), int64(size)), imageData)
	if err != nil {
		return nil, fmt.Errorf("ppdoclayoutv3 image tensor: %w", err)
	}
	defer func() { _ = imageTensor.Destroy() }()

	// scale_factor tensor: [1][2]float32 = [[scale_h, scale_w]]
	scaleTensor, err := ort.NewTensor(ort.NewShape(1, 2), []float32{scaleH, scaleW})
	if err != nil {
		return nil, fmt.Errorf("ppdoclayoutv3 scale_factor tensor: %w", err)
	}
	defer func() { _ = scaleTensor.Destroy() }()

	outputs := make([]ort.Value, 3)
	if err := m.GetSession().Run([]ort.Value{imShapeTensor, imageTensor, scaleTensor}, outputs); err != nil {
		return nil, fmt.Errorf("ppdoclayoutv3 inference: %w", err)
	}
	// outputs[1] and outputs[2] are unused auxiliary outputs.
	for _, v := range outputs[1:] {
		if v != nil {
			_ = v.Destroy()
		}
	}
	outTensor, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		_ = outputs[0].Destroy()
		return nil, fmt.Errorf("ppdoclayoutv3: unexpected output type")
	}
	defer func() { _ = outTensor.Destroy() }()

	raw := outTensor.GetData()
	outShape := outTensor.GetShape()

	// Output shape: [N, 7] — each row: [label_id, score, xmin, ymin, xmax, ymax, read_order]
	if len(outShape) < 2 || outShape[1] != 7 {
		return nil, fmt.Errorf("ppdoclayoutv3: unexpected output shape %v", outShape)
	}
	N := int(outShape[0])

	return m.postprocess(config, raw, N, origW, origH), nil
}

// postprocess converts raw ONNX output (N×7 flat float32) into sorted, clamped Box values.
// Each row is [label_id, score, xmin, ymin, xmax, ymax, read_order].
func (m *Model) postprocess(
	config *ModelConfig, raw []float32, N, origW, origH int,
) []utils.Box {
	type rawBox struct {
		classID                int
		score                  float64
		xmin, ymin, xmax, ymax float64
		order                  int
	}
	var candidates []rawBox
	for i := 0; i < N; i++ {
		row := raw[i*7 : i*7+7]
		score := float64(row[1])
		if float32(score) < config.ScoreThreshold {
			continue
		}
		candidates = append(candidates, rawBox{
			classID: int(row[0]),
			score:   score,
			xmin:    float64(row[2]),
			ymin:    float64(row[3]),
			xmax:    float64(row[4]),
			ymax:    float64(row[5]),
			order:   int(row[6]),
		})
	}

	// Sort by reading order.
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].order < candidates[j].order
	})

	clampW := origW - 1
	clampH := origH - 1
	if clampW < 0 {
		clampW = 0
	}
	if clampH < 0 {
		clampH = 0
	}

	boxes := make([]utils.Box, len(candidates))
	for i, c := range candidates {
		// Axis-aligned box → [TL, TR, BR, BL] quad.
		xMin := utils.ClampInt(int(c.xmin), 0, clampW)
		yMin := utils.ClampInt(int(c.ymin), 0, clampH)
		xMax := utils.ClampInt(int(c.xmax), 0, clampW)
		yMax := utils.ClampInt(int(c.ymax), 0, clampH)
		quad := [4][2]int{
			{xMin, yMin}, // TL
			{xMax, yMin}, // TR
			{xMax, yMax}, // BR
			{xMin, yMax}, // BL
		}
		boxes[i] = utils.Box{
			Quad:    quad,
			Score:   c.score,
			ClassID: c.classID,
			Order:   c.order,
		}
	}
	return boxes
}
