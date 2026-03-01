package recognize

import (
	"bufio"
	"fmt"
	"image"
	"math"
	"os"

	"github.com/multippt/gopaddleocr/pkg/ocr/onnx"
	"github.com/multippt/gopaddleocr/pkg/ocr/utils"
	ort "github.com/yalue/onnxruntime_go"
)

type ModelConfig struct {
	Height int
	Mean   [3]float64
	Std    [3]float64

	OnnxConfig onnx.Config
}

var DefaultConfig = &ModelConfig{
	Height: 48,
	Mean:   [3]float64{0.5, 0.5, 0.5},
	Std:    [3]float64{0.5, 0.5, 0.5},

	OnnxConfig: onnx.Config{
		InputName:  "x",
		OutputName: "fetch_name_0",
	},
}

// ---------------------------------------------------------------------------
// Model wraps the CTC recognition ONNX session
// ---------------------------------------------------------------------------

type Model struct {
	session  *ort.DynamicAdvancedSession
	charDict []string
	config   *ModelConfig
}

func NewModel(modelPath, dictPath string) (*Model, error) {
	if err := EnsureCharDict(modelPath, dictPath); err != nil {
		return nil, fmt.Errorf("char dict: %w", err)
	}
	dict, err := loadCharDict(dictPath)
	if err != nil {
		return nil, fmt.Errorf("load char dict: %w", err)
	}

	m := &Model{
		config:   DefaultConfig,
		charDict: dict,
	}
	session, err := ort.NewDynamicAdvancedSession(modelPath,
		[]string{m.config.OnnxConfig.InputName},
		[]string{m.config.OnnxConfig.OutputName},
		m.config.OnnxConfig.Options)
	if err != nil {
		return nil, err
	}
	m.session = session
	return m, nil
}

// Run recognizes text in the region defined by quad (already ordered tl→tr→br→bl).
func (m *Model) Run(img image.Image, quad [4][2]int) (string, float64, error) {
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
			return "", 0, nil
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
			return "", 0, nil
		}
	}

	// Build NCHW float32 tensor.
	data := utils.ImageToNCHW(warpedImg, m.config.Height, targetW, m.config.Mean, m.config.Std)

	shape := ort.NewShape(1, 3, int64(m.config.Height), int64(targetW))
	inTensor, err := ort.NewTensor(shape, data)
	if err != nil {
		return "", 0, fmt.Errorf("rec input tensor: %w", err)
	}
	defer func() {
		_ = inTensor.Destroy()
	}()

	outputs := make([]ort.Value, 1)
	if err := m.session.Run([]ort.Value{inTensor}, outputs); err != nil {
		return "", 0, fmt.Errorf("rec inference: %w", err)
	}
	outTensor, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		_ = outputs[0].Destroy()
		return "", 0, fmt.Errorf("unexpected rec output type")
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
		return "", 0, fmt.Errorf("unexpected rec output shape: %v", outShape)
	}

	text, score := ctcDecode(logits, T, numClasses, m.charDict)
	return text, score, nil
}

// ---------------------------------------------------------------------------
// Character dictionary
// ---------------------------------------------------------------------------

// loadCharDict reads ppocr_keys_v1.txt (one char per line).
// Index 0 is the CTC blank token; indices 1..N map to the file lines.
// A trailing space character (standard PaddleOCR convention) is appended.
func loadCharDict(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() {
		_ = f.Close()
	}()

	dict := []string{""} // index 0 = blank
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		dict = append(dict, sc.Text())
	}
	dict = append(dict, " ") // trailing space
	return dict, sc.Err()
}

// ---------------------------------------------------------------------------
// CTC decoding
// ---------------------------------------------------------------------------

// ctcDecode runs argmax-CTC on logits shaped (T × numClasses) flattened.
// Returns the decoded string and the geometric-mean confidence of emitted chars.
func ctcDecode(logits []float32, T, numClasses int, charDict []string) (string, float64) {
	if T == 0 || numClasses == 0 {
		return "", 0
	}
	if len(logits) < T*numClasses {
		return "", 0
	}

	type step struct {
		class int
		prob  float32
	}
	steps := make([]step, T)
	for t := 0; t < T; t++ {
		best := 0
		bestP := logits[t*numClasses]
		for c := 1; c < numClasses; c++ {
			if logits[t*numClasses+c] > bestP {
				bestP = logits[t*numClasses+c]
				best = c
			}
		}
		steps[t] = step{best, bestP}
	}

	var runes []rune
	var probs []float64
	prev := -1
	for _, s := range steps {
		if s.class == 0 {
			prev = 0
			continue
		}
		if s.class == prev {
			continue
		}
		if s.class < len(charDict) {
			for _, r := range charDict[s.class] {
				runes = append(runes, r)
			}
			probs = append(probs, float64(s.prob))
		}
		prev = s.class
	}

	if len(runes) == 0 {
		return "", 0
	}

	// Geometric mean confidence.
	logSum := 0.0
	for _, p := range probs {
		if p > 0 {
			logSum += math.Log(p)
		}
	}
	score := math.Exp(logSum / float64(len(probs)))

	return string(runes), score
}
