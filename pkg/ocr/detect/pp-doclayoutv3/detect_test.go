package ppdoclayoutv3

import (
	"image"
	"image/color"
	"math"
	"testing"

	"github.com/multippt/gopaddleocr/pkg/ocr/common"
	"github.com/multippt/gopaddleocr/pkg/ocr/testutil"
)

type useDefault struct{}

func (useDefault) GetConfig(string) common.ModelConfig { return nil }

var testCfg = &ModelConfig{InputSize: 800, ScoreThreshold: 0.3}

func TestGetName(t *testing.T) {
	m := NewModel()
	if got := m.GetName(); got != ModelName {
		t.Errorf("GetName()=%q want %q", got, ModelName)
	}
}

func TestGetDefaultConfig(t *testing.T) {
	m := NewModel()
	cfg, ok := m.GetDefaultConfig().(*ModelConfig)
	if !ok {
		t.Fatal("GetDefaultConfig did not return *ModelConfig")
	}
	if cfg.InputSize != 800 {
		t.Errorf("InputSize=%d want 800", cfg.InputSize)
	}
	if cfg.ScoreThreshold != 0.3 {
		t.Errorf("ScoreThreshold=%f want 0.3", cfg.ScoreThreshold)
	}
}

func TestDetect_ZeroSizeImage(t *testing.T) {
	m := &Model{}
	img := image.NewRGBA(image.Rect(0, 0, 0, 0))
	boxes, err := m.Detect(img)
	if err != nil {
		t.Errorf("Detect with 0×0 image returned error: %v", err)
	}
	if boxes != nil {
		t.Errorf("expected nil boxes for 0×0 image, got %v", boxes)
	}
}

// makeRaw constructs a flat N×7 float32 slice.
// Each entry is [classID, score, xmin, ymin, xmax, ymax, order].
func makeRaw(rows [][7]float32) []float32 {
	out := make([]float32, len(rows)*7)
	for i, r := range rows {
		copy(out[i*7:], r[:])
	}
	return out
}

func TestPostprocess_ExactThresholdBoundary(t *testing.T) {
	m := &Model{}
	// Score == ScoreThreshold (0.3): condition is strict <, so it must NOT be filtered.
	raw := makeRaw([][7]float32{
		{0, 0.3, 10, 10, 50, 50, 0},
	})
	boxes := m.postprocess(testCfg, raw, 1, 100, 100)
	if len(boxes) != 1 {
		t.Errorf("expected 1 box (score == threshold must pass), got %d", len(boxes))
	}
}

func TestPostprocess_ClassIDAndOrderPreserved(t *testing.T) {
	m := &Model{}
	raw := makeRaw([][7]float32{
		{7, 0.9, 10, 10, 50, 50, 42},
	})
	boxes := m.postprocess(testCfg, raw, 1, 100, 100)
	if len(boxes) != 1 {
		t.Fatalf("expected 1 box, got %d", len(boxes))
	}
	if boxes[0].ClassID != 7 {
		t.Errorf("ClassID=%d, want 7", boxes[0].ClassID)
	}
	if boxes[0].Order != 42 {
		t.Errorf("Order=%d, want 42", boxes[0].Order)
	}
}

func TestPostprocess_ScorePreserved(t *testing.T) {
	m := &Model{}
	raw := makeRaw([][7]float32{
		{0, 0.753, 10, 10, 50, 50, 0},
	})
	boxes := m.postprocess(testCfg, raw, 1, 100, 100)
	if len(boxes) != 1 {
		t.Fatalf("expected 1 box, got %d", len(boxes))
	}
	if math.Abs(boxes[0].Score-0.753) > 0.0001 {
		t.Errorf("Score=%f, want ≈0.753", boxes[0].Score)
	}
}

func TestPostprocess_QuadLayout(t *testing.T) {
	m := &Model{}
	// xmin=10, ymin=20, xmax=50, ymax=80
	raw := makeRaw([][7]float32{
		{0, 0.9, 10, 20, 50, 80, 0},
	})
	boxes := m.postprocess(testCfg, raw, 1, 200, 200)
	if len(boxes) != 1 {
		t.Fatalf("expected 1 box, got %d", len(boxes))
	}
	q := boxes[0].Quad
	// TL
	if q[0] != [2]int{10, 20} {
		t.Errorf("TL=%v, want [10 20]", q[0])
	}
	// TR
	if q[1] != [2]int{50, 20} {
		t.Errorf("TR=%v, want [50 20]", q[1])
	}
	// BR
	if q[2] != [2]int{50, 80} {
		t.Errorf("BR=%v, want [50 80]", q[2])
	}
	// BL
	if q[3] != [2]int{10, 80} {
		t.Errorf("BL=%v, want [10 80]", q[3])
	}
}

func TestPostprocess_Empty(t *testing.T) {
	m := &Model{}
	boxes := m.postprocess(testCfg, nil, 0, 100, 100)
	if len(boxes) != 0 {
		t.Errorf("expected 0 boxes for N=0, got %d", len(boxes))
	}
}

func TestPostprocess_ScoreFilter(t *testing.T) {
	m := &Model{}
	// One row with score below threshold (0.2 < 0.3).
	raw := makeRaw([][7]float32{
		{1, 0.2, 10, 10, 50, 50, 0},
	})
	boxes := m.postprocess(testCfg, raw, 1, 100, 100)
	if len(boxes) != 0 {
		t.Errorf("expected 0 boxes after score filter, got %d", len(boxes))
	}
}

func TestPostprocess_ReadingOrderSort(t *testing.T) {
	m := &Model{}
	// Two boxes: first with order=5, second with order=2 → should be sorted [2, 5].
	raw := makeRaw([][7]float32{
		{1, 0.8, 10, 10, 50, 50, 5},
		{2, 0.9, 60, 10, 90, 50, 2},
	})
	boxes := m.postprocess(testCfg, raw, 2, 200, 200)
	if len(boxes) != 2 {
		t.Fatalf("expected 2 boxes, got %d", len(boxes))
	}
	if boxes[0].Order != 2 || boxes[1].Order != 5 {
		t.Errorf("expected orders [2,5], got [%d,%d]", boxes[0].Order, boxes[1].Order)
	}
}

func TestPostprocess_CoordClamping(t *testing.T) {
	m := &Model{}
	// Box with coords outside 100×100 image bounds → should be clamped.
	raw := makeRaw([][7]float32{
		{0, 0.9, -20, -10, 150, 130, 0},
	})
	boxes := m.postprocess(testCfg, raw, 1, 100, 100)
	if len(boxes) != 1 {
		t.Fatalf("expected 1 box, got %d", len(boxes))
	}
	q := boxes[0].Quad
	// All coordinates must be in [0, 99].
	for i, corner := range q {
		if corner[0] < 0 || corner[0] > 99 {
			t.Errorf("corner %d x=%d out of [0,99]", i, corner[0])
		}
		if corner[1] < 0 || corner[1] > 99 {
			t.Errorf("corner %d y=%d out of [0,99]", i, corner[1])
		}
	}
}

// ---------------------------------------------------------------------------
// Integration tests — require ORT + model file
// ---------------------------------------------------------------------------

func TestDocLayout_PlainWhiteImage(t *testing.T) {
	testutil.RequireORT(t)
	m := NewModel()
	testutil.RequireModel(t, m.GetDefaultConfig().GetOnnxConfig().GetModelPath())
	if err := m.Init(useDefault{}); err != nil {
		t.Fatalf("Init: %v", err)
	}
	defer m.Close()

	img := image.NewRGBA(image.Rect(0, 0, 400, 300))
	for y := 0; y < 300; y++ {
		for x := 0; x < 400; x++ {
			img.SetRGBA(x, y, color.RGBA{R: 255, G: 255, B: 255, A: 255})
		}
	}
	boxes, err := m.Detect(img)
	if err != nil {
		t.Fatalf("Detect: %v", err)
	}
	t.Logf("DocLayout returned %d boxes", len(boxes))
}

func TestPostprocess_MultiBox(t *testing.T) {
	m := &Model{}
	// Three boxes, all above threshold.
	raw := makeRaw([][7]float32{
		{0, 0.9, 5, 5, 20, 20, 0},
		{1, 0.7, 30, 5, 60, 20, 1},
		{2, 0.5, 70, 5, 90, 20, 2},
	})
	boxes := m.postprocess(testCfg, raw, 3, 200, 200)
	if len(boxes) != 3 {
		t.Errorf("expected 3 boxes, got %d", len(boxes))
	}
}
