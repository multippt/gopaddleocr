package ppdoclayoutv3

import (
	"image"
	"testing"
)

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
	m := &Model{config: testCfg}
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

func TestPostprocess_Empty(t *testing.T) {
	m := &Model{config: testCfg}
	boxes := m.postprocess(nil, 0, 100, 100)
	if len(boxes) != 0 {
		t.Errorf("expected 0 boxes for N=0, got %d", len(boxes))
	}
}

func TestPostprocess_ScoreFilter(t *testing.T) {
	m := &Model{config: testCfg}
	// One row with score below threshold (0.2 < 0.3).
	raw := makeRaw([][7]float32{
		{1, 0.2, 10, 10, 50, 50, 0},
	})
	boxes := m.postprocess(raw, 1, 100, 100)
	if len(boxes) != 0 {
		t.Errorf("expected 0 boxes after score filter, got %d", len(boxes))
	}
}

func TestPostprocess_ReadingOrderSort(t *testing.T) {
	m := &Model{config: testCfg}
	// Two boxes: first with order=5, second with order=2 → should be sorted [2, 5].
	raw := makeRaw([][7]float32{
		{1, 0.8, 10, 10, 50, 50, 5},
		{2, 0.9, 60, 10, 90, 50, 2},
	})
	boxes := m.postprocess(raw, 2, 200, 200)
	if len(boxes) != 2 {
		t.Fatalf("expected 2 boxes, got %d", len(boxes))
	}
	if boxes[0].Order != 2 || boxes[1].Order != 5 {
		t.Errorf("expected orders [2,5], got [%d,%d]", boxes[0].Order, boxes[1].Order)
	}
}

func TestPostprocess_CoordClamping(t *testing.T) {
	m := &Model{config: testCfg}
	// Box with coords outside 100×100 image bounds → should be clamped.
	raw := makeRaw([][7]float32{
		{0, 0.9, -20, -10, 150, 130, 0},
	})
	boxes := m.postprocess(raw, 1, 100, 100)
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

func TestPostprocess_MultiBox(t *testing.T) {
	m := &Model{config: testCfg}
	// Three boxes, all above threshold.
	raw := makeRaw([][7]float32{
		{0, 0.9, 5, 5, 20, 20, 0},
		{1, 0.7, 30, 5, 60, 20, 1},
		{2, 0.5, 70, 5, 90, 20, 2},
	})
	boxes := m.postprocess(raw, 3, 200, 200)
	if len(boxes) != 3 {
		t.Errorf("expected 3 boxes, got %d", len(boxes))
	}
}
