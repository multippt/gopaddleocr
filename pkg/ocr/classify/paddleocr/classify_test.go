package paddleocr

import (
	"image"
	"image/color"
	"testing"

	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"

	"github.com/multippt/gopaddleocr/pkg/ocr/common"
	"github.com/multippt/gopaddleocr/pkg/ocr/testutil"
	"github.com/multippt/gopaddleocr/pkg/ocr/utils"
)

// useDefault is a ConfigSource that returns nil, so OnnxModel.Init falls back
// to the model's own GetDefaultConfig().
type useDefault struct{}

func (useDefault) GetConfig(string) common.ModelConfig { return nil }

// newTextImage creates an RGBA image with white background and black ASCII text.
func newTextImage(w, h int, text string) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.SetRGBA(x, y, color.RGBA{R: 255, G: 255, B: 255, A: 255})
		}
	}
	d := &font.Drawer{
		Dst:  img,
		Src:  image.NewUniform(color.RGBA{R: 0, G: 0, B: 0, A: 255}),
		Face: basicfont.Face7x13,
		Dot:  fixed.P(5, h/2+4),
	}
	d.DrawString(text)
	return img
}

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
	if cfg.Height != 48 {
		t.Errorf("Height=%d want 48", cfg.Height)
	}
	if cfg.Width != 192 {
		t.Errorf("Width=%d want 192", cfg.Width)
	}
	if cfg.Threshold != 0.9 {
		t.Errorf("Threshold=%f want 0.9", cfg.Threshold)
	}
	if cfg.Mean != [3]float64{0.5, 0.5, 0.5} {
		t.Errorf("Mean=%v want [0.5,0.5,0.5]", cfg.Mean)
	}
	if cfg.Std != [3]float64{0.5, 0.5, 0.5} {
		t.Errorf("Std=%v want [0.5,0.5,0.5]", cfg.Std)
	}
}

func TestClose_NoInit(t *testing.T) {
	m := NewModel()
	if err := m.Close(); err != nil {
		t.Errorf("Close() on uninitialised model returned error: %v", err)
	}
}

func TestClassifyPreprocess_WhitePixelNormalization(t *testing.T) {
	cfg := NewModel().GetDefaultConfig().(*ModelConfig)
	img := image.NewRGBA(image.Rect(0, 0, cfg.Width, cfg.Height))
	for y := 0; y < cfg.Height; y++ {
		for x := 0; x < cfg.Width; x++ {
			img.SetRGBA(x, y, color.RGBA{R: 255, G: 255, B: 255, A: 255})
		}
	}
	data := utils.ImageToNCHW(img, cfg.Height, cfg.Width, cfg.Mean, cfg.Std)
	for i, v := range data {
		if v < 0.99 || v > 1.01 {
			t.Errorf("data[%d]=%f, want ≈1.0 (white pixel with mean/std=0.5)", i, v)
			break
		}
	}
}

func TestClassifyPreprocess_BlackPixelNormalization(t *testing.T) {
	cfg := NewModel().GetDefaultConfig().(*ModelConfig)
	img := image.NewRGBA(image.Rect(0, 0, cfg.Width, cfg.Height))
	for y := 0; y < cfg.Height; y++ {
		for x := 0; x < cfg.Width; x++ {
			img.SetRGBA(x, y, color.RGBA{R: 0, G: 0, B: 0, A: 255})
		}
	}
	data := utils.ImageToNCHW(img, cfg.Height, cfg.Width, cfg.Mean, cfg.Std)
	for i, v := range data {
		if v > -0.99 || v < -1.01 {
			t.Errorf("data[%d]=%f, want ≈-1.0 (black pixel with mean/std=0.5)", i, v)
			break
		}
	}
}

func TestClassifyPreprocess_PortraitCrop(t *testing.T) {
	cfg := NewModel().GetDefaultConfig().(*ModelConfig)
	// Portrait image 30×200.
	src := image.NewRGBA(image.Rect(0, 0, 30, 200))
	for y := 0; y < 200; y++ {
		for x := 0; x < 30; x++ {
			src.SetRGBA(x, y, color.RGBA{R: 128, G: 128, B: 128, A: 255})
		}
	}
	quad := utils.FloatQuad([4][2]int{{0, 0}, {29, 0}, {29, 199}, {0, 199}})
	crop := utils.PerspectiveWarp(src, quad, cfg.Width, cfg.Height)
	if crop == nil {
		t.Fatal("PerspectiveWarp returned nil for portrait crop")
	}
	b := crop.Bounds()
	if b.Dx() != cfg.Width || b.Dy() != cfg.Height {
		t.Errorf("crop size=%d×%d, want %d×%d", b.Dx(), b.Dy(), cfg.Width, cfg.Height)
	}
	data := utils.ImageToNCHW(crop, cfg.Height, cfg.Width, cfg.Mean, cfg.Std)
	want := 3 * cfg.Height * cfg.Width
	if len(data) != want {
		t.Errorf("ImageToNCHW length=%d, want %d", len(data), want)
	}
}

// ---------------------------------------------------------------------------
// Integration tests — require ORT + model file
// ---------------------------------------------------------------------------

func TestClassify_UprightWhiteImage(t *testing.T) {
	testutil.RequireORT(t)
	m := NewModel()
	testutil.RequireModel(t, m.GetDefaultConfig().GetOnnxConfig().GetModelPath())
	if err := m.Init(useDefault{}); err != nil {
		t.Fatalf("Init: %v", err)
	}
	defer m.Close()

	// A plain white 300×60 image — the model should not panic and should
	// return a definite answer (upright is expected for a featureless image).
	img := newTextImage(300, 60, "")
	quad := [4][2]int{{0, 0}, {299, 0}, {299, 59}, {0, 59}}
	rotated, err := m.Classify(img, quad)
	if err != nil {
		t.Fatalf("Classify: %v", err)
	}
	t.Logf("Classify result: rotated=%v", rotated)
}

func TestClassify_TextImage(t *testing.T) {
	testutil.RequireORT(t)
	m := NewModel()
	testutil.RequireModel(t, m.GetDefaultConfig().GetOnnxConfig().GetModelPath())
	if err := m.Init(useDefault{}); err != nil {
		t.Fatalf("Init: %v", err)
	}
	defer m.Close()

	img := newTextImage(300, 60, "Hello World")
	quad := [4][2]int{{0, 0}, {299, 0}, {299, 59}, {0, 59}}
	_, err := m.Classify(img, quad)
	if err != nil {
		t.Fatalf("Classify: %v", err)
	}
}

func TestClassifyPreprocess_TensorSize(t *testing.T) {
	cfg := NewModel().GetDefaultConfig().(*ModelConfig)
	img := newTextImage(300, 60, "Hello World")
	// Valid axis-aligned quad covering the whole image.
	quad := [4][2]int{{0, 0}, {299, 0}, {299, 59}, {0, 59}}
	crop := utils.PerspectiveWarp(img, utils.FloatQuad(quad), cfg.Width, cfg.Height)
	if crop == nil {
		t.Fatal("PerspectiveWarp returned nil for valid quad")
	}
	data := utils.ImageToNCHW(crop, cfg.Height, cfg.Width, cfg.Mean, cfg.Std)
	want := 3 * cfg.Height * cfg.Width
	if len(data) != want {
		t.Errorf("tensor length=%d want %d (3×%d×%d)", len(data), want, cfg.Height, cfg.Width)
	}
}
