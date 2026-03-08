package paddleocr

import (
	"image"
	"image/color"
	"testing"

	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"

	"github.com/multippt/gopaddleocr/pkg/ocr/utils"
)

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

func TestClassifyPreprocess_TensorSize(t *testing.T) {
	cfg := NewModel().GetDefaultConfig().(*ModelConfig)
	m := &Model{config: cfg}
	img := newTextImage(300, 60, "Hello World")
	// Valid axis-aligned quad covering the whole image.
	quad := [4][2]int{{0, 0}, {299, 0}, {299, 59}, {0, 59}}
	crop := utils.PerspectiveWarp(img, utils.FloatQuad(quad), m.config.Width, m.config.Height)
	if crop == nil {
		t.Fatal("PerspectiveWarp returned nil for valid quad")
	}
	data := utils.ImageToNCHW(crop, m.config.Height, m.config.Width, m.config.Mean, m.config.Std)
	want := 3 * m.config.Height * m.config.Width
	if len(data) != want {
		t.Errorf("tensor length=%d want %d (3×%d×%d)", len(data), want, m.config.Height, m.config.Width)
	}
}
