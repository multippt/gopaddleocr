package paddleocr

import (
	"image"
	"image/color"
	"math"
	"testing"

	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"

	"github.com/multippt/gopaddleocr/pkg/ocr/utils"
)

// newRecTextImage creates an RGBA image with white background and black ASCII text.
func newRecTextImage(w, h int, text string) *image.RGBA {
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

func TestRecGetName(t *testing.T) {
	m := NewModel()
	if got := m.GetName(); got != ModelName {
		t.Errorf("GetName()=%q want %q", got, ModelName)
	}
}

func TestRecGetDefaultConfig(t *testing.T) {
	m := NewModel()
	cfg, ok := m.GetDefaultConfig().(*ModelConfig)
	if !ok {
		t.Fatal("GetDefaultConfig did not return *ModelConfig")
	}
	if cfg.Height != 48 {
		t.Errorf("Height=%d want 48", cfg.Height)
	}
	if cfg.DictPath == "" {
		t.Error("DictPath is empty")
	}
	if cfg.OnnxConfig.ModelPath == "" {
		t.Error("ModelPath is empty")
	}
}

func TestRecognizeLineOnly(t *testing.T) {
	m := NewModel()
	if !m.RecognizeLineOnly() {
		t.Error("RecognizeLineOnly() returned false, want true")
	}
}

func TestRecClose_NoInit(t *testing.T) {
	m := NewModel()
	if err := m.Close(); err != nil {
		t.Errorf("Close() on uninitialised model returned error: %v", err)
	}
}

func TestRecognizePreprocess_Horizontal(t *testing.T) {
	cfg := NewModel().GetDefaultConfig().(*ModelConfig)

	// Wide source image (landscape) → horizontal preprocessing path.
	srcW, srcH := 200, 30
	img := newRecTextImage(srcW, srcH, "Hello World")
	quad := [4][2]int{{0, 0}, {srcW - 1, 0}, {srcW - 1, srcH - 1}, {0, srcH - 1}}

	ordered := utils.OrderPoints4(utils.FloatQuad(quad))
	topW := utils.PointDistance(ordered[0], ordered[1])
	leftH := utils.PointDistance(ordered[0], ordered[3])

	// srcH < srcW → horizontal path.
	if leftH >= topW {
		t.Skip("test image is not landscape; skipping horizontal path test")
	}

	targetW := int(math.Round(float64(cfg.Height) * topW / leftH))
	if targetW < 1 {
		targetW = 1
	}

	warp := utils.PerspectiveWarp(img, ordered, targetW, cfg.Height)
	if warp == nil {
		t.Fatal("PerspectiveWarp returned nil")
	}
	if warp.Bounds().Dx() != targetW {
		t.Errorf("warp width=%d want %d", warp.Bounds().Dx(), targetW)
	}
	if warp.Bounds().Dy() != cfg.Height {
		t.Errorf("warp height=%d want %d", warp.Bounds().Dy(), cfg.Height)
	}
}

func TestRecognizePreprocess_Vertical(t *testing.T) {
	cfg := NewModel().GetDefaultConfig().(*ModelConfig)

	// Tall source image (portrait) → vertical preprocessing path.
	srcW, srcH := 30, 200
	img := newRecTextImage(srcW, srcH, "Hi")
	quad := [4][2]int{{0, 0}, {srcW - 1, 0}, {srcW - 1, srcH - 1}, {0, srcH - 1}}

	ordered := utils.OrderPoints4(utils.FloatQuad(quad))
	topW := utils.PointDistance(ordered[0], ordered[1])
	leftH := utils.PointDistance(ordered[0], ordered[3])

	// srcH > srcW → vertical path.
	if leftH <= topW {
		t.Skip("test image is not portrait; skipping vertical path test")
	}

	targetH := int(math.Round(float64(cfg.Height) * leftH / topW))
	if targetH < 1 {
		targetH = 1
	}

	portrait := utils.PerspectiveWarp(img, ordered, cfg.Height, targetH)
	if portrait == nil {
		t.Fatal("PerspectiveWarp returned nil for portrait")
	}
	rotated := utils.Rotate90CCW(portrait)

	// After 90° CCW rotation: width = former height, height = former width.
	if rotated.Bounds().Dx() != targetH {
		t.Errorf("rotated width=%d want %d (former portrait height)", rotated.Bounds().Dx(), targetH)
	}
	if rotated.Bounds().Dy() != cfg.Height {
		t.Errorf("rotated height=%d want %d (former portrait width)", rotated.Bounds().Dy(), cfg.Height)
	}
}
