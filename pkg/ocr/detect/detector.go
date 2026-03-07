package detect

import (
	"image"
	"strings"
)

// Box is the result of a detection stage.
// ClassID is -1 for generic text detection (PaddleOCR), or a layout class index (DocLayoutV3).
// Order is -1 if reading order is not provided.
// Children, when non-empty, represent sub-boxes (e.g. individual text lines within a merged region).
// Text is set by the recognition pipeline.
type Box struct {
	Quad     [4][2]int
	Score    float64
	ClassID  int
	Order    int
	Children []Box
	Text     string
}

type Detector interface {
	Detect(img image.Image) ([]Box, error)
	Close() error
}

// BoxAABB returns [minX, minY, maxX, maxY] for a quad.
func BoxAABB(q [4][2]int) [4]int {
	minX, minY := q[0][0], q[0][1]
	maxX, maxY := q[0][0], q[0][1]
	for _, p := range q[1:] {
		if p[0] < minX {
			minX = p[0]
		}
		if p[0] > maxX {
			maxX = p[0]
		}
		if p[1] < minY {
			minY = p[1]
		}
		if p[1] > maxY {
			maxY = p[1]
		}
	}
	return [4]int{minX, minY, maxX, maxY}
}

func (b Box) GetText() string {
	if b.Text != "" || len(b.Children) == 0 {
		return b.Text
	}
	parts := make([]string, 0, len(b.Children))
	for _, c := range b.Children {
		if t := c.GetText(); t != "" {
			parts = append(parts, t)
		}
	}
	return strings.Join(parts, " ")
}

func (b Box) GetTextSize() float64 {
	if len(b.Children) == 0 {
		aabb := BoxAABB(b.Quad)
		w := float64(aabb[2] - aabb[0])
		h := float64(aabb[3] - aabb[1])
		if h > w {
			return w
		}
		return h
	}
	var sum float64
	for _, c := range b.Children {
		sum += c.GetTextSize()
	}
	return sum / float64(len(b.Children))
}

// StubDetector returns a single box covering the entire image.
type StubDetector struct{}

func (StubDetector) Detect(img image.Image) ([]Box, error) {
	b := img.Bounds()
	quad := [4][2]int{
		{b.Min.X, b.Min.Y},
		{b.Max.X, b.Min.Y},
		{b.Max.X, b.Max.Y},
		{b.Min.X, b.Max.Y},
	}
	return []Box{{Quad: quad, Score: 1, ClassID: -1, Order: -1}}, nil
}

func (StubDetector) Close() error { return nil }
