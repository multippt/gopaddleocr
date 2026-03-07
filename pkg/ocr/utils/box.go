package utils

import (
	"math"
	"sort"
	"strings"
)

// ---------------------------------------------------------------------------
// AABB utilities
// ---------------------------------------------------------------------------

// AABB is an axis-aligned bounding box [minX, minY, maxX, maxY].
type AABB [4]int

func (a AABB) Width() int    { return a[2] - a[0] }
func (a AABB) Height() int   { return a[3] - a[1] }
func (a AABB) Area() float64 { return float64(a.Width()) * float64(a.Height()) }
func (a AABB) CenterX() int  { return (a[0] + a[2]) / 2 }
func (a AABB) CenterY() int  { return (a[1] + a[3]) / 2 }

// BoundedArea returns 0 for degenerate (zero/negative) boxes, otherwise Area.
func (a AABB) BoundedArea() float64 {
	if a.Width() <= 0 || a.Height() <= 0 {
		return 0
	}
	return a.Area()
}

// IntersectionArea returns the area of the overlap between a and b.
func (a AABB) IntersectionArea(b AABB) float64 {
	x1 := a[0]
	if b[0] > x1 {
		x1 = b[0]
	}
	y1 := a[1]
	if b[1] > y1 {
		y1 = b[1]
	}
	x2 := a[2]
	if b[2] < x2 {
		x2 = b[2]
	}
	y2 := a[3]
	if b[3] < y2 {
		y2 = b[3]
	}
	w := x2 - x1
	h := y2 - y1
	if w <= 0 || h <= 0 {
		return 0
	}
	return float64(w) * float64(h)
}

// Gap returns the minimum distance between a and b (0 if overlapping).
func (a AABB) Gap(b AABB) float64 {
	dx := 0.0
	if a[2] < b[0] {
		dx = float64(b[0] - a[2])
	} else if b[2] < a[0] {
		dx = float64(a[0] - b[2])
	}
	dy := 0.0
	if a[3] < b[1] {
		dy = float64(b[1] - a[3])
	} else if b[3] < a[1] {
		dy = float64(a[1] - b[3])
	}
	return math.Sqrt(dx*dx + dy*dy)
}

// ---------------------------------------------------------------------------
// Quad utilities
// ---------------------------------------------------------------------------

// Quad is a four-point polygon in [TL, TR, BR, BL] order, each point [x, y].
type Quad [4][2]int

// AABB returns the axis-aligned bounding box for the quad.
func (q Quad) AABB() AABB {
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
	return AABB{minX, minY, maxX, maxY}
}

// IsVertical returns true when the quad is taller than it is wide.
func (q Quad) IsVertical() bool {
	a := q.AABB()
	return a.Height() > a.Width()
}

// ---------------------------------------------------------------------------
// Box utilities
// ---------------------------------------------------------------------------

// Box is the result of a detection stage.
// ClassID is -1 for generic text detection (PaddleOCR), or a layout class index (DocLayoutV3).
// Order is -1 if reading order is not provided.
// Children, when non-empty, represent sub-boxes (e.g. individual text lines within a merged region).
// Text is set by the recognition pipeline.
type Box struct {
	Quad     Quad
	Score    float64
	ClassID  int
	Order    int
	Children []Box
	Text     string
}

func (b Box) GetText() string {
	if b.Text != "" || len(b.Children) == 0 {
		return b.Text
	}

	// Sort children by reading order based on child orientation.
	// (Parent AABB is unreliable: a row of vertical columns is wider than tall.)
	children := make([]Box, len(b.Children))
	copy(children, b.Children)

	vertical := children[0].Quad.IsVertical()

	sort.Slice(children, func(i, j int) bool {
		ai := children[i].Quad.AABB()
		aj := children[j].Quad.AABB()
		if vertical {
			// Vertical text: right-to-left columns (CJK convention), then top-to-bottom within column.
			if ai.CenterX() != aj.CenterX() {
				return ai.CenterX() > aj.CenterX()
			}
			return ai.CenterY() < aj.CenterY()
		}
		// Horizontal text: top-to-bottom, then left-to-right.
		if ai.CenterY() != aj.CenterY() {
			return ai.CenterY() < aj.CenterY()
		}
		return ai.CenterX() < aj.CenterX()
	})

	parts := make([]string, 0, len(children))
	for _, c := range children {
		if t := c.GetText(); t != "" {
			parts = append(parts, t)
		}
	}
	return strings.Join(parts, " ")
}

func (b Box) GetTextSize() float64 {
	if len(b.Children) == 0 {
		aabb := b.Quad.AABB()
		w := float64(aabb.Width())
		h := float64(aabb.Height())
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
