package utils

import (
	"sort"
	"strings"
)

// Quad is a four-point polygon in [TL, TR, BR, BL] order, each point [x, y].
type Quad [4][2]int

// AABB returns [minX, minY, maxX, maxY] for the quad.
func (q Quad) AABB() [4]int {
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

	firstAABB := children[0].Quad.AABB()
	vertical := (firstAABB[3]-firstAABB[1]) > (firstAABB[2]-firstAABB[0])

	sort.Slice(children, func(i, j int) bool {
		ai := children[i].Quad.AABB()
		aj := children[j].Quad.AABB()
		if vertical {
			// Vertical text: right-to-left columns (CJK convention), then top-to-bottom within column.
			cx1 := (ai[0] + ai[2]) / 2
			cx2 := (aj[0] + aj[2]) / 2
			if cx1 != cx2 {
				return cx1 > cx2
			}
			return (ai[1]+ai[3])/2 < (aj[1]+aj[3])/2
		}
		// Horizontal text: top-to-bottom, then left-to-right.
		cy1 := (ai[1] + ai[3]) / 2
		cy2 := (aj[1] + aj[3]) / 2
		if cy1 != cy2 {
			return cy1 < cy2
		}
		return (ai[0]+ai[2])/2 < (aj[0]+aj[2])/2
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
