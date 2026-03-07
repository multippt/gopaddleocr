package utils

import (
	"strings"
	"testing"
)

func TestAABB_Methods(t *testing.T) {
	a := AABB{10, 20, 50, 80}
	if got := a.Width(); got != 40 {
		t.Errorf("Width() = %d, want 40", got)
	}
	if got := a.Height(); got != 60 {
		t.Errorf("Height() = %d, want 60", got)
	}
	if got := a.Area(); got != 2400 {
		t.Errorf("Area() = %f, want 2400", got)
	}
	if got := a.CenterX(); got != 30 {
		t.Errorf("CenterX() = %d, want 30", got)
	}
	if got := a.CenterY(); got != 50 {
		t.Errorf("CenterY() = %d, want 50", got)
	}
}

func TestQuad_AABB(t *testing.T) {
	// Irregular quad: TL(5,10), TR(50,8), BR(52,40), BL(3,42)
	q := Quad{{5, 10}, {50, 8}, {52, 40}, {3, 42}}
	a := q.AABB()
	if a[0] != 3 || a[1] != 8 || a[2] != 52 || a[3] != 42 {
		t.Errorf("AABB() = %v, want [3 8 52 42]", a)
	}
}

func TestQuad_IsVertical(t *testing.T) {
	wide := Quad{{0, 0}, {100, 0}, {100, 30}, {0, 30}} // width=100, height=30
	if wide.IsVertical() {
		t.Error("wide quad should not be vertical")
	}
	tall := Quad{{0, 0}, {30, 0}, {30, 100}, {0, 100}} // width=30, height=100
	if !tall.IsVertical() {
		t.Error("tall quad should be vertical")
	}
}

func makeLeafBox(x, y, w, h int, text string) Box {
	return Box{
		Quad: Quad{
			{x, y},
			{x + w, y},
			{x + w, y + h},
			{x, y + h},
		},
		Text: text,
	}
}

func TestAABB_BoundedArea(t *testing.T) {
	if got := (AABB{10, 20, 50, 80}).BoundedArea(); got != 2400 {
		t.Errorf("BoundedArea() = %f, want 2400", got)
	}
	if got := (AABB{10, 10, 10, 50}).BoundedArea(); got != 0 {
		t.Errorf("BoundedArea() degenerate = %f, want 0", got)
	}
}

func TestAABB_IntersectionArea(t *testing.T) {
	a := AABB{0, 0, 10, 10}
	b := AABB{5, 5, 15, 15}
	if got := a.IntersectionArea(b); got != 25 {
		t.Errorf("IntersectionArea() = %f, want 25", got)
	}
	c := AABB{20, 20, 30, 30}
	if got := a.IntersectionArea(c); got != 0 {
		t.Errorf("IntersectionArea() non-overlapping = %f, want 0", got)
	}
}

func TestAABB_Gap(t *testing.T) {
	a := AABB{0, 0, 10, 10}
	// Touching on the right: gap should be 0.
	b := AABB{10, 0, 20, 10}
	if got := a.Gap(b); got != 0 {
		t.Errorf("Gap() touching = %f, want 0", got)
	}
	// Separated horizontally by 5.
	c := AABB{15, 0, 25, 10}
	if got := a.Gap(c); got != 5 {
		t.Errorf("Gap() horizontal = %f, want 5", got)
	}
	// Overlapping: gap should be 0.
	d := AABB{5, 5, 15, 15}
	if got := a.Gap(d); got != 0 {
		t.Errorf("Gap() overlapping = %f, want 0", got)
	}
}

func TestBox_GetText_Leaf(t *testing.T) {
	b := makeLeafBox(0, 0, 100, 20, "hello")
	if got := b.GetText(); got != "hello" {
		t.Errorf("GetText() = %q, want %q", got, "hello")
	}
}

func TestBox_GetText_Horizontal(t *testing.T) {
	// Three horizontal boxes: B is top-left, A is top-right, C is bottom-left.
	// Expected reading order: B, A, C (top-to-bottom, then left-to-right within row).
	b := Box{
		Children: []Box{
			makeLeafBox(50, 0, 40, 15, "A"),  // top, right
			makeLeafBox(0, 0, 40, 15, "B"),   // top, left
			makeLeafBox(0, 50, 40, 15, "C"),  // bottom, left
		},
	}
	got := b.GetText()
	parts := strings.Split(got, " ")
	want := []string{"B", "A", "C"}
	for i, w := range want {
		if i >= len(parts) || parts[i] != w {
			t.Errorf("GetText() = %q, want tokens %v", got, want)
			break
		}
	}
}

func TestBox_GetText_Vertical(t *testing.T) {
	// Three vertical boxes (taller than wide).
	// Right column comes first (CJK convention), then within column top-to-bottom.
	// col2 (x=100): top, col1 (x=50): top, col1 (x=50): bottom
	b := Box{
		Children: []Box{
			makeLeafBox(50, 50, 20, 40, "B"),  // left col, bottom
			makeLeafBox(50, 0, 20, 40, "A"),   // left col, top
			makeLeafBox(100, 0, 20, 40, "C"),  // right col, top
		},
	}
	got := b.GetText()
	parts := strings.Split(got, " ")
	want := []string{"C", "A", "B"}
	for i, w := range want {
		if i >= len(parts) || parts[i] != w {
			t.Errorf("GetText() = %q, want tokens %v", got, want)
			break
		}
	}
}

func TestBox_GetTextSize_Leaf(t *testing.T) {
	// 100x30: min dimension is 30
	b := makeLeafBox(0, 0, 100, 30, "x")
	b.Text = "x"
	if got := b.GetTextSize(); got != 30 {
		t.Errorf("GetTextSize() = %f, want 30", got)
	}
	// 30x100: min dimension is 30
	b2 := makeLeafBox(0, 0, 30, 100, "x")
	b2.Text = "x"
	if got := b2.GetTextSize(); got != 30 {
		t.Errorf("GetTextSize() = %f, want 30", got)
	}
}

func TestBox_GetTextSize_Parent(t *testing.T) {
	// Two children: sizes 20 and 40 → average 30.
	b := Box{
		Children: []Box{
			makeLeafBox(0, 0, 100, 20, "A"), // min dim = 20
			makeLeafBox(0, 30, 100, 40, "B"), // min dim = 40
		},
	}
	if got := b.GetTextSize(); got != 30 {
		t.Errorf("GetTextSize() = %f, want 30", got)
	}
}
