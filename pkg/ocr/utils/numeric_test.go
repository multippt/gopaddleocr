package utils

import "testing"

// ---------------------------------------------------------------------------
// ClampInt
// ---------------------------------------------------------------------------

func TestClampInt(t *testing.T) {
	tests := []struct {
		v, lo, hi, want int
	}{
		{-5, 0, 10, 0},   // below lo
		{0, 0, 10, 0},    // at lo
		{5, 0, 10, 5},    // in range
		{10, 0, 10, 10},  // at hi
		{15, 0, 10, 10},  // above hi
		{0, 0, 0, 0},     // degenerate lo==hi
	}
	for _, tc := range tests {
		if got := ClampInt(tc.v, tc.lo, tc.hi); got != tc.want {
			t.Errorf("ClampInt(%d, %d, %d) = %d, want %d", tc.v, tc.lo, tc.hi, got, tc.want)
		}
	}
}

// ---------------------------------------------------------------------------
// MinInt4
// ---------------------------------------------------------------------------

func TestMinInt4(t *testing.T) {
	tests := []struct {
		a    [4]int
		want int
	}{
		{[4]int{3, 1, 4, 2}, 1},    // min in middle
		{[4]int{1, 2, 3, 4}, 1},    // min at front
		{[4]int{4, 3, 2, 1}, 1},    // min at back
		{[4]int{7, 7, 7, 7}, 7},    // all equal
		{[4]int{-3, -1, 0, 2}, -3}, // negative values
	}
	for _, tc := range tests {
		if got := MinInt4(tc.a); got != tc.want {
			t.Errorf("MinInt4(%v) = %d, want %d", tc.a, got, tc.want)
		}
	}
}

// ---------------------------------------------------------------------------
// MaxInt4
// ---------------------------------------------------------------------------

func TestMaxInt4(t *testing.T) {
	tests := []struct {
		a    [4]int
		want int
	}{
		{[4]int{3, 1, 4, 2}, 4},   // max in middle
		{[4]int{4, 3, 2, 1}, 4},   // max at front
		{[4]int{1, 2, 3, 4}, 4},   // max at back
		{[4]int{5, 5, 5, 5}, 5},   // all equal
		{[4]int{-3, -1, 0, 2}, 2}, // negative values
	}
	for _, tc := range tests {
		if got := MaxInt4(tc.a); got != tc.want {
			t.Errorf("MaxInt4(%v) = %d, want %d", tc.a, got, tc.want)
		}
	}
}
