package paddleocr

import (
	"bufio"
	"bytes"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// testModelPath returns the path to a model file inside the "models"
// subdirectory of the package source tree.  It resolves the package root by
// walking up from os.Getwd() until it finds a go.mod, then descending into
// the known package path.  This works regardless of the CWD set by the IDE.
func testModelPath(name string) string {
	dir, err := os.Getwd()
	if err != nil {
		return filepath.Join("models", name)
	}
	// Walk upward to the module root (directory containing go.mod).
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			break
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			// Reached fs root without finding go.mod; fall back to relative.
			return filepath.Join("models", name)
		}
		dir = parent
	}
	return filepath.Join(dir, "models", name)
}

func TestONNXMetadataExtract(t *testing.T) {
	model := testModelPath("ch_PP-OCRv5_rec_server_infer.onnx")
	if _, err := os.Stat(model); err != nil {
		t.Skipf("skipping: %s not found", model)
	}

	got, err := onnxMetadataLookup(model, "character")
	if err != nil {
		t.Fatalf("onnxMetadataLookup: %v", err)
	}

	lines := splitLines(got)
	t.Logf("extracted %d entries; first=%q last=%q", len(lines), lines[0], lines[len(lines)-1])

	if len(lines) < 6000 {
		t.Errorf("expected at least 6000 character entries, got %d", len(lines))
	}
}

func TestEnsureCharDictCreatesFile(t *testing.T) {
	model := testModelPath("ch_PP-OCRv5_rec_server_infer.onnx")
	if _, err := os.Stat(model); err != nil {
		t.Skipf("skipping: %s not found", model)
	}

	tmp := t.TempDir() + "/test_keys.txt"

	d, err := NewCharsetDict(model, tmp)
	if err != nil {
		t.Fatalf("NewCharsetDict: %v", err)
	}

	entries := d.Entries()
	t.Logf("loaded %d entries (incl. blank at 0)", len(entries))

	if len(entries) < 6000 {
		t.Errorf("expected at least 6000 dict entries, got %d", len(entries))
	}

	// Calling again should be a no-op (file already exists).
	d2, err := NewCharsetDict(model, tmp)
	if err != nil {
		t.Fatalf("second NewCharsetDict: %v", err)
	}
	if len(d2.Entries()) != len(entries) {
		t.Errorf("second load: got %d entries, want %d", len(d2.Entries()), len(entries))
	}
}

// ---------------------------------------------------------------------------
// CharsetDict model-free tests
// ---------------------------------------------------------------------------

func TestCharsetDictLoad_FromTempFile(t *testing.T) {
	tmp, err := os.CreateTemp(t.TempDir(), "dict*.txt")
	if err != nil {
		t.Fatalf("CreateTemp: %v", err)
	}
	if _, err := tmp.WriteString("A\nB\nC\n"); err != nil {
		t.Fatalf("WriteString: %v", err)
	}
	tmp.Close()

	d := &CharsetDict{}
	if err := d.Load(tmp.Name()); err != nil {
		t.Fatalf("Load: %v", err)
	}
	entries := d.Entries()
	// entries = ["" (blank), "A", "B", "C", " " (trailing space)]
	if len(entries) != 5 {
		t.Errorf("len(entries)=%d want 5 (blank + A + B + C + space)", len(entries))
	}
	if entries[0] != "" {
		t.Errorf("entries[0]=%q want \"\" (blank)", entries[0])
	}
	if entries[1] != "A" || entries[2] != "B" || entries[3] != "C" {
		t.Errorf("entries[1..3]=%v want [A B C]", entries[1:4])
	}
	if entries[4] != " " {
		t.Errorf("entries[4]=%q want \" \" (trailing space)", entries[4])
	}
}

func TestCharsetDictDecode_AllBlanks(t *testing.T) {
	d := &CharsetDict{entries: []string{"", "A", "B", "C", " "}}
	// T=3, numClasses=5; argmax at each step = 0 (blank).
	logits := make([]float32, 3*5)
	for t := 0; t < 3; t++ {
		logits[t*5+0] = 0.9 // class 0 is highest → blank
	}
	text, score := d.Decode(logits, 3, 5)
	if text != "" {
		t.Errorf("text=%q want \"\"", text)
	}
	if score != 0 {
		t.Errorf("score=%f want 0", score)
	}
}

func TestCharsetDictDecode_Normal(t *testing.T) {
	d := &CharsetDict{entries: []string{"", "A", "B", "C", " "}}
	// T=3, numClasses=5; sequence A B C.
	logits := make([]float32, 3*5)
	logits[0*5+1] = 0.9 // t=0 → class 1 (A)
	logits[1*5+2] = 0.9 // t=1 → class 2 (B)
	logits[2*5+3] = 0.9 // t=2 → class 3 (C)
	text, score := d.Decode(logits, 3, 5)
	if text != "ABC" {
		t.Errorf("text=%q want \"ABC\"", text)
	}
	if score <= 0 {
		t.Errorf("score=%f want > 0", score)
	}
}

func TestCharsetDictDecode_CTCCollapse(t *testing.T) {
	d := &CharsetDict{entries: []string{"", "A", "B", "C", " "}}
	// Sequence classes: [1, 1, 0, 1] = A, A, blank, A.
	// Consecutive A's collapse to one; blank resets duplicate-suppression,
	// so the final A is emitted again → "AA".
	logits := make([]float32, 4*5)
	logits[0*5+1] = 0.9 // A
	logits[1*5+1] = 0.9 // A (collapsed)
	logits[2*5+0] = 0.9 // blank
	logits[3*5+1] = 0.9 // A (re-emitted after blank reset)
	text, _ := d.Decode(logits, 4, 5)
	if text != "AA" {
		t.Errorf("text=%q want \"AA\" (blank resets dup-suppress, A re-emitted)", text)
	}
}

func TestCharsetDictDecode_Score(t *testing.T) {
	d := &CharsetDict{entries: []string{"", "A", "B", "C", " "}}
	// T=1, single character; bestP=0.9 → score = exp(log(0.9)/1) = 0.9.
	logits := []float32{0.1, 0.9, 0.1, 0.1, 0.1}
	_, score := d.Decode(logits, 1, 5)
	if math.Abs(score-0.9) > 1e-6 {
		t.Errorf("score=%f want ≈0.9", score)
	}
}

func TestCharsetDictDecode_Empty(t *testing.T) {
	d := &CharsetDict{entries: []string{"", "A", "B"}}
	text, score := d.Decode(nil, 0, 3)
	if text != "" || score != 0 {
		t.Errorf("T=0: got (%q, %f) want (\"\", 0)", text, score)
	}
	text, score = d.Decode(nil, 3, 0)
	if text != "" || score != 0 {
		t.Errorf("numClasses=0: got (%q, %f) want (\"\", 0)", text, score)
	}
}

// ---------------------------------------------------------------------------
// pbScanner and parseStringStringEntry tests
// ---------------------------------------------------------------------------

func TestPBScanner_ReadVarint(t *testing.T) {
	cases := []struct {
		data []byte
		want uint64
	}{
		{[]byte{0x01}, 1},
		{[]byte{0x00}, 0},
		// 300 = 0x12C → [0xAC, 0x02]
		{[]byte{0xAC, 0x02}, 300},
		// 127 → [0x7F]
		{[]byte{0x7F}, 127},
	}
	for _, tc := range cases {
		s := newPBScanner(bufio.NewReader(bytes.NewReader(tc.data)))
		got, err := s.readVarint()
		if err != nil {
			t.Errorf("readVarint(%x): %v", tc.data, err)
			continue
		}
		if got != tc.want {
			t.Errorf("readVarint(%x)=%d want %d", tc.data, got, tc.want)
		}
	}
}

func TestParseStringStringEntry(t *testing.T) {
	// Hand-encode StringStringEntryProto {key="hello", value="world"}.
	// Field 1 (key), wire 2:   tag=0x0A, len=5, "hello"
	// Field 2 (value), wire 2: tag=0x12, len=5, "world"
	data := []byte{
		0x0A, 0x05, 'h', 'e', 'l', 'l', 'o',
		0x12, 0x05, 'w', 'o', 'r', 'l', 'd',
	}
	key, value, err := parseStringStringEntry(data)
	if err != nil {
		t.Fatalf("parseStringStringEntry: %v", err)
	}
	if key != "hello" {
		t.Errorf("key=%q want \"hello\"", key)
	}
	if value != "world" {
		t.Errorf("value=%q want \"world\"", value)
	}
}

// splitLines splits on \n and drops any trailing empty string.
func splitLines(s string) []string {
	out := make([]string, 0)
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			out = append(out, s[start:i])
			start = i + 1
		}
	}
	if start < len(s) {
		out = append(out, s[start:])
	}
	return out
}
