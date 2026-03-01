package recognize

import (
	"os"
	"testing"
)

func TestONNXMetadataExtract(t *testing.T) {
	const model = "./models/ch_PP-OCRv5_rec_server_infer.onnx"
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
	const model = "./models/ch_PP-OCRv5_rec_server_infer.onnx"
	if _, err := os.Stat(model); err != nil {
		t.Skipf("skipping: %s not found", model)
	}

	tmp := t.TempDir() + "/test_keys.txt"

	if err := EnsureCharDict(model, tmp); err != nil {
		t.Fatalf("EnsureCharDict: %v", err)
	}

	dict, err := loadCharDict(tmp)
	if err != nil {
		t.Fatalf("loadCharDict: %v", err)
	}
	t.Logf("loaded %d entries (incl. blank at 0)", len(dict))

	if len(dict) < 6000 {
		t.Errorf("expected at least 6000 dict entries, got %d", len(dict))
	}

	// Calling again should be a no-op (file already exists).
	if err := EnsureCharDict(model, tmp); err != nil {
		t.Fatalf("second EnsureCharDict: %v", err)
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
