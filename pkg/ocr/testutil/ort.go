// Package testutil provides shared test helpers for ORT-backed model tests.
package testutil

import (
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"testing"

	ort "github.com/yalue/onnxruntime_go"
)

var (
	ortOnce sync.Once
	ortErr  error
)

// RequireORT initializes the ONNX Runtime once per test binary (sync.Once)
// and skips t if the runtime library cannot be loaded.
// Safe to call from multiple tests in parallel — initialization happens exactly once.
func RequireORT(t *testing.T) {
	t.Helper()
	ortOnce.Do(func() {
		if !ort.IsInitialized() {
			ort.SetSharedLibraryPath(findORTLib())
			ortErr = ort.InitializeEnvironment()
		}
	})
	if ortErr != nil {
		t.Skipf("ORT not available: %v", ortErr)
	}
}

// findORTLib returns the path to the ORT shared library, preferring the
// ORT_LIB_PATH environment variable, then walking up from the working
// directory to the module root.
func findORTLib() string {
	if p := os.Getenv("ORT_LIB_PATH"); p != "" {
		return p
	}

	libName := "libonnxruntime.so"
	switch runtime.GOOS {
	case "windows":
		libName = "onnxruntime.dll"
	case "darwin":
		libName = "libonnxruntime.dylib"
	}

	dir, _ := os.Getwd()
	for {
		candidate := filepath.Join(dir, "onnxruntime", "lib", libName)
		if _, err := os.Stat(candidate); err == nil {
			return candidate
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	return filepath.Join("onnxruntime", "lib", libName) // best-effort fallback
}

// RequireModel skips t if the model file at path does not exist.
func RequireModel(t *testing.T, path string) {
	t.Helper()
	if _, err := os.Stat(path); err != nil {
		t.Skipf("model not found at %q — skipping", path)
	}
}

