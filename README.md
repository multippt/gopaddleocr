# GoPaddleOCR

Golang implementation of PaddleOCR based on onnxruntime.

## Prerequisites

### CGO

This project uses [onnxruntime_go](https://github.com/yalue/onnxruntime_go), which depends on the ONNX Runtime C API. You must build with **CGO enabled** (the default when a C compiler is available). Ensure a C toolchain is installed:

- **Windows:** Install [MinGW-w64](https://www.mingw-w64.org/) or use the compiler that comes with MSVC build tools, and have `gcc` (or `cc`) on your PATH.
- **Linux:** `sudo apt install build-essential` (Debian/Ubuntu) or equivalent.
- **macOS:** Xcode Command Line Tools: `xcode-select --install`.

### ONNX Runtime

The engine loads the ONNX Runtime shared library at runtime. The repo includes scripts that download and extract a prebuilt build into `./onnxruntime/` (layout expected by the engine).

**Using the download scripts:**

- **Linux / macOS:** From the project root, run `./download_onnx.sh`. Requires `curl`. Optional: set `ORT_VERSION` (default `1.24.2`).
- **Windows:** From the project root, run `download_onnx.bat`. Requires `curl` and PowerShell. Optional: set env `ORT_VERSION`.

The scripts create `./onnxruntime/` with the correct `lib/` layout. If the directory already exists, they skip the download.

**Manual setup:** Download a prebuilt package from [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases), extract it so the shared library is at `./onnxruntime/lib/` (`onnxruntime.dll` on Windows, `libonnxruntime.so` on Linux, `libonnxruntime.dylib` on macOS). To use a different path, set environment variable `ORT_LIB_PATH` to the full path of the library file.

### Model download

The default PaddleOCR workflow expects the following files under `./models/` (relative to the process working directory):

| File | Description |
|------|-------------|
| `ch_PP-OCRv5_server_det.onnx` | Detection model |
| `ch_ppocr_mobile_v2.0_cls_infer.onnx` | Direction classifier |
| `ch_PP-OCRv5_rec_server_infer.onnx` | Recognition model |

**Using the download scripts:**

- **Linux / macOS:** From the project root, run `./download_models.sh`. Downloads the three ONNX files (and `PP-DocLayoutV3.onnx`) from [RapidOCR on ModelScope](https://www.modelscope.cn/models/RapidAI/RapidOCR/tree/master/onnx/) and Hugging Face into `./models/`. Requires `curl`.
- **Windows:** From the project root, run `download_models.bat`. Same set of models into `./models/`. Requires `curl`.

**Manual setup:** Use the same sources as the scripts to avoid compatibility issues. Download from [RapidOCR on ModelScope](https://www.modelscope.cn/models/RapidAI/RapidOCR/tree/master/onnx/): `PP-OCRv5/det/ch_PP-OCRv5_server_det.onnx`, `PP-OCRv5/rec/ch_PP-OCRv5_rec_server_infer.onnx`, and `PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_infer.onnx` (save into `./models/` with the names from the table). For layout detection, [PP-DocLayoutV3-ONNX](https://huggingface.co/alex-dinh/PP-DocLayoutV3-ONNX/resolve/main/PP-DocLayoutV3.onnx) → `./models/PP-DocLayoutV3.onnx`. To use a different directory or filenames, configure the engine with `ocr.WithModelConfig` before calling `Init()`.

## Using the engine as a library

Import the OCR package and create an engine with optional configuration:

```go
package main

import (
	"fmt"
	"image"
	_ "image/png"
	"os"

	"github.com/multippt/gopaddleocr/pkg/ocr"
)

func main() {
	// Load an image from file.
	f, err := os.Open("image.png")
	if err != nil {
		fmt.Printf("%v\n", err)
		return
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		fmt.Printf("%v\n", err)
		return
	}

	// Create engine (optional: ocr.WithModelConfig, ocr.WithWorkflowType, ocr.WithBoxMerge)
	engine := ocr.NewEngine(
		ocr.WithWorkflowType("PaddleOCR"), // or "GLM-OCR"
		ocr.WithBoxMerge(true),
	)
	defer engine.Close()

	// Set ORT_LIB_PATH if needed (default: ./onnxruntime/lib/onnxruntime.dll)
	if err := engine.Init(); err != nil {
		fmt.Printf("%v\n", err)
		return
	}

	// Run full OCR on an image
	results, _ := engine.RunOCR(img)
	// results: []ocr.Result with Box, Text, Score (and optional Children)
	fmt.Printf("%v\n", results)

	// Or run detection only (text boxes, no recognition)
	boxes, _ := engine.DetectOnly(img)
	fmt.Printf("%v\n", boxes)
}
```

## Running the server

From the project root:

```bash
go run .
```

The listening address can be revised with `-listen` or `-l`:

```bash
go run . -listen 0.0.0.0:8051
```

### Example request

```bash
curl -X POST http://localhost:8051/ocr \
  -F "image=@/path/to/your/image.png"
```

### Example response

```json
{
  "results": [
    {
      "box": [[10, 20], [100, 20], [100, 40], [10, 40]],
      "text": "Hello world",
      "score": 0.98,
      "text_color": [0, 0, 0],
      "word_colors": []
    }
  ],
  "full_text": "Hello world\n",
  "elapsed_ms": 45.2
}
```
