# GoPaddleOCR

Golang implementation of PaddleOCR based on onnxruntime.

## Quick Start

For Linux / macOS:
```bash
./download_onnx.sh
./download_models.sh
go run .
```

For Windows:
```bash
download_onnx.bat
download_models.bat
go run .
```

## Prerequisites

### CGO

This project uses [onnxruntime_go](https://github.com/yalue/onnxruntime_go), which depends on the ONNX Runtime C API. You must build with **CGO enabled**, ensure a C toolchain is installed:

- **Windows:** Install [MinGW-w64](https://www.mingw-w64.org/) or [TDM-GCC](https://jmeubank.github.io/tdm-gcc/), and have `gcc` (or `cc`) on your PATH.
- **Linux:** `sudo apt install build-essential` (Debian/Ubuntu) or equivalent.
- **macOS:** Xcode Command Line Tools: `xcode-select --install`.

### ONNX Runtime

The engine loads the ONNX Runtime shared library at runtime. The repo includes scripts that download and extract a prebuilt build into `./onnxruntime/` (layout expected by the engine).

**Using the download scripts:**

- **Linux / macOS:** From the project root, run `./download_onnx.sh`.
- **Windows:** From the project root, run `download_onnx.bat`.

**Manual setup:** 

Download a prebuilt package from [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases), extract it ./onnxruntime relative to the working directory. To use a different path, set environment variable `ORT_LIB_PATH` to the full path of the library file.

### Model download

The default PaddleOCR workflow expects the following files under `./models/` (relative to the process working directory):

| File | Description | Download Link                                                                   |
|------|-------------|---------------------------------------------------------------------------------|
| `ch_PP-OCRv5_server_det.onnx` | Detection model | https://www.modelscope.cn/models/RapidAI/RapidOCR/tree/master/onnx/PP-OCRv5/det |
| `ch_ppocr_mobile_v2.0_cls_infer.onnx` | Direction classifier | https://www.modelscope.cn/models/RapidAI/RapidOCR/tree/master/onnx/PP-OCRv4/cls |
| `ch_PP-OCRv5_rec_server_infer.onnx` | Recognition model | https://www.modelscope.cn/models/RapidAI/RapidOCR/tree/master/onnx/PP-OCRv5/rec |

**Using the download scripts:**

For simplicity, the models can be quickly downloaded with the scripts.

- **Linux / macOS:** From the project root, run `./download_models.sh`.
- **Windows:** From the project root, run `download_models.bat`.

## Usage (Library)

Import the OCR package and create an engine with optional configuration:

```go
package main

import (
	"fmt"
	"io"
	"net/http"

	"github.com/multippt/gopaddleocr/pkg/ocr"
)

func downloadImage(url string) ([]byte, error) {
	resp, err := http.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	return io.ReadAll(resp.Body)
}

func main() {
	data, err := downloadImage(
		"https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png")
	if err != nil {
		fmt.Printf("%v\n", err)
		return
	}

	engine := ocr.NewEngine()
	if err := engine.Init(); err != nil {
		fmt.Printf("%v\n", err)
		return
	}
	defer engine.Close()

	results, err := engine.RunOCR(data)
	if err != nil {
		fmt.Printf("%v\n", err)
		return
	}

	// results: []ocr.Result with Box, Text, Score (and optional Children)
	fmt.Printf("%v\n", results)
}
```

## Usage (Server)

A basic server has been provided which exposes a basic OCR endpoint at `/ocr`.

```bash
go run . -listen 0.0.0.0:8051
```

### Example request

```bash
curl -X POST http://localhost:8051/ocr -F "image=@./image.png"
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

# License

The Paddle-OCR models originate from the Baidu's PaddlePaddle [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) project.

This project is released under the Apache 2.0 license.
