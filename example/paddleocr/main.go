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

	engine := ocr.NewEngine(
		ocr.WithWorkflowType("PaddleOCR"), // or "GLM-OCR"
		ocr.WithBoxMerge(true),
	)
	defer engine.Close()

	if err := engine.Init(); err != nil {
		fmt.Printf("%v\n", err)
		return
	}

	results, err := engine.RunOCR(data)
	if err != nil {
		fmt.Printf("%v\n", err)
		return
	}

	// results: []ocr.Result with Box, Text, Score (and optional Children)
	fmt.Printf("%v\n", results)
}
