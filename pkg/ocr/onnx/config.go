package onnx

import ort "github.com/yalue/onnxruntime_go"

type Config struct {
	InputName  string
	OutputName string
	Options    *ort.SessionOptions
}
