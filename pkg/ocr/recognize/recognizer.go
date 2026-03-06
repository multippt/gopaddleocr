package recognize

import "image"

type Result struct {
	Text  string
	Score float64
}

type Recognizer interface {
	Recognize(img image.Image, quad [4][2]int) (Result, error)
	Close() error
}
