package detect

import "image"

// Box is the result of a detection stage.
// ClassID is -1 for generic text detection (PaddleOCR), or a layout class index (DocLayoutV3).
// Order is -1 if reading order is not provided.
type Box struct {
	Quad    [4][2]int
	Score   float64
	ClassID int
	Order   int
}

type Detector interface {
	Detect(img image.Image) ([]Box, error)
	Close() error
}
