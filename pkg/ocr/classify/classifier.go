package classify

import "image"

type Classifier interface {
	Classify(img image.Image, quad [4][2]int) (rotated bool, err error)
	Close() error
}
