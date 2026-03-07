package classify

import "image"

type Classifier interface {
	Classify(img image.Image, quad [4][2]int) (rotated bool, err error)
	Close() error
}

// StubClassifier always returns (false, nil) — no rotation needed.
type StubClassifier struct{}

func (StubClassifier) Classify(_ image.Image, _ [4][2]int) (bool, error) { return false, nil }
func (StubClassifier) Close() error                                       { return nil }
