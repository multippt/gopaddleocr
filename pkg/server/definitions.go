package server

import "github.com/multippt/gopaddleocr/pkg/colordetect"

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

type OCRLine struct {
	Box        [][2]int                     `json:"box"`
	Text       string                       `json:"text"`
	Score      float64                      `json:"score"`
	TextColor  []int                        `json:"text_color,omitempty"`
	WordColors []colordetect.WordColorEntry `json:"word_colors,omitempty"`
}

type OCRResponse struct {
	Results   []OCRLine `json:"results"`
	FullText  string    `json:"full_text"`
	ElapsedMs float64   `json:"elapsed_ms"`
}

type DetectResponse struct {
	Boxes     [][][2]int `json:"boxes"`
	ElapsedMs float64    `json:"elapsed_ms"`
}

type SessionResponse struct {
	SessionID string `json:"session_id"`
}

type BoundingBoxReq struct {
	X1 int `json:"x1"`
	Y1 int `json:"y1"`
	X2 int `json:"x2"`
	Y2 int `json:"y2"`
	X3 int `json:"x3"`
	Y3 int `json:"y3"`
	X4 int `json:"x4"`
	Y4 int `json:"y4"`
}

type SessionOCRReq struct {
	SessionID     string           `json:"session_id"`
	Language      string           `json:"language,omitempty"`
	Model         string           `json:"model,omitempty"`
	BoundingBoxes []BoundingBoxReq `json:"bounding_boxes,omitempty"`
}
