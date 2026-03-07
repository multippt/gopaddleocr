package openai

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/png"
	"io"
	"net/http"

	"github.com/multippt/gopaddleocr/pkg/ocr/common"
	"github.com/multippt/gopaddleocr/pkg/ocr/recognize"
	"github.com/multippt/gopaddleocr/pkg/ocr/utils"
)

const ModelName = "openai-rec"

type ModelConfig struct {
	Endpoint     string // e.g. "https://open.bigmodel.cn/api/paas/v4"
	APIKey       string
	Model        string // e.g. "glm-4v-plus"
	SystemPrompt string // default: "You are an OCR assistant. Output only the exact text you see, with no explanation."
	UserPrompt   string // default: "Please transcribe all text in this image exactly as it appears."
	common.BaseModelConfig
}

// ---------------------------------------------------------------------------
// Model implements recognize.Recognizer via an OpenAI-compatible chat/completions API
// ---------------------------------------------------------------------------

type Model struct {
	config ModelConfig
	client *http.Client
}

func NewModel() *Model {
	return &Model{client: &http.Client{}}
}

func (m *Model) GetName() string { return ModelName }

func (m *Model) GetDefaultConfig() common.ModelConfig {
	return &ModelConfig{
		Endpoint:        "http://localhost:8000/v1",
		APIKey:          "",
		Model:           "zai-org/GLM-OCR",
		SystemPrompt:    "You are an OCR assistant. Output only the exact text you see, with no explanation.",
		UserPrompt:      "Please transcribe all text in this image exactly as it appears.",
		BaseModelConfig: common.BaseModelConfig{},
	}
}

func (m *Model) Init(config common.ModelConfig) error {
	cfg, ok := config.(*ModelConfig)
	if !ok {
		return fmt.Errorf("openai recognizer: expected *ModelConfig, got %T", config)
	}
	if cfg.Endpoint == "" {
		return fmt.Errorf("openai recognizer: Endpoint is required")
	}
	if cfg.Model == "" {
		return fmt.Errorf("openai recognizer: Model is required")
	}
	if cfg.SystemPrompt == "" {
		cfg.SystemPrompt = "You are an OCR assistant. Output only the exact text you see, with no explanation."
	}
	if cfg.UserPrompt == "" {
		cfg.UserPrompt = "Please transcribe all text in this image exactly as it appears."
	}
	m.config = *cfg
	return nil
}

func (m *Model) Close() error { return nil }

// Recognize crops the quad region and sends it to the OpenAI-compatible API for recognition.
func (m *Model) Recognize(img image.Image, quad [4][2]int) (recognize.Result, error) {
	ordered := utils.OrderPoints4(utils.FloatQuad(quad))

	// Determine crop dimensions.
	topW := utils.PointDistance(ordered[0], ordered[1])
	bottomW := utils.PointDistance(ordered[3], ordered[2])
	leftH := utils.PointDistance(ordered[0], ordered[3])
	rightH := utils.PointDistance(ordered[1], ordered[2])
	dstW := int(max64(topW, bottomW))
	dstH := int(max64(leftH, rightH))
	if dstW < 1 {
		dstW = 1
	}
	if dstH < 1 {
		dstH = 1
	}

	crop := utils.PerspectiveWarp(img, ordered, dstW, dstH)
	if crop == nil {
		return recognize.Result{}, fmt.Errorf("openai recognizer: perspective warp returned nil")
	}

	// Encode as base64 PNG.
	var buf bytes.Buffer
	if err := png.Encode(&buf, crop); err != nil {
		return recognize.Result{}, fmt.Errorf("openai recognizer: encode image: %w", err)
	}
	b64 := base64.StdEncoding.EncodeToString(buf.Bytes())
	dataURL := "data:image/png;base64," + b64

	// Build request body.
	reqBody := map[string]any{
		"model": m.config.Model,
		"messages": []any{
			map[string]any{
				"role":    "system",
				"content": m.config.SystemPrompt,
			},
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "image_url",
						"image_url": map[string]any{
							"url": dataURL,
						},
					},
					map[string]any{
						"type": "text",
						"text": m.config.UserPrompt,
					},
				},
			},
		},
	}

	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return recognize.Result{}, fmt.Errorf("openai recognizer: marshal request: %w", err)
	}

	url := m.config.Endpoint + "/chat/completions"
	req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(bodyBytes))
	if err != nil {
		return recognize.Result{}, fmt.Errorf("openai recognizer: create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if m.config.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+m.config.APIKey)
	}

	resp, err := m.client.Do(req)
	if err != nil {
		return recognize.Result{}, fmt.Errorf("openai recognizer: http request: %w", err)
	}
	defer resp.Body.Close()

	respBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return recognize.Result{}, fmt.Errorf("openai recognizer: read response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return recognize.Result{}, fmt.Errorf("openai recognizer: API returned %d: %s", resp.StatusCode, respBytes)
	}

	// Parse choices[0].message.content.
	var result struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(respBytes, &result); err != nil {
		return recognize.Result{}, fmt.Errorf("openai recognizer: parse response: %w", err)
	}
	if len(result.Choices) == 0 {
		return recognize.Result{}, fmt.Errorf("openai recognizer: no choices in response")
	}

	text := result.Choices[0].Message.Content
	return recognize.Result{Text: text, Score: 1.0}, nil
}

func max64(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}
