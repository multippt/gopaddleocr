package server

import (
	"bytes"
	"encoding/base64"
	"image"
	"image/draw"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
	_ "golang.org/x/image/webp"
)

// ---------------------------------------------------------------------------
// Helper: parse image bytes from multipart or JSON request
// ---------------------------------------------------------------------------

func parseImageBytes(c *gin.Context) ([]byte, error) {
	ct := c.GetHeader("Content-Type")
	switch {
	case strings.Contains(ct, "multipart/form-data"):
		f, _, err := c.Request.FormFile("image")
		if err != nil {
			return nil, err
		}
		defer f.Close()
		return io.ReadAll(f)

	case strings.Contains(ct, "application/json"):
		var body struct {
			Image string `json:"image"`
		}
		if err := c.ShouldBindJSON(&body); err != nil {
			return nil, err
		}
		// Accept both standard and URL-safe base64
		body.Image = strings.TrimSpace(body.Image)
		raw, err := base64.StdEncoding.DecodeString(body.Image)
		if err != nil {
			raw, err = base64.URLEncoding.DecodeString(body.Image)
			if err != nil {
				return nil, err
			}
		}
		return raw, nil

	default:
		c.Status(http.StatusUnsupportedMediaType)
		return nil, nil
	}
}

func decodeImage(data []byte) (image.Image, error) {
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	// Normalise to RGBA
	rgba := image.NewRGBA(img.Bounds())
	draw.Draw(rgba, rgba.Bounds(), img, img.Bounds().Min, draw.Src)
	return rgba, nil
}
