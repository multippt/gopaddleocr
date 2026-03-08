package server

import (
	"encoding/base64"
	"image"
	"io"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/pkg/errors"
	_ "golang.org/x/image/webp"
)

// ---------------------------------------------------------------------------
// Helper: parse image bytes from multipart or JSON request
// ---------------------------------------------------------------------------

func (s *Server) parseImage(c *gin.Context) (image.Image, error) {
	data, err := s.parseImageBytes(c)
	if err != nil {
		return nil, errors.Wrap(err, "cannot parse image")
	}
	if data == nil {
		return nil, errors.New("empty image")
	}
	img, err := s.ocrEngine.DecodeImage(data)
	if err != nil {
		return nil, errors.Wrap(err, "cannot decode image")
	}
	return img, nil
}

func (s *Server) parseImageBytes(c *gin.Context) ([]byte, error) {
	ct := c.GetHeader("Content-Type")
	switch {
	case strings.Contains(ct, "multipart/form-data"):
		f, _, err := c.Request.FormFile("image")
		if err != nil {
			return nil, err
		}
		defer func() {
			_ = f.Close()
		}()
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
		return nil, errors.New("unsupported content type")
	}
}
