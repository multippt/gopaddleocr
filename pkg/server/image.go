package server

import (
	"encoding/base64"
	"encoding/json"
	"image"
	"io"
	"net/http"
	"strings"

	"github.com/pkg/errors"
)

// ---------------------------------------------------------------------------
// Helper: parse image bytes from multipart or JSON request
// ---------------------------------------------------------------------------

func (s *Server) parseImage(r *http.Request) (image.Image, error) {
	data, err := s.parseImageBytes(r)
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

func (s *Server) parseImageBytes(r *http.Request) ([]byte, error) {
	ct := r.Header.Get("Content-Type")
	switch {
	case strings.Contains(ct, "multipart/form-data"):
		f, _, err := r.FormFile("image")
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
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
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
