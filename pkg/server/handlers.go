package server

import (
	"encoding/json"
	"image"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/multippt/gopaddleocr/pkg/colordetect"
	"github.com/multippt/gopaddleocr/pkg/ocr"
	"github.com/multippt/gopaddleocr/pkg/ocr/utils"
)

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	if s.ocrEngine == nil || !s.ocrEngine.IsReady() {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"detail": "ocr engine not ready"})
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *Server) handleOCR(w http.ResponseWriter, r *http.Request) {
	img, err := s.parseImage(r)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"detail": "invalid or missing image"})
		return
	}

	t0 := time.Now()
	results, err := s.ocrEngine.ImageRunOCR(img)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"detail": err.Error()})
		return
	}
	elapsed := float64(time.Since(t0).Microseconds()) / 1000.0

	lines, texts := buildOCRLines(img, results)
	writeJSON(w, http.StatusOK, OCRResponse{
		Results:   lines,
		FullText:  strings.Join(texts, "\n"),
		ElapsedMs: elapsed,
	})
}

func (s *Server) handleDetect(w http.ResponseWriter, r *http.Request) {
	img, err := s.parseImage(r)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"detail": "invalid or missing image"})
		return
	}

	s.doImageDetect(w, r, img)
}

func (s *Server) handleSessionCreate(w http.ResponseWriter, r *http.Request) {
	img, err := s.parseImage(r)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"detail": "invalid or missing image"})
		return
	}

	// Honour optional existing session_id from form or JSON.
	sid := r.FormValue("session_id")
	if sid == "" {
		sid = s.sessionManager.NewID()
	}
	s.sessionManager.Store(sid, img)
	writeJSON(w, http.StatusOK, SessionResponse{SessionID: sid})
}

func (s *Server) handleSessionDelete(w http.ResponseWriter, r *http.Request) {
	sid := r.FormValue("session_id")
	s.sessionManager.Delete(sid)
	writeJSON(w, http.StatusOK, SessionResponse{SessionID: sid})
}

func (s *Server) handleSessionDetect(w http.ResponseWriter, r *http.Request) {
	var req SessionReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.SessionID == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"detail": "missing session_id"})
		return
	}

	img, err := s.sessionManager.Get(req.SessionID)
	if err != nil {
		writeJSON(w, http.StatusNotFound, map[string]string{"detail": err.Error()})
		return
	}

	s.doImageDetect(w, r, img)
}

func (s *Server) doImageDetect(w http.ResponseWriter, r *http.Request, img image.Image) {
	t0 := time.Now()
	detBoxes, err := s.ocrEngine.ImageDetectBoundingBoxes(img)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"detail": err.Error()})
		return
	}
	elapsed := float64(time.Since(t0).Microseconds()) / 1000.0

	boxes := make([][][2]int, len(detBoxes))
	for i, b := range detBoxes {
		q := b.Quad
		boxes[i] = [][2]int{q[0], q[1], q[2], q[3]}
	}
	writeJSON(w, http.StatusOK, DetectResponse{Boxes: boxes, ElapsedMs: elapsed})
}

const cropPadding = 10

func (s *Server) handleSessionOCR(w http.ResponseWriter, r *http.Request) {
	var req SessionOCRReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.SessionID == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"detail": "missing session_id"})
		return
	}

	img, err := s.sessionManager.Get(req.SessionID)
	if err != nil {
		writeJSON(w, http.StatusNotFound, map[string]string{"detail": err.Error()})
		return
	}

	t0 := time.Now()

	if len(req.BoundingBoxes) > 0 {
		writeJSON(w, http.StatusOK, s.runSessionOCRWithBoxes(img, req, t0))
	} else {
		writeJSON(w, http.StatusOK, s.runSessionOCRFull(img, t0))
	}
}

func (s *Server) runSessionOCRFull(img image.Image, t0 time.Time) OCRResponse {
	results, err := s.ocrEngine.ImageRunOCR(img)
	elapsed := float64(time.Since(t0).Microseconds()) / 1000.0
	if err != nil || len(results) == 0 {
		return OCRResponse{Results: []OCRLine{}, ElapsedMs: elapsed}
	}
	lines, texts := buildOCRLines(img, results)
	return OCRResponse{Results: lines, FullText: strings.Join(texts, "\n"), ElapsedMs: elapsed}
}

func (s *Server) runSessionOCRWithBoxes(img image.Image, req SessionOCRReq, t0 time.Time) OCRResponse {
	var allLines []OCRLine
	var allTexts []string
	totalElapsed := 0.0

	for _, bb := range req.BoundingBoxes {
		quad := [4][2]int{{bb.X1, bb.Y1}, {bb.X2, bb.Y2}, {bb.X3, bb.Y3}, {bb.X4, bb.Y4}}

		cropped := cropByQuad(img, quad)

		bT0 := time.Now()
		results, err := s.ocrEngine.ImageRunOCR(cropped)
		totalElapsed += float64(time.Since(bT0).Microseconds()) / 1000.0
		if err != nil || len(results) == 0 {
			continue
		}

		// Compute AABB of the original quad to get translation offsets.
		xs := [4]int{bb.X1, bb.X2, bb.X3, bb.X4}
		ys := [4]int{bb.Y1, bb.Y2, bb.Y3, bb.Y4}
		minX, minY := xs[0], ys[0]
		maxX, maxY := xs[0], ys[0]
		for _, x := range xs {
			if x < minX {
				minX = x
			}
			if x > maxX {
				maxX = x
			}
		}
		for _, y := range ys {
			if y < minY {
				minY = y
			}
			if y > maxY {
				maxY = y
			}
		}

		for i, r := range results {
			// Translate from cropped-image space back to original image space.
			translatedBox := make([][2]int, 4)
			for j, pt := range r.Box {
				tx := pt[0] - cropPadding + minX
				ty := pt[1] - cropPadding + minY
				if tx < minX {
					tx = minX
				}
				if tx > maxX {
					tx = maxX
				}
				if ty < minY {
					ty = minY
				}
				if ty > maxY {
					ty = maxY
				}
				translatedBox[j] = [2]int{tx, ty}
			}
			results[i].Box = translatedBox
		}

		lines, texts := buildOCRLines(cropped, results)
		// Fix boxes in lines to use translated coordinates.
		for i, r := range results {
			lines[i].Box = r.Box
		}
		allLines = append(allLines, lines...)
		allTexts = append(allTexts, texts...)
	}

	elapsed := float64(time.Since(t0).Microseconds()) / 1000.0
	_ = totalElapsed
	if allLines == nil {
		allLines = []OCRLine{}
	}
	return OCRResponse{
		Results:   allLines,
		FullText:  strings.Join(allTexts, "\n"),
		ElapsedMs: elapsed,
	}
}

// cropByQuad crops img to the AABB of quad with polygon masking and padding.
func cropByQuad(img image.Image, quad [4][2]int) image.Image {
	xs := [4]int{quad[0][0], quad[1][0], quad[2][0], quad[3][0]}
	ys := [4]int{quad[0][1], quad[1][1], quad[2][1], quad[3][1]}
	minX, minY := xs[0], ys[0]
	maxX, maxY := xs[0], ys[0]
	for _, x := range xs {
		if x < minX {
			minX = x
		}
		if x > maxX {
			maxX = x
		}
	}
	for _, y := range ys {
		if y < minY {
			minY = y
		}
		if y > maxY {
			maxY = y
		}
	}

	bounds := img.Bounds()
	minX = utils.ClampInt(minX, bounds.Min.X, bounds.Max.X-1)
	minY = utils.ClampInt(minY, bounds.Min.Y, bounds.Max.Y-1)
	maxX = utils.ClampInt(maxX, bounds.Min.X+1, bounds.Max.X)
	maxY = utils.ClampInt(maxY, bounds.Min.Y+1, bounds.Max.Y)

	w := maxX - minX
	h := maxY - minY
	if w <= 0 || h <= 0 {
		return image.NewRGBA(image.Rect(0, 0, 1, 1))
	}

	// Build polygon mask in crop-local coords.
	localPts := [4][2]float64{
		{float64(quad[0][0] - minX), float64(quad[0][1] - minY)},
		{float64(quad[1][0] - minX), float64(quad[1][1] - minY)},
		{float64(quad[2][0] - minX), float64(quad[2][1] - minY)},
		{float64(quad[3][0] - minX), float64(quad[3][1] - minY)},
	}

	const padColor = uint8(255)
	out := image.NewRGBA(image.Rect(0, 0, w+2*cropPadding, h+2*cropPadding))
	// Fill with white.
	for y := 0; y < h+2*cropPadding; y++ {
		for x := 0; x < w+2*cropPadding; x++ {
			out.SetRGBA(x, y, utils.ColorRGBAWhite)
		}
	}
	_ = padColor
	// Copy pixels that are inside the polygon.
	for py := 0; py < h; py++ {
		for px := 0; px < w; px++ {
			if utils.PointInQuad4(float64(px), float64(py), localPts) {
				r32, g32, b32, a32 := img.At(minX+px, minY+py).RGBA()
				out.SetRGBA(px+cropPadding, py+cropPadding, utils.ColorRGBA(r32, g32, b32, a32))
			}
		}
	}
	return out
}

// buildOCRLines converts raw engine results into OCRLine responses with color extraction.
func buildOCRLines(img image.Image, results []ocr.Result) ([]OCRLine, []string) {
	lines := make([]OCRLine, len(results))
	texts := make([]string, len(results))

	colorResults := make([]colordetect.ColorResult, len(results))

	var wg sync.WaitGroup
	for i, r := range results {
		wg.Add(1)
		go func(idx int, res ocr.Result) {
			defer wg.Done()
			colorResults[idx] = colordetect.ComputeTextColorResult(img, res.Box, res.Text)
		}(i, r)
	}
	wg.Wait()

	for i, r := range results {
		box := make([][2]int, 4)
		for j, pt := range r.Box {
			box[j] = pt
		}
		lines[i] = OCRLine{
			Box:        box,
			Text:       r.Text,
			Score:      r.Score,
			TextColor:  colorResults[i].TextColor,
			WordColors: colorResults[i].WordColors,
		}
		texts[i] = r.Text
	}
	return lines, texts
}
