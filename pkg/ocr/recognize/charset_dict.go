package recognize

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"strings"
)

// ---------------------------------------------------------------------------
// CharsetDict
// ---------------------------------------------------------------------------

// CharsetDict holds the character list for CTC decoding (index 0 = blank, then
// file lines, then trailing space per PaddleOCR convention).
type CharsetDict struct {
	entries []string
}

// NewCharsetDict ensures the dict file exists at dictPath (extracting from the
// ONNX model at modelPath if needed), then loads it. Single entry point for
// "use this model and dict path."
func NewCharsetDict(modelPath, dictPath string) (*CharsetDict, error) {
	d := &CharsetDict{}
	if err := d.Ensure(modelPath, dictPath); err != nil {
		return nil, err
	}
	if err := d.Load(dictPath); err != nil {
		return nil, err
	}
	return d, nil
}

// Ensure ensures the dict file exists at dictPath; if not, extracts the
// embedded 'character' metadata from the ONNX model at modelPath and writes
// it to dictPath. Idempotent.
func (d *CharsetDict) Ensure(modelPath, dictPath string) error {
	if _, err := os.Stat(dictPath); err == nil {
		return nil // already cached
	}

	log.Printf("char dict %q not found — extracting from %s ...", dictPath, modelPath)

	raw, err := onnxMetadataLookup(modelPath, "character")
	if err != nil {
		return fmt.Errorf("read model metadata: %w", err)
	}

	f, err := os.Create(dictPath)
	if err != nil {
		return fmt.Errorf("create dict file: %w", err)
	}
	defer f.Close()

	if _, err = f.WriteString(raw); err != nil {
		return err
	}
	// bufio.Scanner ignores a trailing newline, but ensure one is present.
	if !strings.HasSuffix(raw, "\n") {
		if _, err = f.WriteString("\n"); err != nil {
			return err
		}
	}

	n := strings.Count(raw, "\n")
	log.Printf("char dict: wrote %d entries to %q", n, dictPath)
	return nil
}

// Load reads the dict file at path (one char per line). Index 0 is the CTC
// blank token; indices 1..N map to the file lines. A trailing space is appended.
func (d *CharsetDict) Load(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer func() {
		_ = f.Close()
	}()

	d.entries = []string{""} // index 0 = blank
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		d.entries = append(d.entries, sc.Text())
	}
	d.entries = append(d.entries, " ") // trailing space
	return sc.Err()
}

// Entries returns the character list (for tests or other callers).
func (d *CharsetDict) Entries() []string {
	return d.entries
}

// Decode runs argmax-CTC on logits shaped (T × numClasses) flattened using
// this dict. Returns the decoded string and the geometric-mean confidence.
func (d *CharsetDict) Decode(logits []float32, T, numClasses int) (string, float64) {
	charDict := d.entries
	if T == 0 || numClasses == 0 {
		return "", 0
	}
	if len(logits) < T*numClasses {
		return "", 0
	}

	type step struct {
		class int
		prob  float32
	}
	steps := make([]step, T)
	for t := 0; t < T; t++ {
		best := 0
		bestP := logits[t*numClasses]
		for c := 1; c < numClasses; c++ {
			if logits[t*numClasses+c] > bestP {
				bestP = logits[t*numClasses+c]
				best = c
			}
		}
		steps[t] = step{best, bestP}
	}

	var runes []rune
	var probs []float64
	prev := -1
	for _, s := range steps {
		if s.class == 0 {
			prev = 0
			continue
		}
		if s.class == prev {
			continue
		}
		if s.class < len(charDict) {
			for _, r := range charDict[s.class] {
				runes = append(runes, r)
			}
			probs = append(probs, float64(s.prob))
		}
		prev = s.class
	}

	if len(runes) == 0 {
		return "", 0
	}

	// Geometric mean confidence.
	logSum := 0.0
	for _, p := range probs {
		if p > 0 {
			logSum += math.Log(p)
		}
	}
	score := math.Exp(logSum / float64(len(probs)))

	return string(runes), score
}

// ---------------------------------------------------------------------------
// ONNX model metadata scanner
// ---------------------------------------------------------------------------

// onnxMetadataLookup opens the ONNX model file and scans its top-level
// protobuf fields until it finds a metadata_props entry (field 14) whose
// key matches targetKey.  It returns the associated value string.
//
// No ONNX protobuf schema is required — only a minimal wire-format decoder.
func onnxMetadataLookup(modelPath, targetKey string) (string, error) {
	f, err := os.Open(modelPath)
	if err != nil {
		return "", err
	}
	defer f.Close()

	s := newPBScanner(bufio.NewReaderSize(f, 1<<20)) // 1 MB read buffer
	return s.scanForMetadata(targetKey)
}

// ---------------------------------------------------------------------------
// Minimal protobuf wire-format scanner
// ---------------------------------------------------------------------------

// ONNX ModelProto field numbers (partial):
//
//	14 = metadata_props (repeated StringStringEntryProto)
//
// StringStringEntryProto:
//
//	1 = key   (string, wire 2)
//	2 = value (string, wire 2)
const (
	pbWireVarint   = uint64(0)
	pbWire64bit    = uint64(1)
	pbWireLenDelim = uint64(2)
	pbWire32bit    = uint64(5)

	onnxMetaPropsField = uint64(14)
)

type pbScanner struct{ r *bufio.Reader }

func newPBScanner(r *bufio.Reader) *pbScanner { return &pbScanner{r: r} }

func (s *pbScanner) readVarint() (uint64, error) {
	var v uint64
	for shift := uint(0); ; shift += 7 {
		b, err := s.r.ReadByte()
		if err != nil {
			return 0, err
		}
		v |= uint64(b&0x7f) << shift
		if b&0x80 == 0 {
			return v, nil
		}
		if shift >= 63 {
			return 0, fmt.Errorf("protobuf varint overflow")
		}
	}
}

func (s *pbScanner) readBytes() ([]byte, error) {
	n, err := s.readVarint()
	if err != nil {
		return nil, err
	}
	buf := make([]byte, n)
	_, err = io.ReadFull(s.r, buf)
	return buf, err
}

// skipField discards the value for the given wire type without allocating.
func (s *pbScanner) skipField(wireType uint64) error {
	switch wireType {
	case pbWireVarint:
		_, err := s.readVarint()
		return err
	case pbWire64bit:
		var buf [8]byte
		_, err := io.ReadFull(s.r, buf[:])
		return err
	case pbWireLenDelim:
		n, err := s.readVarint()
		if err != nil {
			return err
		}
		_, err = io.CopyN(io.Discard, s.r, int64(n))
		return err
	case pbWire32bit:
		var buf [4]byte
		_, err := io.ReadFull(s.r, buf[:])
		return err
	default:
		return fmt.Errorf("protobuf: unknown wire type %d", wireType)
	}
}

// scanForMetadata iterates the top-level ModelProto fields, decoding each
// metadata_props entry and returning the value whose key matches targetKey.
func (s *pbScanner) scanForMetadata(targetKey string) (string, error) {
	for {
		tag, err := s.readVarint()
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", err
		}

		fieldNum := tag >> 3
		wireType := tag & 0x7

		if fieldNum == onnxMetaPropsField && wireType == pbWireLenDelim {
			data, err := s.readBytes()
			if err != nil {
				return "", err
			}
			k, v, err := parseStringStringEntry(data)
			if err != nil {
				continue // malformed entry — skip
			}
			if k == targetKey {
				return v, nil
			}
		} else {
			if err := s.skipField(wireType); err != nil {
				return "", err
			}
		}
	}
	return "", fmt.Errorf("ONNX metadata key %q not found in %T", targetKey, s.r)
}

// parseStringStringEntry decodes a serialised StringStringEntryProto.
// Both fields (key=1, value=2) are length-delimited strings.
func parseStringStringEntry(data []byte) (key, value string, err error) {
	s := newPBScanner(bufio.NewReader(bytes.NewReader(data)))
	for {
		tag, err := s.readVarint()
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", "", err
		}
		fieldNum := tag >> 3
		wireType := tag & 0x7

		if wireType == pbWireLenDelim && (fieldNum == 1 || fieldNum == 2) {
			b, err := s.readBytes()
			if err != nil {
				return "", "", err
			}
			if fieldNum == 1 {
				key = string(b)
			} else {
				value = string(b)
			}
		} else {
			if err := s.skipField(wireType); err != nil {
				return "", "", err
			}
		}
	}
	return key, value, nil
}
