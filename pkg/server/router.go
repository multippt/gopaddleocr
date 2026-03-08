package server

import (
	"log"
	"net/http"

	"github.com/multippt/gopaddleocr/pkg/ocr"
)

type Server struct {
	server         *http.ServeMux
	ocrEngine      *ocr.Engine
	sessionManager *SessionManager
}

func NewServer() *Server {
	return &Server{
		sessionManager: NewSessionManager(),
	}
}

func (s *Server) registerHandlers() {
	s.server.HandleFunc("GET /health", s.handleHealth)
	s.server.HandleFunc("POST /ocr", s.handleOCR)
	s.server.HandleFunc("POST /detect", s.handleDetect)
	s.server.HandleFunc("POST /session/create", s.handleSessionCreate)
	s.server.HandleFunc("POST /session/detect", s.handleSessionDetect)
	s.server.HandleFunc("POST /session/ocr", s.handleSessionOCR)
	s.server.HandleFunc("POST /session/delete", s.handleSessionDelete)
}

func (s *Server) Start(listenAddr string) {
	// Build and preload the OCR engine in the background.
	ocrEngine := ocr.NewEngine()
	go func() {
		if err := ocrEngine.Init(); err != nil {
			log.Printf("engine preload error: %v", err)
		} else {
			log.Println("engine loaded successfully")
		}
	}()
	defer func() {
		_ = ocrEngine.Close()
	}()
	s.ocrEngine = ocrEngine

	s.server = http.NewServeMux()
	s.registerHandlers()

	log.Printf("OCR service listening on %s", listenAddr)
	if err := http.ListenAndServe(listenAddr, s.server); err != nil {
		log.Fatalf("server error: %v", err)
	}
}
