package server

import (
	"log"

	"github.com/gin-gonic/gin"
	"github.com/multippt/gopaddleocr/pkg/ocr"
)

type Server struct {
	server    *gin.Engine
	ocrEngine *ocr.Engine
}

func NewServer() *Server {
	return &Server{}
}

func (s *Server) registerHandlers() {
	s.server.GET("/health", s.handleHealth)
	s.server.POST("/ocr", s.handleOCR)
	s.server.POST("/detect", s.handleDetect)
	s.server.POST("/session/create", s.handleSessionCreate)
	s.server.POST("/session/detect", s.handleSessionDetect)
	s.server.POST("/session/ocr", s.handleSessionOCR)
	s.server.POST("/session/delete", s.handleSessionDelete)
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

	s.server = gin.Default()
	_ = s.server.SetTrustedProxies(nil)
	s.registerHandlers()

	log.Printf("OCR service listening on %s", listenAddr)
	if err := s.server.Run(listenAddr); err != nil {
		log.Fatalf("server error: %v", err)
	}
}
