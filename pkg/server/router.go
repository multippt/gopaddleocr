package server

import (
	"log"
	"os"

	"github.com/gin-gonic/gin"
	"github.com/multippt/gopaddleocr/pkg/ocr"
	ort "github.com/yalue/onnxruntime_go"
)

func getEnv(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

type Server struct {
	server    *gin.Engine
	ocrEngine *ocr.PaddleOCREngine
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
	ortPath := getEnv("ORT_LIB_PATH", "./onnxruntime/lib/onnxruntime.dll")
	ort.SetSharedLibraryPath(ortPath)
	if err := ort.InitializeEnvironment(); err != nil {
		log.Fatalf("init ORT: %v", err)
	}
	defer ort.DestroyEnvironment()

	// Build and preload the OCR engine in the background.
	ocrEngine := ocr.NewPaddleOCREngine()
	go func() {
		if err := ocrEngine.Load(); err != nil {
			log.Printf("engine preload error: %v", err)
		} else {
			log.Println("engine loaded successfully")
		}
	}()
	s.ocrEngine = ocrEngine

	s.server = gin.Default()
	s.registerHandlers()

	log.Printf("OCR service listening on %s", listenAddr)
	if err := s.server.Run(listenAddr); err != nil {
		log.Fatalf("server error: %v", err)
	}
}
