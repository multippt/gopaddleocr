package main

import (
	"flag"

	"github.com/multippt/gopaddleocr/pkg/server"
)

func main() {
	var listen string
	flag.StringVar(&listen, "listen", ":8051", "server listening address (e.g. :8051 or 0.0.0.0:8051)")
	flag.StringVar(&listen, "l", ":8051", "alias for -listen")
	flag.Parse()

	s := server.NewServer()
	s.Start(listen)
}
