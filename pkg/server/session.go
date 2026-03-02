package server

import (
	"fmt"
	"image"
	"sync"

	"github.com/google/uuid"
)

// sessions holds in-memory session data: session_id → image.Image
var sessions sync.Map

func storeSession(id string, img image.Image) {
	sessions.Store(id, img)
}

func deleteSession(id string) {
	sessions.Delete(id)
}

func getSession(id string) (image.Image, error) {
	v, ok := sessions.Load(id)
	if !ok {
		return nil, fmt.Errorf("session %s not found", id)
	}
	return v.(image.Image), nil
}

func newSessionID() string {
	return uuid.New().String()
}
