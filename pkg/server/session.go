package server

import (
	"fmt"
	"image"
	"sync"

	"github.com/google/uuid"
)

// SessionManager holds in-memory session data: session_id → image.Image
type SessionManager struct {
	sessions sync.Map
}

// NewSessionManager returns a new SessionManager.
func NewSessionManager() *SessionManager {
	return &SessionManager{}
}

// Store saves an image for the given session ID.
func (m *SessionManager) Store(id string, img image.Image) {
	m.sessions.Store(id, img)
}

// Delete removes the session for the given ID.
func (m *SessionManager) Delete(id string) {
	m.sessions.Delete(id)
}

// Get returns the image for the given session ID, or an error if not found.
func (m *SessionManager) Get(id string) (image.Image, error) {
	v, ok := m.sessions.Load(id)
	if !ok {
		return nil, fmt.Errorf("session %s not found", id)
	}
	return v.(image.Image), nil
}

// NewID returns a new unique session ID.
func (m *SessionManager) NewID() string {
	return uuid.New().String()
}
