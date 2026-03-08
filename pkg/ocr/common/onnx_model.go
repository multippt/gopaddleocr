package common

import (
	"errors"

	ort "github.com/yalue/onnxruntime_go"
)

var ErrInvalidConfig = errors.New("invalid config")

type OnnxModel struct {
	Model

	session *ort.DynamicAdvancedSession
	config  ModelConfig
}

func NewOnnxModel(impl Model) *OnnxModel {
	return &OnnxModel{
		Model: impl,
	}
}

func (m *OnnxModel) GetConfig() ModelConfig { return m.config }

func (m *OnnxModel) Init(configSrc ConfigSource) error {
	cfg := configSrc.GetConfig(m.GetName())
	if cfg == nil {
		cfg = m.GetDefaultConfig()
	}
	m.config = cfg
	session, err := cfg.GetOnnxConfig().GetORTSession()
	if err != nil {
		return err
	}
	m.session = session
	return nil
}

func (m *OnnxModel) GetSession() *ort.DynamicAdvancedSession { return m.session }

func (m *OnnxModel) Close() error {
	if m.session != nil {
		return m.session.Destroy()
	}
	return nil
}
