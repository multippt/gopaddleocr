#!/usr/bin/env bash
# Download and extract ONNX Runtime from https://github.com/microsoft/onnxruntime/releases
# Compatible with Mac (Apple Silicon) and Linux (x64, aarch64). On Linux, uses GPU build if nvidia-smi is present.

set -e

ORT_VERSION="${ORT_VERSION:-1.24.2}"
BASE_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}"
INSTALL_DIR="onnxruntime"

if [ -d "$INSTALL_DIR" ]; then
  echo "Directory $INSTALL_DIR already exists; skipping download."
  exit 0
fi

OS=$(uname -s)
ARCH=$(uname -m)

# Choose asset and optional version override (e.g. macOS x86_64 needs older release)
ASSET=""
case "$OS" in
  Darwin)
    case "$ARCH" in
      arm64)
        ASSET="onnxruntime-osx-arm64-${ORT_VERSION}.tgz"
        ;;
      x86_64)
        # x86_64 macOS binaries dropped in 1.24.x; use last version that provided them
        ORT_VERSION="1.23.2"
        BASE_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}"
        ASSET="onnxruntime-osx-x64-${ORT_VERSION}.tgz"
        ;;
      *)
        echo "Unsupported macOS architecture: $ARCH"
        exit 1
        ;;
    esac
    ;;
  Linux)
    USE_GPU=""
    if [ "$ARCH" = "x86_64" ] && command -v nvidia-smi &>/dev/null; then
      USE_GPU="-gpu"
    fi
    case "$ARCH" in
      x86_64)
        ASSET="onnxruntime-linux-x64${USE_GPU}-${ORT_VERSION}.tgz"
        ;;
      aarch64|arm64)
        ASSET="onnxruntime-linux-aarch64-${ORT_VERSION}.tgz"
        ;;
      *)
        echo "Unsupported Linux architecture: $ARCH"
        exit 1
        ;;
    esac
    ;;
  *)
    echo "Unsupported OS: $OS"
    exit 1
    ;;
esac

URL="${BASE_URL}/${ASSET}"
echo "Downloading ONNX Runtime v${ORT_VERSION}: $ASSET"
if ! curl -fSL -o "$ASSET" "$URL"; then
  echo "Download failed. Check version and URL: $URL"
  exit 1
fi

echo "Extracting..."
tar xzf "$ASSET"
rm -f "$ASSET"

# Extracted folder name is asset name without .tgz
EXTRACTED="${ASSET%.tgz}"
if [ -d "$EXTRACTED" ]; then
  mv "$EXTRACTED" "$INSTALL_DIR"
  echo "ONNX Runtime installed under $INSTALL_DIR/"
else
  echo "Unexpected archive layout: expected directory $EXTRACTED"
  exit 1
fi
