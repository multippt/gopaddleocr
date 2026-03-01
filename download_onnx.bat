@echo off
setlocal EnableDelayedExpansion
REM Download and extract ONNX Runtime from https://github.com/microsoft/onnxruntime/releases
REM Compatible with Windows (x64, ARM64). Uses GPU build if nvidia-smi is present on x64.

if not defined ORT_VERSION set "ORT_VERSION=1.24.2"
set "BASE_URL=https://github.com/microsoft/onnxruntime/releases/download/v%ORT_VERSION%"
set "INSTALL_DIR=onnxruntime"

if exist "%INSTALL_DIR%" (
  echo Directory %INSTALL_DIR% already exists; skipping download.
  exit /b 0
)

REM Detect architecture (AMD64, ARM64)
set "ARCH=%PROCESSOR_ARCHITECTURE%"
if "%ARCH%"=="AMD64" set "ARCH=x64"

set "ASSET="
set "USE_GPU="

REM Check for NVIDIA GPU on x64 (nvidia-smi in PATH)
if "%ARCH%"=="x64" (
  where nvidia-smi >nul 2>&1
  if !errorlevel! equ 0 set "USE_GPU=-gpu"
)

if "%ARCH%"=="x64" (
  set "ASSET=onnxruntime-win-x64!USE_GPU!-%ORT_VERSION%.zip"
) else if "%ARCH%"=="ARM64" (
  set "ASSET=onnxruntime-win-arm64-%ORT_VERSION%.zip"
) else (
  echo Unsupported Windows architecture: %ARCH%
  exit /b 1
)

set "URL=%BASE_URL%/%ASSET%"
echo Downloading ONNX Runtime v%ORT_VERSION%: %ASSET%

curl -fSL -o "%ASSET%" "%URL%"
if errorlevel 1 (
  echo Download failed. Check version and URL: %URL%
  exit /b 1
)

echo Extracting...
powershell -NoProfile -Command "Expand-Archive -Path '%ASSET%' -DestinationPath '.' -Force"

REM Extracted folder name is asset name without .zip
set "EXTRACTED=%ASSET:.zip=%"
if exist "%EXTRACTED%" (
  ren "%EXTRACTED%" "%INSTALL_DIR%"
) else (
  echo Unexpected archive layout: expected directory %EXTRACTED%
  del "%ASSET%" 2>nul
  exit /b 1
)

del "%ASSET%" 2>nul
echo ONNX Runtime installed under %INSTALL_DIR%/
exit /b 0
