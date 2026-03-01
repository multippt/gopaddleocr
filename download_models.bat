@echo off
setlocal
REM The library is compatible with pre-converted models from the RapidOCR project.
REM Repository: https://www.modelscope.cn/models/RapidAI/RapidOCR/tree/master/onnx/

if not exist "models" mkdir models
cd models

if not exist "ch_PP-OCRv5_rec_server_infer.onnx" (
  echo Downloading ch_PP-OCRv5_rec_server_infer.onnx
  curl -o "ch_PP-OCRv5_rec_server_infer.onnx" "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/master/onnx/PP-OCRv5/rec/ch_PP-OCRv5_rec_server_infer.onnx"
)

if not exist "ch_PP-OCRv5_server_det.onnx" (
  echo Downloading ch_PP-OCRv5_server_det.onnx
  curl -o "ch_PP-OCRv5_server_det.onnx" "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/master/onnx/PP-OCRv5/det/ch_PP-OCRv5_server_det.onnx"
)

if not exist "ch_ppocr_mobile_v2.0_cls_infer.onnx" (
  echo Downloading ch_ppocr_mobile_v2.0_cls_infer.onnx
  curl -o "ch_ppocr_mobile_v2.0_cls_infer.onnx" "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/master/onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_infer.onnx"
)

cd ..
echo Done.
exit /b 0
