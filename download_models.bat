@echo off
setlocal
REM The library is compatible with pre-converted models from the RapidOCR project.
REM Repository: https://www.modelscope.cn/models/RapidAI/RapidOCR/tree/master/onnx/

if not exist "models" mkdir models
cd models

if not exist "ch_PP-OCRv5_rec_server_infer.onnx" (
  curl -Lo "ch_PP-OCRv5_rec_server_infer.onnx" "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/master/onnx/PP-OCRv5/rec/ch_PP-OCRv5_rec_server_infer.onnx"
)

if not exist "ch_PP-OCRv5_server_det.onnx" (
  curl -Lo "ch_PP-OCRv5_server_det.onnx" "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/master/onnx/PP-OCRv5/det/ch_PP-OCRv5_server_det.onnx"
)

if not exist "PP-LCNet_x1_0_textline_ori_infer.onnx" (
  curl -Lo "PP-LCNet_x1_0_textline_ori_infer.onnx" "https://huggingface.co/marsena/paddleocr-onnx-models/resolve/main/PP-LCNet_x1_0_textline_ori_infer.onnx"
)

if not exist "PP-DocLayoutV3.onnx" (
  curl -Lo "PP-DocLayoutV3.onnx" "https://huggingface.co/alex-dinh/PP-DocLayoutV3-ONNX/resolve/main/PP-DocLayoutV3.onnx"
)

cd ..
exit /b 0
