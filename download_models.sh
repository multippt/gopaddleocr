# The library is compatible with pre-converted models from the RapidOCR project.
# Repository: https://www.modelscope.cn/models/RapidAI/RapidOCR/tree/master/onnx/

mkdir -p models

FILES=(
  "ch_PP-OCRv5_rec_server_infer.onnx"
  "ch_PP-OCRv5_server_det.onnx"
  "PP-LCNet_x1_0_textline_ori_infer.onnx"
  "PP-DocLayoutV3.onnx"
)

URLS=(
  "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/master/onnx/PP-OCRv5/rec/ch_PP-OCRv5_rec_server_infer.onnx"
  "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/master/onnx/PP-OCRv5/det/ch_PP-OCRv5_server_det.onnx"
  "https://huggingface.co/marsena/paddleocr-onnx-models/resolve/main/PP-LCNet_x1_0_textline_ori_infer.onnx"
  "https://huggingface.co/alex-dinh/PP-DocLayoutV3-ONNX/resolve/main/PP-DocLayoutV3.onnx"
)

for i in "${!FILES[@]}"; do
  if [ ! -f "models/${FILES[$i]}" ]; then
    curl -Lo "models/${FILES[$i]}" "${URLS[$i]}"
  fi
done
