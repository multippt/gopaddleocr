# The library is compatible with pre-converted models from the RapidOCR project.
# Repository: https://www.modelscope.cn/models/RapidAI/RapidOCR/tree/master/onnx/

mkdir -p models

declare -A MODELS=(
  ["ch_PP-OCRv5_rec_server_infer.onnx"]="https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/master/onnx/PP-OCRv5/rec/ch_PP-OCRv5_rec_server_infer.onnx"
  ["ch_PP-OCRv5_server_det.onnx"]="https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/master/onnx/PP-OCRv5/det/ch_PP-OCRv5_server_det.onnx"
  ["ch_ppocr_mobile_v2.0_cls_infer.onnx"]="https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/master/onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_infer.onnx"
  ["PP-DocLayoutV3.onnx"]="https://huggingface.co/alex-dinh/PP-DocLayoutV3-ONNX/resolve/main/PP-DocLayoutV3.onnx"
)

(
cd models
for file in "${!MODELS[@]}"; do
  if [ ! -f "$file" ]; then
    curl -Lo "$file" "${MODELS[$file]}"
  fi
done
)
