"""
Cross-platform PP-OCRv5 ONNX export pipeline.

Invoked by export.bat / export.sh after venv activation.
Steps:
  1. Download models (skips already-cached)
  2. Convert each Paddle model to ONNX via paddle2onnx
  3. Embed character dict into the rec model's ONNX metadata
"""

import json
import subprocess
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("ERROR: huggingface_hub is not installed. Run: pip install huggingface_hub")
    sys.exit(1)

try:
    import onnx
except ImportError:
    print("ERROR: onnx is not installed. Run: pip install onnx")
    sys.exit(1)

OUTPUT_DIR = Path("../models")
CACHE_DIR = Path(".cache")

OPSET = "11"

MODELS = [
    "PP-OCRv5_server_det",
    "PP-OCRv5_server_rec",
    "PP-LCNet_x1_0_textline_ori",
]

HF_REPOS = [f"PaddlePaddle/{name}" for name in MODELS]

REQUIRED_FILES = ["inference.json", "inference.pdiparams"]


def is_cached(model_dir: Path) -> bool:
    return all((model_dir / f).is_file() for f in REQUIRED_FILES)


def download_models() -> None:
    CACHE_DIR.mkdir(exist_ok=True)

    for repo_id in HF_REPOS:
        model_name = repo_id.split("/")[1]
        local_dir = CACHE_DIR / model_name

        if is_cached(local_dir):
            print(f"[skip]     {model_name}  (already cached)")
            continue

        print(f"[download] {model_name}  ->  {local_dir}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            ignore_patterns=["*.md", ".gitattributes"],
        )
        print(f"[done]     {model_name}")

    print("\nAll models ready in ./cache/")


def convert(model_name: str) -> None:
    output_path = OUTPUT_DIR / f"{model_name}.onnx"
    if output_path.exists():
        print(f"[skip]     {model_name}  (already converted)")
        return

    subprocess.run(
        [
            "paddle2onnx",
            "--model_dir",           str(CACHE_DIR / model_name),
            "--model_filename",      "inference.json",
            "--params_filename",     "inference.pdiparams",
            "--save_file",           str(output_path),
            "--opset_version",       OPSET,
            "--enable_onnx_checker", "True",
        ],
        check=True,
    )


def embed_rec_charset() -> None:
    config_path = CACHE_DIR / "PP-OCRv5_server_rec" / "config.json"
    model_path = OUTPUT_DIR / "PP-OCRv5_server_rec.onnx"

    with open(config_path, encoding="utf-8") as f:
        chars = json.load(f)["PostProcess"]["character_dict"]

    model = onnx.load_model(str(model_path))
    value = "\n".join(chars)
    for entry in model.metadata_props:
        if entry.key == "character":
            if entry.value == value:
                print(f"[embed] character metadata already up to date in {model_path}")
                return
            entry.value = value
            break
    else:
        meta = model.metadata_props.add()
        meta.key = "character"
        meta.value = value
    onnx.save_model(model, str(model_path))
    print(f"[embed] wrote {len(chars)} characters to ONNX metadata of {model_path}")


REC_MODEL = "PP-OCRv5_server_rec"


def main() -> None:
    print("=== PP-OCRv5 ONNX Export ===")
    print()

    OUTPUT_DIR.mkdir(exist_ok=True)

    total = 1 + len(MODELS)

    print(f"[1/{total}] Downloading models...")
    download_models()
    print()

    for i, model_name in enumerate(MODELS, start=2):
        print(f"[{i}/{total}] Converting {model_name}...")
        convert(model_name)
        if model_name == REC_MODEL:
            print(f"  Embedding character dict into {model_name}.onnx...")
            embed_rec_charset()
        print()

    onnx_files = sorted(OUTPUT_DIR.glob("*.onnx"))
    print(f"=== All done. ONNX files written to ./{OUTPUT_DIR}/ ===")
    for f in onnx_files:
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
