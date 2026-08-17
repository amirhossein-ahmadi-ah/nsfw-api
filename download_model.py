#!/usr/bin/env python3
"""
Download the ONNX model from Hugging Face Hub into ./models/.

Usage:
    python download_model.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "OwenElliott/image-safety-classifier-l"
FILENAME = "onnx/image-safety-classifier-l.onnx"

PROJECT_ROOT = Path(__file__).resolve().parent
DEST_DIR = PROJECT_ROOT / "models"
DEST_PATH = DEST_DIR / "image-safety-classifier-l.onnx"


def main() -> int:
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {FILENAME} from {REPO_ID} ...")
    downloaded_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

    shutil.copyfile(downloaded_path, DEST_PATH)
    size_mb = DEST_PATH.stat().st_size / (1024 * 1024)
    print(f"Model saved to {DEST_PATH} ({size_mb:.1f} MB)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
