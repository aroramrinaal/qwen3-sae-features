"""Utilities for storing Qwen3 model files in a Modal Volume."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

MODEL_ID = "Qwen/Qwen3-4B-Base"
VOLUME_NAME = "qwen3-sae-features"
VOLUME_ROOT = Path("/vol")
MODEL_SUBDIR = Path("models") / "Qwen3-4B-Base"
MODEL_DIR = VOLUME_ROOT / MODEL_SUBDIR

MODEL_FILE_PATTERNS = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "*.safetensors",
    "model.safetensors.index.json",
]


def clean_model_dir(model_dir: Path = MODEL_DIR) -> None:
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)


def save_model_snapshot(model_dir: Path = MODEL_DIR) -> dict:
    token = os.getenv("HF_TOKEN")
    if token is None:
        raise RuntimeError("HF_TOKEN is not set. Check the 'huggingface-secret' secret.")

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    from huggingface_hub import snapshot_download

    clean_model_dir(model_dir)

    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(model_dir),
        token=token,
        allow_patterns=MODEL_FILE_PATTERNS,
    )

    shutil.rmtree(model_dir / ".cache", ignore_errors=True)

    saved_files = sorted(
        str(path.relative_to(model_dir))
        for path in model_dir.rglob("*")
        if path.is_file()
    )
    return {
        "model_id": MODEL_ID,
        "volume_name": VOLUME_NAME,
        "saved_path": str(model_dir),
        "saved_files": saved_files,
    }
