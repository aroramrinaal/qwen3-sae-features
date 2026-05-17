"""Modal app to download Qwen3 safetensor weights into a persistent Modal Volume."""

from __future__ import annotations

import os
from pathlib import Path

import modal

APP_NAME = "qwen3-sae-features-save-weights"
VOLUME_NAME = "qwen3-sae-features"
MODEL_ID = "Qwen/Qwen3-4B-Base"
VOLUME_ROOT = Path("/vol")
MODEL_DIR = VOLUME_ROOT / "models" / "Qwen3-4B-Base"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "huggingface_hub[hf_transfer]",
    "safetensors",
)


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={str(VOLUME_ROOT): volume},
)
def save_model_weights_to_volume() -> dict:
    from huggingface_hub import snapshot_download

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    token = os.getenv("HF_TOKEN")
    if token is None:
        raise RuntimeError("HF_TOKEN is not set. Check the 'huggingface-secret' secret.")

    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(MODEL_DIR),
        local_dir_use_symlinks=False,
        token=token,
        allow_patterns=[
            "*.safetensors",
            "model.safetensors.index.json",
        ],
    )

    volume.commit()

    saved_files = sorted(p.name for p in MODEL_DIR.glob("*"))
    return {
        "model_id": MODEL_ID,
        "volume_name": VOLUME_NAME,
        "saved_path": str(MODEL_DIR),
        "saved_files": saved_files,
    }


@app.local_entrypoint()
def main():
    result = save_model_weights_to_volume.remote()
    print("Saved model weights to Modal Volume")
    print(result)
