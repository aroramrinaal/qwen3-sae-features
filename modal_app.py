"""Modal remote entrypoint for Qwen3 inference on an NVIDIA H100."""

from __future__ import annotations

import os
import sys

import modal

from scripts.activations import capture_activation_metadata
from scripts.infer import (
    InferConfig,
    build_model,
    build_tokenizer,
    format_inference_output,
    generate_text,
)
from scripts.weights import MODEL_DIR, VOLUME_NAME, VOLUME_ROOT, save_model_snapshot

app = modal.App("qwen3-sae-features")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "safetensors",
        "huggingface_hub[hf_transfer]",
    )
    .add_local_dir("scripts", remote_path="/root/scripts")
)


def _gpu_request_from_command() -> str | None:
    if gpu := os.getenv("MODAL_GPU"):
        return gpu

    for idx, arg in enumerate(sys.argv):
        if arg == "--gpu" and idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
        if arg.startswith("--gpu="):
            return arg.split("=", 1)[1]

    return None


GPU_REQUEST = _gpu_request_from_command()


@app.function(
    image=image,
    gpu=GPU_REQUEST,
    volumes={str(VOLUME_ROOT): volume},
    timeout=60 * 20,
)
def run_inference(prompt: str, max_new_tokens: int = 128) -> str:
    volume.reload()
    tokenizer = build_tokenizer()
    model = build_model()
    config = InferConfig(max_new_tokens=max_new_tokens)
    return generate_text(prompt=prompt, tokenizer=tokenizer, model=model, config=config)


@app.function(
    image=image,
    gpu=GPU_REQUEST,
    volumes={str(VOLUME_ROOT): volume},
    timeout=60 * 20,
)
def capture_layer_activations(prompt: str = "The capital of France is") -> dict:
    volume.reload()
    return capture_activation_metadata(prompt=prompt)


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={str(VOLUME_ROOT): volume},
)
def save_model_weights_to_volume() -> dict:
    result = save_model_snapshot(MODEL_DIR)
    volume.commit()
    return result


@app.local_entrypoint()
def main(gpu: str, prompt: str = "The capital of France is"):
    completion = run_inference.remote(prompt=prompt)
    print(format_inference_output(prompt=prompt, completion=completion))


@app.local_entrypoint()
def smoke_test_activations(gpu: str, prompt: str = "The capital of France is"):
    result = capture_layer_activations.remote(prompt=prompt)
    print(f"layer: {result['layer']}")
    print(f"shape: {result['shape']}")
    print(f"dtype: {result['dtype']}")
    print(f"device: {result['device']}")
    print(f"token_count: {result['token_count']}")


@app.local_entrypoint()
def save_weights():
    result = save_model_weights_to_volume.remote()
    print("Saved model files to Modal Volume")
    print(result)
