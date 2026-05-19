"""Modal remote entrypoint for Qwen3 inference on an NVIDIA H100."""

from __future__ import annotations

import os
import pprint
import shutil
import sys
from pathlib import Path

import modal

from scripts.activations import capture_activation_metadata
from scripts.collect_activations import get_cache_output_path, run_collect
from scripts.inspect_activations import inspect_cached_activations
from scripts.inspect_sae import inspect_sae
from scripts.infer import (
    InferConfig,
    build_model,
    build_tokenizer,
    format_inference_output,
    generate_text,
)
from scripts.prepare_dataset import run_prepare
from scripts.train_sae import run_train
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
        "datasets",
        "sae-lens",
        "pyyaml",
    )
    .add_local_dir("config", remote_path="/root/config")
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


def _remote_config_path(config_path: str) -> str:
    path = Path(config_path)
    if path.is_absolute():
        return str(path)

    parts = path.parts
    if parts and parts[0] == "config":
        return str(Path("/root") / path)

    return str(Path("/root/config") / path)


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
    gpu=GPU_REQUEST,
    volumes={str(VOLUME_ROOT): volume},
    timeout=60 * 20,
)
def inspect_qwen_module_names(limit: int = 40, hook_name: str = "model.layers.20") -> dict:
    volume.reload()
    model = build_model()
    module_names = [name for name, _module in model.named_modules()]
    return {
        "requested_hook": hook_name,
        "hook_exists": hook_name in module_names,
        "first_module_names": module_names[:limit],
    }


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


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={str(VOLUME_ROOT): volume},
)
def prepare_dataset_on_volume(config_path: str) -> dict:
    volume.reload()
    dataset = run_prepare(config_path)
    volume.commit()
    return {
        "config_path": config_path,
        "num_rows": getattr(dataset, "num_rows", None),
        "columns": list(getattr(dataset, "column_names", []) or []),
    }


@app.function(
    image=image,
    gpu=GPU_REQUEST,
    cpu=8,
    memory=65536,
    timeout=60 * 60 * 6,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={str(VOLUME_ROOT): volume},
)
def cache_activations_on_volume(config_path: str) -> dict:
    volume.reload()
    output_path = get_cache_output_path(config_path)
    if not output_path.is_absolute():
        raise ValueError("new_cached_activations_path must be an absolute /vol path.")
    if not output_path.is_relative_to(VOLUME_ROOT / "activations"):
        raise ValueError(f"Refusing to overwrite outside {VOLUME_ROOT / 'activations'}.")

    if output_path.exists():
        shutil.rmtree(output_path)
        volume.commit()

    dataset = run_collect(config_path)
    volume.commit()
    return {
        "config_path": config_path,
        "cached_activations_path": str(output_path),
        "num_rows": getattr(dataset, "num_rows", None),
        "columns": list(getattr(dataset, "column_names", []) or []),
    }


@app.function(
    image=image,
    cpu=1,
    memory=1024,
    timeout=60 * 5,
    volumes={str(VOLUME_ROOT): volume},
)
def delete_activation_cache_path(cache_path: str) -> dict:
    volume.reload()
    target_path = Path(cache_path)
    activations_root = VOLUME_ROOT / "activations"

    if not target_path.is_absolute():
        raise ValueError("cache_path must be an absolute /vol path.")
    if not target_path.is_relative_to(activations_root):
        raise ValueError(f"Refusing to delete outside {activations_root}.")

    existed = target_path.exists()
    if existed:
        shutil.rmtree(target_path)
        volume.commit()

    return {
        "deleted": existed,
        "path": str(target_path),
    }


@app.function(
    image=image,
    cpu=2,
    memory=8192,
    timeout=60 * 10,
    volumes={str(VOLUME_ROOT): volume},
)
def inspect_cached_activations_on_volume(config_path: str) -> dict:
    volume.reload()
    return inspect_cached_activations(config_path)


@app.function(
    image=image,
    gpu=GPU_REQUEST,
    cpu=8,
    memory=65536,
    timeout=60 * 60 * 3,
    volumes={str(VOLUME_ROOT): volume},
)
def train_sae_on_volume(config_path: str) -> dict:
    volume.reload()
    result = run_train(config_path)
    volume.commit()
    return result


@app.function(
    image=image,
    cpu=2,
    memory=8192,
    timeout=60 * 10,
    volumes={str(VOLUME_ROOT): volume},
)
def inspect_sae_on_volume(config_path: str) -> dict:
    volume.reload()
    return inspect_sae(config_path)


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
def inspect_modules(gpu: str = "H100", limit: int = 40, hook_name: str = "model.layers.20"):
    result = inspect_qwen_module_names.remote(limit=limit, hook_name=hook_name)
    print(f"requested_hook: {result['requested_hook']}")
    print(f"hook_exists: {result['hook_exists']}")
    print("first_module_names:")
    for module_name in result["first_module_names"]:
        print(module_name)


@app.local_entrypoint()
def save_weights():
    result = save_model_weights_to_volume.remote()
    print("Saved model files to Modal Volume")
    print(result)


@app.local_entrypoint()
def smoke_tokenize():
    call = prepare_dataset_on_volume.spawn(_remote_config_path("smoke_tokenize.yaml"))
    print(f"Spawned smoke tokenization: {call.object_id}")


@app.local_entrypoint()
def smoke_cache_activations(gpu: str = "H100"):
    call = cache_activations_on_volume.spawn(_remote_config_path("smoke_cache.yaml"))
    print(f"Spawned smoke activation caching: {call.object_id}")


@app.local_entrypoint()
def run_tokenize(config: str = "config/tokenize_1m.yaml"):
    call = prepare_dataset_on_volume.spawn(_remote_config_path(config))
    print(f"Spawned tokenization: {call.object_id}")


@app.local_entrypoint()
def run_cache_activations(config: str = "config/cache_1m.yaml", gpu: str = "H100"):
    call = cache_activations_on_volume.spawn(_remote_config_path(config))
    print(f"Spawned activation caching: {call.object_id}")


@app.local_entrypoint()
def clear_activation_cache(path: str):
    result = delete_activation_cache_path.remote(path)
    print(result)


@app.local_entrypoint()
def inspect_smoke_activations():
    result = inspect_cached_activations_on_volume.remote(
        _remote_config_path("inspect_smoke_activations.yaml")
    )
    pprint.pp(result)


@app.local_entrypoint()
def inspect_activations(config: str = "config/inspect_smoke_activations.yaml"):
    result = inspect_cached_activations_on_volume.remote(_remote_config_path(config))
    pprint.pp(result)


@app.local_entrypoint()
def inspect_1m_activations():
    result = inspect_cached_activations_on_volume.remote(
        _remote_config_path("inspect_1m_activations.yaml")
    )
    pprint.pp(result)


@app.local_entrypoint()
def train_smoke_sae(gpu: str = "H100"):
    call = train_sae_on_volume.spawn(_remote_config_path("train_sae_smoke.yaml"))
    print(f"Spawned smoke SAE training: {call.object_id}")


@app.local_entrypoint()
def train_sae(config: str = "config/train_sae_smoke.yaml", gpu: str = "H100"):
    call = train_sae_on_volume.spawn(_remote_config_path(config))
    print(f"Spawned SAE training: {call.object_id}")


@app.local_entrypoint()
def train_sae_debug(config: str = "config/train_sae_smoke.yaml", gpu: str = "H100"):
    result = train_sae_on_volume.remote(_remote_config_path(config))
    pprint.pp(result)


@app.local_entrypoint()
def train_1m_sae(gpu: str = "H100"):
    call = train_sae_on_volume.spawn(_remote_config_path("train_sae_1m.yaml"))
    print(f"Spawned 1M SAE training: {call.object_id}")


@app.local_entrypoint()
def train_smoke_sae_debug(gpu: str = "H100"):
    result = train_sae_on_volume.remote(_remote_config_path("train_sae_smoke.yaml"))
    pprint.pp(result)


@app.local_entrypoint()
def inspect_smoke_sae():
    result = inspect_sae_on_volume.remote(_remote_config_path("inspect_sae_smoke.yaml"))
    pprint.pp(result)


@app.local_entrypoint()
def inspect_sae(config: str = "config/inspect_sae_smoke.yaml"):
    result = inspect_sae_on_volume.remote(_remote_config_path(config))
    pprint.pp(result)


@app.local_entrypoint()
def inspect_1m_sae():
    result = inspect_sae_on_volume.remote(_remote_config_path("inspect_sae_1m.yaml"))
    pprint.pp(result)
