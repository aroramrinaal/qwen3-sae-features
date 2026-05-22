"""Config-driven Modal entrypoint for Qwen3 SAE experiments.

Run examples:

    modal run --detach modal_app.py --config config/tokenize_1m.yaml
    modal run --detach modal_app.py --config config/cache_1m.yaml
    modal run modal_app.py --config config/inspect_1m_activations.yaml --wait

Set the GPU for GPU-backed jobs with MODAL_GPU, for example:

    MODAL_GPU=H100 modal run --detach modal_app.py --config config/train_sae_1m.yaml
"""

from __future__ import annotations

import argparse
import os
import pprint
import shutil
from enum import StrEnum
from pathlib import Path
from typing import Any

import modal

from scripts.collect_activations import get_cache_output_path, run_collect
from scripts.feature_dashboard import run_feature_dashboard
from scripts.inspect_activations import inspect_cached_activations
from scripts.inspect_sae import inspect_sae as inspect_sae_artifact
from scripts.prepare_dataset import run_prepare
from scripts.train_sae import run_train
from scripts.weights import VOLUME_NAME, VOLUME_ROOT

APP_NAME = "qwen3-sae-features"
REMOTE_CONFIG_ROOT = Path("/root/config")
DEFAULT_GPU = "H100"
GPU_REQUEST = os.getenv("MODAL_GPU", DEFAULT_GPU)

app = modal.App(APP_NAME)
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
        "wandb",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_dir("config", remote_path=str(REMOTE_CONFIG_ROOT))
    .add_local_dir("scripts", remote_path="/root/scripts")
)


class JobKind(StrEnum):
    TOKENIZE = "tokenize"
    CACHE_ACTIVATIONS = "cache_activations"
    INSPECT_ACTIVATIONS = "inspect_activations"
    TRAIN_SAE = "train_sae"
    INSPECT_SAE = "inspect_sae"
    FEATURE_DASHBOARD = "feature_dashboard"


def _load_local_config(config_path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError:
        return _load_top_level_yaml_keys(config_path)

    with config_path.open() as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")
    return config


def _load_top_level_yaml_keys(config_path: Path) -> dict[str, Any]:
    """Read enough YAML locally to choose the Modal function.

    The full config is parsed remotely with PyYAML. Keeping this local fallback
    small avoids requiring PyYAML in the client venv just to dispatch a job.
    """

    config: dict[str, Any] = {}
    for line in config_path.read_text().splitlines():
        if not line or line.startswith((" ", "\t", "#")) or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if not key:
            continue
        value = value.split("#", 1)[0].strip()
        config[key] = value.strip("'\"") if value else None

    if not config:
        raise ValueError(f"Expected a YAML mapping in {config_path}")
    return config


def _resolve_local_config_path(config: str) -> Path:
    config_path = Path(config)
    if config_path.is_absolute():
        return config_path

    local_path = Path.cwd() / config_path
    if local_path.exists() or (config_path.parts and config_path.parts[0] == "config"):
        return local_path

    return Path.cwd() / "config" / config_path


def _remote_config_path(config: str) -> str:
    path = Path(config)
    if path.is_absolute():
        return str(path)

    if path.parts and path.parts[0] == "config":
        return str(Path("/root") / path)

    return str(REMOTE_CONFIG_ROOT / path)


def _infer_job_kind(config: dict[str, Any]) -> JobKind:
    if kind := config.get("job"):
        try:
            return JobKind(str(kind))
        except ValueError as exc:
            allowed = ", ".join(job.value for job in JobKind)
            raise ValueError(f"Unknown job={kind!r}. Expected one of: {allowed}") from exc

    if "tokenizer_name" in config and "save_path" in config:
        return JobKind.TOKENIZE
    if "new_cached_activations_path" in config:
        return JobKind.CACHE_ACTIVATIONS
    if "sae_output_path" in config:
        return JobKind.TRAIN_SAE
    if "output_path" in config and "activation_path" in config and "sae_path" in config:
        return JobKind.FEATURE_DASHBOARD
    if "activation_path" in config:
        return JobKind.INSPECT_ACTIVATIONS
    if "sae_path" in config or "load_path" in config:
        return JobKind.INSPECT_SAE

    raise ValueError(
        "Could not infer Modal job kind from config. Add a 'job' field with one "
        f"of: {', '.join(job.value for job in JobKind)}"
    )


def _run_or_spawn(function: modal.Function, config_path: str, wait: bool) -> Any:
    if wait:
        return function.remote(config_path)

    call = function.spawn(config_path)
    return {"spawned": True, "function_call_id": call.object_id}


def _print_result(result: Any) -> None:
    if isinstance(result, dict):
        pprint.pp(result)
    else:
        print(result)


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={str(VOLUME_ROOT): volume},
)
def prepare_dataset_on_volume(config_path: str) -> dict[str, Any]:
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
def cache_activations_on_volume(config_path: str) -> dict[str, Any]:
    volume.reload()
    output_path = get_cache_output_path(config_path)
    activations_root = VOLUME_ROOT / "activations"
    if not output_path.is_absolute():
        raise ValueError("new_cached_activations_path must be an absolute /vol path.")
    if not output_path.is_relative_to(activations_root):
        raise ValueError(f"Refusing to overwrite outside {activations_root}.")

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
    cpu=2,
    memory=8192,
    timeout=60 * 10,
    volumes={str(VOLUME_ROOT): volume},
)
def inspect_cached_activations_on_volume(config_path: str) -> dict[str, Any]:
    volume.reload()
    return inspect_cached_activations(config_path)


@app.function(
    image=image,
    gpu=GPU_REQUEST,
    cpu=8,
    memory=65536,
    timeout=60 * 60 * 12,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
    volumes={str(VOLUME_ROOT): volume},
)
def train_sae_on_volume(config_path: str) -> dict[str, Any]:
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
def inspect_sae_on_volume(config_path: str) -> dict[str, Any]:
    volume.reload()
    return inspect_sae_artifact(config_path)


@app.function(
    image=image,
    gpu=GPU_REQUEST,
    cpu=8,
    memory=65536,
    timeout=60 * 60 * 8,
    volumes={str(VOLUME_ROOT): volume},
)
def feature_dashboard_on_volume(config_path: str) -> dict[str, Any]:
    volume.reload()
    result = run_feature_dashboard(config_path)
    volume.commit()
    return result


def _dispatch_config(config: str, wait: bool) -> dict[str, Any]:
    local_path = _resolve_local_config_path(config)
    local_config = _load_local_config(local_path)
    job_kind = _infer_job_kind(local_config)
    remote_path = _remote_config_path(config)

    if job_kind == JobKind.TOKENIZE:
        result = _run_or_spawn(prepare_dataset_on_volume, remote_path, wait)
    elif job_kind == JobKind.CACHE_ACTIVATIONS:
        result = _run_or_spawn(cache_activations_on_volume, remote_path, wait)
    elif job_kind == JobKind.INSPECT_ACTIVATIONS:
        result = inspect_cached_activations_on_volume.remote(remote_path)
    elif job_kind == JobKind.TRAIN_SAE:
        result = _run_or_spawn(train_sae_on_volume, remote_path, wait)
    elif job_kind == JobKind.INSPECT_SAE:
        result = inspect_sae_on_volume.remote(remote_path)
    elif job_kind == JobKind.FEATURE_DASHBOARD:
        result = _run_or_spawn(feature_dashboard_on_volume, remote_path, wait)
    else:
        raise AssertionError(f"Unhandled job kind: {job_kind}")

    return {
        "app": APP_NAME,
        "job": job_kind.value,
        "local_config_path": str(local_path),
        "remote_config_path": remote_path,
        "gpu": GPU_REQUEST
        if job_kind
        in {JobKind.CACHE_ACTIVATIONS, JobKind.TRAIN_SAE, JobKind.FEATURE_DASHBOARD}
        else None,
        "result": result,
    }


@app.local_entrypoint()
def main(*argv: str) -> None:
    parser = argparse.ArgumentParser(
        description="Run a qwen3-sae-features Modal job from a YAML config."
    )
    parser.add_argument("config_arg", nargs="?", help="YAML config path, e.g. config/cache_1m.yaml")
    parser.add_argument("--config", dest="config_opt", help="YAML config path")
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Block for tokenization/cache/training results instead of spawning a background call.",
    )
    args = parser.parse_args(list(argv))

    config = args.config_opt or args.config_arg
    if config is None:
        parser.error("provide a config path, either as a positional argument or with --config")

    _print_result(_dispatch_config(config=config, wait=args.wait))
