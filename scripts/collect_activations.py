"""SAELens activation-cache wrapper for Modal runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_MODEL_CLASS_NAME = "AutoModelForCausalLM"
DEFAULT_HOOK_NAME = "model.layers.20"
DEFAULT_D_IN = 2560


def load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path) as file:
        cfg = yaml.safe_load(file)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return cfg


def _normalize_cache_config(cfg_dict: dict[str, Any]) -> dict[str, Any]:
    import torch

    normalized = dict(cfg_dict)
    normalized.setdefault("model_class_name", DEFAULT_MODEL_CLASS_NAME)
    normalized.setdefault("hook_name", DEFAULT_HOOK_NAME)
    normalized.setdefault("d_in", DEFAULT_D_IN)

    if "store_batch_size_prompts" in normalized:
        normalized.setdefault("model_batch_size", normalized.pop("store_batch_size_prompts"))

    # Not part of current CacheActivationsRunnerConfig, but useful to keep in YAML
    # beside the hook for human readability and future training configs.
    normalized.pop("hook_layer", None)
    normalized.pop("is_dataset_tokenized", None)

    model_kwargs = dict(normalized.get("model_from_pretrained_kwargs") or {})
    for dtype_key in ("torch_dtype", "dtype"):
        dtype_value = model_kwargs.get(dtype_key)
        if isinstance(dtype_value, str) and hasattr(torch, dtype_value):
            model_kwargs[dtype_key] = getattr(torch, dtype_value)
    normalized["model_from_pretrained_kwargs"] = model_kwargs

    return normalized


def build_cache_config(cfg_dict: dict[str, Any]):
    from sae_lens import CacheActivationsRunnerConfig

    cfg = CacheActivationsRunnerConfig(**_normalize_cache_config(cfg_dict))
    if cfg.n_tokens_in_buffer <= 0:
        raise ValueError(
            "CacheActivationsRunnerConfig produced n_tokens_in_buffer=0. "
            "Increase buffer_size_gb or reduce context_size/d_in."
        )
    return cfg


def get_cache_output_path(config_path: str | Path) -> Path:
    cfg_dict = load_config(config_path)
    output_path = cfg_dict.get("new_cached_activations_path")
    if output_path is None:
        raise ValueError(f"{config_path} must set new_cached_activations_path")
    return Path(output_path)


def run_collect(config_path: str | Path):
    from datasets import load_from_disk
    from sae_lens import CacheActivationsRunner

    cfg_dict = load_config(config_path)
    dataset_path = cfg_dict["dataset_path"]
    tokenized_dataset = load_from_disk(dataset_path)
    cfg = build_cache_config(cfg_dict)
    return CacheActivationsRunner(cfg, override_dataset=tokenized_dataset).run()
