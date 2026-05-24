"""Metadata writing for SAE training outputs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def save_training_metadata(cfg: dict[str, Any], output_dir: Path, result: Any) -> dict[str, Any]:
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_name": cfg["model_name"],
        "model_class_name": cfg.get("model_class_name", "AutoModelForCausalLM"),
        "hook_name": cfg["hook_name"],
        "hook_layer": cfg.get("hook_layer"),
        "d_in": int(cfg["d_in"]),
        "d_sae": int(cfg["d_sae"]),
        "context_size": int(cfg["context_size"]),
        "architecture": cfg.get("architecture", "standard"),
        "cached_activations_path": cfg["cached_activations_path"],
        "dataset_path": cfg["dataset_path"],
        "training_tokens": int(cfg["training_tokens"]),
        "train_batch_size_tokens": int(cfg["train_batch_size_tokens"]),
        "log_to_wandb": bool(cfg.get("log_to_wandb", False)),
        "wandb_project": cfg.get("wandb_project"),
        "run_name": cfg.get("run_name"),
        "wandb_log_frequency": cfg.get("wandb_log_frequency"),
        "eval_every_n_wandb_logs": cfg.get("eval_every_n_wandb_logs"),
        "dtype": cfg.get("dtype", "float32"),
        "device": cfg.get("device", "cuda"),
        "final_sae_path": str(output_dir / "final_sae"),
        "inference_sae_path": str(output_dir / "inference_sae"),
        "runner_result_type": type(result).__name__,
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)
    return metadata
