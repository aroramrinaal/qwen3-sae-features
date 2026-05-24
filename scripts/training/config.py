"""Configuration helpers for SAELens SAE training."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = [
    "model_name",
    "hook_name",
    "d_in",
    "d_sae",
    "context_size",
    "dataset_path",
    "cached_activations_path",
    "training_tokens",
    "train_batch_size_tokens",
    "sae_output_path",
]


def load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path) as file:
        cfg = yaml.safe_load(file)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    missing = [field for field in REQUIRED_FIELDS if field not in cfg]
    if missing:
        raise ValueError(f"Missing required training config fields: {missing}")
    return cfg


def build_training_cfg(cfg: dict[str, Any]):
    from sae_lens import (
        LanguageModelSAERunnerConfig,
        LoggingConfig,
        StandardTrainingSAEConfig,
    )

    architecture = cfg.get("architecture", "standard")
    if architecture != "standard":
        raise ValueError(f"Smoke trainer only supports architecture=standard, got {architecture}")

    sae_cfg = StandardTrainingSAEConfig(
        d_in=int(cfg["d_in"]),
        d_sae=int(cfg["d_sae"]),
        dtype=str(cfg.get("dtype", "float32")),
        device=str(cfg.get("device", "cuda")),
        apply_b_dec_to_input=bool(cfg.get("apply_b_dec_to_input", True)),
        normalize_activations=str(cfg.get("normalize_activations", "none")),
        l1_coefficient=float(cfg.get("l1_coefficient", 1.0)),
        l1_warm_up_steps=int(cfg.get("l1_warm_up_steps", 0)),
    )

    logger_cfg = build_logging_cfg(LoggingConfig, cfg)

    return LanguageModelSAERunnerConfig(
        sae=sae_cfg,
        model_name=str(cfg["model_name"]),
        model_class_name=str(cfg.get("model_class_name", "AutoModelForCausalLM")),
        hook_name=str(cfg["hook_name"]),
        dataset_path=str(cfg["dataset_path"]),
        streaming=bool(cfg.get("streaming", False)),
        is_dataset_tokenized=bool(cfg.get("is_dataset_tokenized", True)),
        context_size=int(cfg["context_size"]),
        use_cached_activations=bool(cfg.get("use_cached_activations", True)),
        cached_activations_path=str(cfg["cached_activations_path"]),
        n_batches_in_buffer=int(cfg.get("n_batches_in_buffer", 2)),
        training_tokens=int(cfg["training_tokens"]),
        store_batch_size_prompts=int(cfg.get("store_batch_size_prompts", 2)),
        activations_mixing_fraction=float(cfg.get("activations_mixing_fraction", 0.0)),
        device=str(cfg.get("device", "cuda")),
        act_store_device=str(cfg.get("act_store_device", "cpu")),
        seed=int(cfg.get("seed", 42)),
        dtype=str(cfg.get("dtype", "float32")),
        prepend_bos=bool(cfg.get("prepend_bos", False)),
        train_batch_size_tokens=int(cfg["train_batch_size_tokens"]),
        lr=float(cfg.get("lr", 3e-4)),
        lr_scheduler_name=str(cfg.get("lr_scheduler_name", "constant")),
        lr_warm_up_steps=int(cfg.get("lr_warm_up_steps", 0)),
        lr_decay_steps=int(cfg.get("lr_decay_steps", 0)),
        n_eval_batches=int(cfg.get("n_eval_batches", 0)),
        logger=logger_cfg,
        n_checkpoints=int(cfg.get("n_checkpoints", 0)),
        checkpoint_path=str(cfg.get("checkpoint_path")),
        save_final_checkpoint=bool(cfg.get("save_final_checkpoint", True)),
        output_path=str(cfg.get("output_path")),
        model_from_pretrained_kwargs=dict(cfg.get("model_from_pretrained_kwargs") or {}),
        verbose=bool(cfg.get("verbose", True)),
    )


def build_logging_cfg(logging_config_class: type, cfg: dict[str, Any]):
    logging_keys = {
        "log_to_wandb": bool(cfg.get("log_to_wandb", False)),
        "log_activations_store_to_wandb": bool(
            cfg.get("log_activations_store_to_wandb", False)
        ),
        "log_optimizer_state_to_wandb": bool(
            cfg.get("log_optimizer_state_to_wandb", False)
        ),
        "log_weights_to_wandb": bool(cfg.get("log_weights_to_wandb", True)),
        "wandb_project": cfg.get("wandb_project"),
        "wandb_entity": cfg.get("wandb_entity"),
        "wandb_id": cfg.get("wandb_id"),
        "run_name": cfg.get("run_name"),
        "wandb_log_frequency": int(cfg.get("wandb_log_frequency", 10)),
        "eval_every_n_wandb_logs": int(cfg.get("eval_every_n_wandb_logs", 100)),
    }
    provided = {
        key: value for key, value in logging_keys.items() if value is not None
    }

    if is_dataclass(logging_config_class):
        supported = {field.name for field in fields(logging_config_class)}
        provided = {key: value for key, value in provided.items() if key in supported}

    return logging_config_class(**provided)
