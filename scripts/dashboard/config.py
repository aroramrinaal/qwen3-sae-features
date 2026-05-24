"""Configuration parsing for SAE feature dashboard runs."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = [
    "model_name",
    "activation_path",
    "sae_path",
    "hook_name",
    "output_path",
]


@dataclass(frozen=True)
class DashboardConfig:
    model_name: str
    activation_path: Path
    sae_path: Path
    load_path: Path
    hook_name: str
    output_path: Path
    overwrite: bool
    top_k: int
    batch_rows: int
    window_tokens: int
    preview_features: int
    max_rows: int | None
    num_features: int | None
    min_activation: float | None
    min_token_position: int
    max_token_position: int | None
    min_example_activation: float
    max_activation_examples_per_feature: int
    diverse_preview_top: int
    diverse_preview_middle: int
    diverse_preview_tail: int
    diverse_preview_seed: int
    progress_interval_batches: int
    checkpoint_interval_batches: int
    resume_from_checkpoint: bool
    feature_ids: list[int] | None
    device: str
    dtype: str
    local_files_only: bool
    trust_remote_code: bool


def load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path) as file:
        cfg = yaml.safe_load(file)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")

    missing = [field for field in REQUIRED_FIELDS if field not in cfg]
    if missing:
        raise ValueError(f"Missing required feature dashboard fields: {missing}")
    return cfg


def parse_dashboard_config(path: str | Path) -> DashboardConfig:
    cfg = load_config(path)
    sae_path = Path(cfg["sae_path"])
    output_path = Path(cfg["output_path"])

    if not output_path.is_absolute():
        raise ValueError("output_path must be an absolute /vol path.")
    if not output_path.is_relative_to(Path("/vol/features")):
        raise ValueError("Refusing to write feature dashboards outside /vol/features.")

    max_rows = cfg.get("max_rows")
    num_features = cfg.get("num_features")
    min_activation = cfg.get("min_activation")
    max_token_position = cfg.get("max_token_position")
    feature_ids = cfg.get("feature_ids")
    device = cfg.get("device")

    return DashboardConfig(
        model_name=str(cfg["model_name"]),
        activation_path=Path(cfg["activation_path"]),
        sae_path=sae_path,
        load_path=Path(cfg.get("load_path", sae_path / "final_sae")),
        hook_name=str(cfg["hook_name"]),
        output_path=output_path,
        overwrite=bool(cfg.get("overwrite", False)),
        top_k=int(cfg.get("top_k", 20)),
        batch_rows=int(cfg.get("batch_rows", 2)),
        window_tokens=int(cfg.get("window_tokens", 32)),
        preview_features=int(cfg.get("preview_features", 20)),
        max_rows=int(max_rows) if max_rows is not None else None,
        num_features=int(num_features) if num_features is not None else None,
        min_activation=float(min_activation) if min_activation is not None else None,
        min_token_position=int(cfg.get("min_token_position", 0)),
        max_token_position=int(max_token_position) if max_token_position is not None else None,
        min_example_activation=float(cfg.get("min_example_activation", 0.0)),
        max_activation_examples_per_feature=int(
            cfg.get("max_activation_examples_per_feature", cfg.get("top_k", 20))
        ),
        diverse_preview_top=int(cfg.get("diverse_preview_top", 20)),
        diverse_preview_middle=int(cfg.get("diverse_preview_middle", 20)),
        diverse_preview_tail=int(cfg.get("diverse_preview_tail", 20)),
        diverse_preview_seed=int(cfg.get("diverse_preview_seed", 42)),
        progress_interval_batches=int(cfg.get("progress_interval_batches", 50)),
        checkpoint_interval_batches=int(cfg.get("checkpoint_interval_batches", 250)),
        resume_from_checkpoint=bool(cfg.get("resume_from_checkpoint", True)),
        feature_ids=[int(feature_id) for feature_id in feature_ids]
        if feature_ids is not None
        else None,
        device=str(device) if device is not None else default_device(),
        dtype=str(cfg.get("dtype", "float32")),
        local_files_only=bool(cfg.get("local_files_only", True)),
        trust_remote_code=bool(cfg.get("trust_remote_code", True)),
    )


def default_device() -> str:
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def prepare_output_dir(config: DashboardConfig) -> None:
    from scripts.dashboard.checkpointing import checkpoint_path

    checkpoint_exists = checkpoint_path(config).exists()
    should_keep_for_resume = config.resume_from_checkpoint and checkpoint_exists
    if config.overwrite and config.output_path.exists() and not should_keep_for_resume:
        if not config.output_path.is_relative_to(Path("/vol/features")):
            raise ValueError(f"Refusing to overwrite outside /vol/features: {config.output_path}")
        shutil.rmtree(config.output_path)
    config.output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path(config).parent.mkdir(parents=True, exist_ok=True)
