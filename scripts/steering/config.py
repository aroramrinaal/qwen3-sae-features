"""Configuration parsing for SAE decoder-direction steering."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = [
    "model_path",
    "sae_path",
    "output_path",
    "feature_ids",
    "prompts",
]


@dataclass(frozen=True)
class SteeringConfig:
    model_path: Path
    sae_path: Path
    sae_load_path: Path
    output_path: Path
    candidate_labels_path: Path | None
    feature_ids: list[int]
    prompts: list[str]
    alphas: list[float]
    hook_layer: int
    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: float
    seeds: list[int]
    steer_on_prompt: bool
    position_mode: str
    normalize_direction: bool
    overwrite: bool
    dtype: str


def load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path) as file:
        cfg = yaml.safe_load(file)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")

    missing = [field for field in REQUIRED_FIELDS if field not in cfg]
    if missing:
        raise ValueError(f"Missing required steering fields: {missing}")
    return cfg


def parse_steering_config(path: str | Path) -> SteeringConfig:
    cfg = load_config(path)
    output_path = Path(cfg["output_path"])
    if not output_path.is_absolute():
        raise ValueError("output_path must be an absolute /vol path.")
    if not output_path.is_relative_to(Path("/vol/features")):
        raise ValueError("Refusing to write steering outputs outside /vol/features.")

    labels_path = cfg.get("candidate_labels_path")
    return SteeringConfig(
        model_path=Path(cfg["model_path"]),
        sae_path=Path(cfg["sae_path"]),
        sae_load_path=Path(cfg.get("sae_load_path", Path(cfg["sae_path"]) / "final_sae")),
        output_path=output_path,
        candidate_labels_path=Path(labels_path) if labels_path else None,
        feature_ids=[int(feature_id) for feature_id in cfg["feature_ids"]],
        prompts=[str(prompt) for prompt in cfg["prompts"]],
        alphas=[float(alpha) for alpha in cfg.get("alphas", [-20, 0, 20, 40])],
        hook_layer=int(cfg.get("hook_layer", 20)),
        max_new_tokens=int(cfg.get("max_new_tokens", 96)),
        do_sample=bool(cfg.get("do_sample", False)),
        temperature=float(cfg.get("temperature", 0.7)),
        top_p=float(cfg.get("top_p", 0.9)),
        seeds=[int(seed) for seed in cfg.get("seeds", [0])],
        steer_on_prompt=bool(cfg.get("steer_on_prompt", False)),
        position_mode=parse_position_mode(cfg.get("position_mode", "last_token")),
        normalize_direction=bool(cfg.get("normalize_direction", False)),
        overwrite=bool(cfg.get("overwrite", False)),
        dtype=str(cfg.get("dtype", "bfloat16")),
    )


def parse_position_mode(value: Any) -> str:
    mode = str(value)
    if mode not in {"last_token", "all_positions"}:
        raise ValueError("position_mode must be one of: last_token, all_positions")
    return mode


def prepare_output_dir(config: SteeringConfig) -> None:
    if config.overwrite and config.output_path.exists():
        if not config.output_path.is_relative_to(Path("/vol/features")):
            raise ValueError(f"Refusing to overwrite outside /vol/features: {config.output_path}")
        shutil.rmtree(config.output_path)
    config.output_path.mkdir(parents=True, exist_ok=True)
