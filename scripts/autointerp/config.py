"""Configuration parsing for SAE autointerp labeling."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = [
    "input_path",
    "output_path",
    "model_id",
    "api_base_url",
]


@dataclass(frozen=True)
class AutointerpConfig:
    input_path: Path
    output_path: Path
    model_id: str
    api_base_url: str
    max_features: int | None
    batch_features_per_call: int
    examples_per_feature: int
    max_context_chars_per_example: int
    concurrency: int
    temperature: float
    max_tokens: int
    skip_existing: bool
    overwrite: bool
    request_timeout_seconds: float
    max_retries: int
    retry_base_seconds: float
    commit_every_batches: int


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError:
        return load_top_level_yaml_keys(Path(path))

    with open(path) as file:
        cfg = yaml.safe_load(file)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")

    missing = [field for field in REQUIRED_FIELDS if field not in cfg]
    if missing:
        raise ValueError(f"Missing required autointerp fields: {missing}")
    return cfg


def load_top_level_yaml_keys(path: Path) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith((" ", "\t", "#")) or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if not key:
            continue
        value = value.split("#", 1)[0].strip()
        if not value:
            config[key] = None
        elif value.lower() == "true":
            config[key] = True
        elif value.lower() == "false":
            config[key] = False
        else:
            config[key] = value.strip("'\"")

    if not config:
        raise ValueError(f"Expected a YAML mapping in {path}")
    missing = [field for field in REQUIRED_FIELDS if field not in config]
    if missing:
        raise ValueError(f"Missing required autointerp fields: {missing}")
    return config


def parse_autointerp_config(path: str | Path) -> AutointerpConfig:
    cfg = load_config(path)
    output_path = Path(cfg["output_path"])
    if not output_path.is_absolute():
        raise ValueError("output_path must be an absolute /vol path.")
    if not output_path.is_relative_to(Path("/vol/features")):
        raise ValueError("Refusing to write autointerp outputs outside /vol/features.")

    max_features = cfg.get("max_features")
    return AutointerpConfig(
        input_path=Path(cfg["input_path"]),
        output_path=output_path,
        model_id=str(cfg["model_id"]),
        api_base_url=str(cfg["api_base_url"]).rstrip("/"),
        max_features=int(max_features) if max_features is not None else None,
        batch_features_per_call=int(cfg.get("batch_features_per_call", 10)),
        examples_per_feature=int(cfg.get("examples_per_feature", 10)),
        max_context_chars_per_example=int(cfg.get("max_context_chars_per_example", 700)),
        concurrency=int(cfg.get("concurrency", 8)),
        temperature=float(cfg.get("temperature", 0.0)),
        max_tokens=int(cfg.get("max_tokens", 4000)),
        skip_existing=bool(cfg.get("skip_existing", True)),
        overwrite=bool(cfg.get("overwrite", False)),
        request_timeout_seconds=float(cfg.get("request_timeout_seconds", 600)),
        max_retries=int(cfg.get("max_retries", 5)),
        retry_base_seconds=float(cfg.get("retry_base_seconds", 2.0)),
        commit_every_batches=int(cfg.get("commit_every_batches", 10)),
    )


def batch_dir(config: AutointerpConfig) -> Path:
    return config.output_path / "batches"


def batch_path(config: AutointerpConfig, batch_id: int) -> Path:
    return batch_dir(config) / f"batch_{batch_id:06d}.json"


def prepare_output_dir(config: AutointerpConfig) -> None:
    if config.overwrite and config.output_path.exists():
        if not config.output_path.is_relative_to(Path("/vol/features")):
            raise ValueError(f"Refusing to overwrite outside /vol/features: {config.output_path}")
        shutil.rmtree(config.output_path)

    batch_dir(config).mkdir(parents=True, exist_ok=True)
