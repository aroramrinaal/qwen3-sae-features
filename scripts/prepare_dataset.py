"""SAELens pretokenization wrapper for Modal runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path) as file:
        cfg = yaml.safe_load(file)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return cfg


def build_pretokenize_config(cfg_dict: dict[str, Any]):
    from sae_lens import PretokenizeRunnerConfig

    normalized = dict(cfg_dict)
    normalized.pop("prepend_bos", None)
    return PretokenizeRunnerConfig(**normalized)


def run_prepare(config_path: str | Path):
    from sae_lens import PretokenizeRunner

    cfg_dict = load_config(config_path)
    cfg = build_pretokenize_config(cfg_dict)
    return PretokenizeRunner(cfg).run()
