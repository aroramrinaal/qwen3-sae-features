"""Inspect a SAELens SAE saved on the Modal Volume."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path) as file:
        cfg = yaml.safe_load(file)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return cfg


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path) as file:
        data = json.load(file)
    return data if isinstance(data, dict) else None


def inspect_sae(config_path: str | Path) -> dict[str, Any]:
    from sae_lens import SAE

    cfg = load_config(config_path)
    sae_root = Path(cfg["sae_path"])
    load_path = Path(cfg.get("load_path", sae_root / "final_sae"))
    metadata_path = Path(cfg.get("metadata_path", sae_root / "metadata.json"))

    sae = SAE.load_from_disk(load_path, device="cpu", dtype="float32")
    state_dict = sae.state_dict()
    metadata = _read_json(metadata_path) or {}

    w_enc_shape = list(state_dict["W_enc"].shape) if "W_enc" in state_dict else None
    w_dec_shape = list(state_dict["W_dec"].shape) if "W_dec" in state_dict else None

    expected_d_in = cfg.get("expected_d_in")
    expected_d_sae = cfg.get("expected_d_sae")
    expected_architecture = cfg.get("expected_architecture")
    expected_hook_name = cfg.get("expected_hook_name")
    expected_cache_path = cfg.get("expected_cached_activations_path")

    checks = {
        "load_succeeded": True,
        "d_in_match": expected_d_in is None or sae.cfg.d_in == expected_d_in,
        "d_sae_match": expected_d_sae is None or sae.cfg.d_sae == expected_d_sae,
        "architecture_match": (
            expected_architecture is None
            or sae.cfg.architecture() == expected_architecture
        ),
        "hook_name_match": (
            expected_hook_name is None
            or getattr(sae.cfg.metadata, "hook_name", None) == expected_hook_name
            or metadata.get("hook_name") == expected_hook_name
        ),
        "cached_activations_path_match": (
            expected_cache_path is None
            or metadata.get("cached_activations_path") == expected_cache_path
        ),
        "w_enc_shape_match": (
            expected_d_in is None
            or expected_d_sae is None
            or w_enc_shape == [expected_d_in, expected_d_sae]
        ),
        "w_dec_shape_match": (
            expected_d_in is None
            or expected_d_sae is None
            or w_dec_shape == [expected_d_sae, expected_d_in]
        ),
    }

    return {
        "sae_path": str(sae_root),
        "load_path": str(load_path),
        "metadata_path": str(metadata_path),
        "checkpoint_files": sorted(
            str(path.relative_to(sae_root))
            for path in sae_root.rglob("*")
            if path.is_file()
        ),
        "architecture": sae.cfg.architecture(),
        "d_in": sae.cfg.d_in,
        "d_sae": sae.cfg.d_sae,
        "dtype": sae.cfg.dtype,
        "device": sae.cfg.device,
        "hook_name": getattr(sae.cfg.metadata, "hook_name", None) or metadata.get("hook_name"),
        "cached_activations_path": metadata.get("cached_activations_path"),
        "training_tokens": metadata.get("training_tokens"),
        "W_enc_shape": w_enc_shape,
        "W_dec_shape": w_dec_shape,
        "b_dec_shape": list(state_dict["b_dec"].shape) if "b_dec" in state_dict else None,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def main() -> None:
    import argparse
    import pprint

    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    args = parser.parse_args()
    pprint.pp(inspect_sae(args.config_path))


if __name__ == "__main__":
    main()
