"""Inspect cached SAELens activation datasets saved on the Modal Volume."""

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


def _feature_dtype(feature: Any) -> str | None:
    return getattr(feature, "dtype", None)


def _feature_shape(feature: Any) -> list[int] | None:
    shape = getattr(feature, "shape", None)
    return list(shape) if shape is not None else None


def inspect_cached_activations(config_path: str | Path) -> dict[str, Any]:
    import torch
    from datasets import load_from_disk

    cfg = load_config(config_path)
    activation_path = cfg["activation_path"]
    hook_name = cfg["hook_name"]
    sample_rows = int(cfg.get("sample_rows", 2))

    dataset = load_from_disk(activation_path)
    if hook_name not in dataset.column_names:
        raise ValueError(
            f"Hook column {hook_name!r} not found. Columns: {dataset.column_names}"
        )

    hook_feature = dataset.features[hook_name]
    token_feature = dataset.features.get("token_ids")
    row_count = len(dataset)
    feature_shape = _feature_shape(hook_feature)
    feature_dtype = _feature_dtype(hook_feature)

    sample_count = min(sample_rows, row_count)
    activation_samples = []
    token_samples = []

    for row_idx in range(sample_count):
        row = dataset[row_idx]
        activations = torch.tensor(row[hook_name], dtype=torch.float32)
        activation_samples.append(activations)
        token_ids = row.get("token_ids")
        if token_ids is not None:
            token_samples.extend(token_ids[: min(8, len(token_ids))])

    stacked = torch.stack(activation_samples) if activation_samples else None
    stats = {}
    if stacked is not None:
        stats = {
            "sample_shape": list(stacked.shape),
            "mean": float(stacked.mean()),
            "std": float(stacked.std()),
            "min": float(stacked.min()),
            "max": float(stacked.max()),
            "has_nan": bool(torch.isnan(stacked).any()),
            "has_inf": bool(torch.isinf(stacked).any()),
        }

    expected_rows = cfg.get("expected_rows")
    expected_context_size = cfg.get("expected_context_size")
    expected_d_in = cfg.get("expected_d_in")
    expected_dtype = cfg.get("expected_dtype")

    checks = {
        "rows_match": expected_rows is None or row_count == expected_rows,
        "shape_match": (
            expected_context_size is None
            or expected_d_in is None
            or feature_shape == [expected_context_size, expected_d_in]
        ),
        "dtype_match": expected_dtype is None or feature_dtype == expected_dtype,
        "no_nan": not stats.get("has_nan", False),
        "no_inf": not stats.get("has_inf", False),
    }

    context_size = feature_shape[0] if feature_shape else None
    total_activation_tokens = row_count * context_size if context_size is not None else None

    return {
        "activation_path": activation_path,
        "columns": list(dataset.column_names),
        "features": {
            hook_name: {
                "shape": feature_shape,
                "dtype": feature_dtype,
                "type": type(hook_feature).__name__,
            },
            "token_ids": {
                "shape": _feature_shape(token_feature) if token_feature else None,
                "dtype": _feature_dtype(token_feature) if token_feature else None,
                "type": type(token_feature).__name__ if token_feature else None,
            },
        },
        "row_count": row_count,
        "total_activation_tokens": total_activation_tokens,
        "sample_token_ids": token_samples,
        "sample_stats": stats,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
