"""Input feature evidence loading and prompt compaction."""

from __future__ import annotations

import json
from typing import Any

from scripts.autointerp.config import AutointerpConfig


def read_feature_rows(config: AutointerpConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(config.input_path) as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("max_activation") is None:
                continue
            rows.append(row)
            if config.max_features is not None and len(rows) >= config.max_features:
                break
    if not rows:
        raise ValueError(f"No feature rows read from {config.input_path}")
    return rows


def make_batches(rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [rows[start : start + batch_size] for start in range(0, len(rows), batch_size)]


def compact_feature(row: dict[str, Any], config: AutointerpConfig) -> dict[str, Any]:
    examples = []
    for example in row.get("top_examples", [])[: config.examples_per_feature]:
        text = str(example.get("text", ""))
        if len(text) > config.max_context_chars_per_example:
            half = max(config.max_context_chars_per_example // 2 - 20, 1)
            text = f"{text[:half]} ... {text[-half:]}"
        examples.append(
            {
                "activation": round(float(example["activation"]), 6),
                "token_position": int(example["token_position"]),
                "token_text": str(example.get("token_text", "")),
                "text": text,
            }
        )

    return {
        "feature_id": int(row["feature_id"]),
        "max_activation": round(float(row["max_activation"]), 6),
        "examples": examples,
    }
