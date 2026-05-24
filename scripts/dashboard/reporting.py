"""Write feature dashboard artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.dashboard.checkpointing import checkpoint_path
from scripts.dashboard.config import DashboardConfig
from scripts.dashboard.examples import sample_rank_indices
from scripts.dashboard.topk import TopKResult


def write_dashboard_outputs(
    config_path: str | Path,
    config: DashboardConfig,
    sae: Any,
    top_k_result: TopKResult,
    tracked_feature_count: int,
    feature_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    jsonl_path = config.output_path / "top_activations.jsonl"
    summary_path = config.output_path / "feature_summary.json"
    preview_path = config.output_path / "preview.md"
    diverse_preview_path = config.output_path / "diverse_preview.md"

    write_jsonl(jsonl_path, feature_rows)
    write_preview(preview_path, feature_rows, config.preview_features)
    write_diverse_preview(diverse_preview_path, feature_rows, config)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "model_name": config.model_name,
        "activation_path": str(config.activation_path),
        "sae_path": str(config.sae_path),
        "load_path": str(config.load_path),
        "hook_name": config.hook_name,
        "d_sae": int(sae.cfg.d_sae),
        "tracked_features": tracked_feature_count,
        "written_features": len(feature_rows),
        "top_k": config.top_k,
        "window_tokens": config.window_tokens,
        "min_token_position": config.min_token_position,
        "max_token_position": config.max_token_position,
        "min_example_activation": config.min_example_activation,
        "max_activation_examples_per_feature": config.max_activation_examples_per_feature,
        "rows_seen": top_k_result.rows_seen,
        "tokens_seen": top_k_result.tokens_seen,
        "batches_seen": top_k_result.batches_seen,
        "batch_rows": config.batch_rows,
        "progress_interval_batches": config.progress_interval_batches,
        "checkpoint_interval_batches": config.checkpoint_interval_batches,
        "checkpoint_path": str(checkpoint_path(config)),
        "device": config.device,
        "dtype": config.dtype,
        "jsonl_path": str(jsonl_path),
        "preview_path": str(preview_path),
        "diverse_preview_path": str(diverse_preview_path),
    }
    with open(summary_path, "w") as file:
        json.dump(summary, file, indent=2, sort_keys=True)

    return {
        "output_path": str(config.output_path),
        "top_activations_path": str(jsonl_path),
        "summary_path": str(summary_path),
        "preview_path": str(preview_path),
        "diverse_preview_path": str(diverse_preview_path),
        "rows_seen": top_k_result.rows_seen,
        "tokens_seen": top_k_result.tokens_seen,
        "written_features": len(feature_rows),
        "top_feature_ids": [row["feature_id"] for row in feature_rows[:10]],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_preview(path: Path, rows: list[dict[str, Any]], preview_features: int) -> None:
    preview_rows = [row for row in rows if row["max_activation"] is not None]
    lines = [
        "# SAE Feature Dashboard Preview",
        "",
        "Bracketed text marks the max-activating token for that example.",
        "",
    ]
    for row in preview_rows[:preview_features]:
        lines.append(f"## Feature {row['feature_id']}")
        lines.append("")
        lines.append(f"max_activation: `{row['max_activation']:.6g}`")
        lines.append("")
        for idx, example in enumerate(row["top_examples"], start=1):
            activation = example["activation"]
            text = example["text"].replace("\n", "\\n")
            lines.append(f"{idx}. `{activation:.6g}` {text}")
        lines.append("")
    path.write_text("\n".join(lines))


def write_diverse_preview(
    path: Path,
    rows: list[dict[str, Any]],
    config: DashboardConfig,
) -> None:
    import random

    rows_with_examples = [row for row in rows if row["max_activation"] is not None]
    selected: list[tuple[str, int, dict[str, Any]]] = []
    seen_feature_ids: set[int] = set()

    def add_row(section: str, rank: int, row: dict[str, Any]) -> None:
        feature_id = int(row["feature_id"])
        if feature_id in seen_feature_ids:
            return
        seen_feature_ids.add(feature_id)
        selected.append((section, rank, row))

    for rank, row in enumerate(rows_with_examples[: config.diverse_preview_top], start=1):
        add_row("Top max-activation features", rank, row)

    rng = random.Random(config.diverse_preview_seed)
    middle_start = min(200, len(rows_with_examples))
    middle_end = min(2000, len(rows_with_examples))
    tail_start = min(2000, len(rows_with_examples))
    tail_end = min(6000, len(rows_with_examples))

    middle_indices = sample_rank_indices(
        rng=rng,
        start=middle_start,
        end=middle_end,
        count=config.diverse_preview_middle,
    )
    tail_indices = sample_rank_indices(
        rng=rng,
        start=tail_start,
        end=tail_end,
        count=config.diverse_preview_tail,
    )

    for idx in middle_indices:
        add_row("Random features from ranks 200-2000", idx + 1, rows_with_examples[idx])
    for idx in tail_indices:
        add_row("Random features from ranks 2000-6000", idx + 1, rows_with_examples[idx])

    lines = [
        "# Diverse SAE Feature Dashboard Preview",
        "",
        "This preview samples beyond the loudest max-activation features.",
        "Bracketed text marks the max-activating token for that example.",
        "",
    ]

    current_section = None
    for section, rank, row in selected:
        if section != current_section:
            lines.append(f"## {section}")
            lines.append("")
            current_section = section
        lines.append(f"### Rank {rank} - Feature {row['feature_id']}")
        lines.append("")
        lines.append(f"max_activation: `{row['max_activation']:.6g}`")
        lines.append("")
        for idx, example in enumerate(row["top_examples"], start=1):
            activation = example["activation"]
            token_position = example["token_position"]
            text = example["text"].replace("\n", "\\n")
            lines.append(f"{idx}. `{activation:.6g}` pos `{token_position}` {text}")
        lines.append("")

    path.write_text("\n".join(lines))
