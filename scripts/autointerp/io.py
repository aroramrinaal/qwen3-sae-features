"""Autointerp output writing and merging."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.autointerp.config import AutointerpConfig, batch_dir


@dataclass
class BatchResult:
    batch_id: int
    feature_ids: list[int]
    ok: bool
    path: str | None = None
    error: str | None = None


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as file:
        json.dump(data, file, indent=2, sort_keys=True, ensure_ascii=False)
    tmp_path.replace(path)


def merge_batch_files(config: AutointerpConfig) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    for path in sorted(batch_dir(config).glob("batch_*.json")):
        with open(path) as file:
            batch = json.load(file)
        for label in batch.get("labels", []):
            labels.append(label)
    labels.sort(key=lambda row: int(row["feature_id"]))
    return labels


def write_labels_jsonl(config: AutointerpConfig, labels: list[dict[str, Any]]) -> Path:
    path = config.output_path / "labels.jsonl"
    with open(path, "w") as file:
        for label in labels:
            file.write(json.dumps(label, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def write_failed_batches(config: AutointerpConfig, results: list[BatchResult]) -> Path:
    path = config.output_path / "failed_batches.jsonl"
    failed = [result for result in results if not result.ok]
    with open(path, "w") as file:
        for result in failed:
            file.write(json.dumps(result.__dict__, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def write_label_summary(config: AutointerpConfig, labels: list[dict[str, Any]]) -> Path:
    path = config.output_path / "label_summary.md"
    rows = sorted(labels, key=lambda row: float(row["confidence"]), reverse=True)
    lines = [
        "# SAE Autointerp Label Summary",
        "",
        f"model: `{config.model_id}`",
        f"labels: `{len(labels)}`",
        "",
        "## High Confidence Sample",
        "",
    ]
    for label in rows[:50]:
        lines.append(
            f"- feature `{label['feature_id']}` ({label['confidence']:.2f}): "
            f"{label['label']} - {label['reason']}"
        )
    path.write_text("\n".join(lines))
    return path


def write_run_summary(
    config_path: str | Path,
    config: AutointerpConfig,
    labels: list[dict[str, Any]],
    results: list[BatchResult],
    elapsed_seconds: float,
) -> Path:
    path = config.output_path / "run_summary.json"
    ok_batches = [result for result in results if result.ok]
    failed_batches = [result for result in results if not result.ok]
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "input_path": str(config.input_path),
        "output_path": str(config.output_path),
        "model_id": config.model_id,
        "api_base_url": config.api_base_url,
        "labels_written": len(labels),
        "ok_batches": len(ok_batches),
        "failed_batches": len(failed_batches),
        "batch_features_per_call": config.batch_features_per_call,
        "examples_per_feature": config.examples_per_feature,
        "concurrency": config.concurrency,
        "elapsed_seconds": elapsed_seconds,
    }
    with open(path, "w") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
    return path
