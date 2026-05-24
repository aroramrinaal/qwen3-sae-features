"""Read and write steering artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def load_label_map(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None or not path.exists():
        return {}

    labels: dict[int, dict[str, Any]] = {}
    with open(path) as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            labels[int(row["feature_id"])] = row
    return labels


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
