"""Analyze SAE autointerp labels and write triage groups."""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


REQUIRED_FIELDS = ["input_path", "output_path"]

CONFIDENCE_BINS = [
    (0.0, 0.2),
    (0.2, 0.4),
    (0.4, 0.6),
    (0.6, 0.8),
    (0.8, 1.0000001),
]

BORING_KEYWORDS = {
    "abbreviation",
    "article",
    "bracket",
    "citation",
    "comma",
    "copyright",
    "digit",
    "end-of-text",
    "equals sign",
    "file extension",
    "formatting",
    "function words",
    "html",
    "isbn",
    "license",
    "newline",
    "number",
    "page",
    "parenthesis",
    "period",
    "preposition",
    "punctuation",
    "registered trademark",
    "space token",
    "syntax",
    "table",
    "token",
    "url",
    "year",
}

UNCLEAR_KEYWORDS = {
    "diverse",
    "miscellaneous",
    "mixed",
    "no clear",
    "unclear",
    "unrelated",
    "varied",
}

TOPIC_KEYWORDS = {
    "academic",
    "astronom",
    "biology",
    "climate",
    "disease",
    "election",
    "environment",
    "europe",
    "genetic",
    "history",
    "legal",
    "medical",
    "medicine",
    "politic",
    "psychology",
    "science",
    "skin",
    "vitamin",
}

STYLE_INSTRUCTION_KEYWORDS = {
    "advice",
    "click",
    "command",
    "disclaimer",
    "encourage",
    "improve",
    "instruction",
    "limitation",
    "must",
    "negation",
    "please",
    "polite",
    "request",
    "should",
    "warning",
}


@dataclass(frozen=True)
class AnalysisConfig:
    input_path: Path
    output_path: Path
    group_output_path: Path
    overwrite: bool
    high_confidence_threshold: float
    low_confidence_threshold: float
    candidate_confidence_threshold: float
    max_examples_in_markdown: int


def load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path) as file:
        cfg = yaml.safe_load(file)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")

    missing = [field for field in REQUIRED_FIELDS if field not in cfg]
    if missing:
        raise ValueError(f"Missing required autointerp analysis fields: {missing}")
    return cfg


def parse_analysis_config(path: str | Path) -> AnalysisConfig:
    cfg = load_config(path)
    input_path = Path(cfg["input_path"])
    output_path = Path(cfg["output_path"])
    group_output_path = Path(cfg.get("group_output_path", output_path / "feature_groups"))

    for name, value in [("output_path", output_path), ("group_output_path", group_output_path)]:
        if not value.is_absolute():
            raise ValueError(f"{name} must be an absolute /vol path.")
        if not value.is_relative_to(Path("/vol/features")):
            raise ValueError(f"Refusing to write {name} outside /vol/features.")

    return AnalysisConfig(
        input_path=input_path,
        output_path=output_path,
        group_output_path=group_output_path,
        overwrite=bool(cfg.get("overwrite", False)),
        high_confidence_threshold=float(cfg.get("high_confidence_threshold", 0.85)),
        low_confidence_threshold=float(cfg.get("low_confidence_threshold", 0.6)),
        candidate_confidence_threshold=float(cfg.get("candidate_confidence_threshold", 0.85)),
        max_examples_in_markdown=int(cfg.get("max_examples_in_markdown", 40)),
    )


def read_labels(path: Path) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    with open(path) as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            row["feature_id"] = int(row["feature_id"])
            row["confidence"] = float(row.get("confidence", 0.0))
            labels.append(row)
    if not labels:
        raise ValueError(f"No labels read from {path}")
    labels.sort(key=lambda row: row["feature_id"])
    return labels


def prepare_output_dirs(config: AnalysisConfig) -> None:
    for path in [config.output_path, config.group_output_path]:
        if config.overwrite and path.exists():
            if not path.is_relative_to(Path("/vol/features")):
                raise ValueError(f"Refusing to overwrite outside /vol/features: {path}")
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def label_text(row: dict[str, Any]) -> str:
    return f"{row.get('label', '')} {row.get('reason', '')}".lower()


def contains_any(row: dict[str, Any], keywords: set[str]) -> bool:
    text = label_text(row)
    return any(keyword in text for keyword in keywords)


def confidence_histogram(labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for start, end in CONFIDENCE_BINS:
        count = sum(1 for row in labels if start <= row["confidence"] < end)
        rows.append(
            {
                "range": f"{start:.1f}-{min(end, 1.0):.1f}",
                "count": count,
                "percent": round(100 * count / len(labels), 2),
            }
        )
    return rows


def build_groups(config: AnalysisConfig, labels: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    high_confidence = [
        row for row in labels if row["confidence"] >= config.high_confidence_threshold
    ]
    low_confidence_or_unclear = [
        row
        for row in labels
        if row["confidence"] < config.low_confidence_threshold or contains_any(row, UNCLEAR_KEYWORDS)
    ]
    boring_formatting = [row for row in labels if contains_any(row, BORING_KEYWORDS)]
    topic_semantic = [
        row
        for row in labels
        if contains_any(row, TOPIC_KEYWORDS) and not contains_any(row, BORING_KEYWORDS)
    ]
    style_instruction = [
        row
        for row in labels
        if contains_any(row, STYLE_INSTRUCTION_KEYWORDS) and not contains_any(row, BORING_KEYWORDS)
    ]
    candidate_steering_features = [
        row
        for row in labels
        if row["confidence"] >= config.candidate_confidence_threshold
        and not contains_any(row, BORING_KEYWORDS | UNCLEAR_KEYWORDS)
        and contains_any(row, TOPIC_KEYWORDS | STYLE_INSTRUCTION_KEYWORDS)
    ]

    groups = {
        "high_confidence": high_confidence,
        "low_confidence_or_unclear": low_confidence_or_unclear,
        "boring_formatting": boring_formatting,
        "topic_semantic": topic_semantic,
        "style_instruction": style_instruction,
        "candidate_steering_features": candidate_steering_features,
    }
    for rows in groups.values():
        rows.sort(key=lambda row: (-row["confidence"], row["feature_id"]))
    return groups


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    config: AnalysisConfig,
    labels: list[dict[str, Any]],
    groups: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    histogram = confidence_histogram(labels)
    label_counts = Counter(str(row.get("label", "")).lower() for row in labels)
    label_frequency = [
        {"label": label, "count": count}
        for label, count in label_counts.most_common()
    ]

    write_csv(config.output_path / "confidence_histogram.csv", ["range", "count", "percent"], histogram)
    write_csv(config.output_path / "label_frequency.csv", ["label", "count"], label_frequency)

    group_paths = {}
    for name, rows in groups.items():
        path = config.group_output_path / f"{name}.jsonl"
        write_jsonl(path, rows)
        group_paths[name] = str(path)

    confidences = [row["confidence"] for row in labels]
    sorted_confidences = sorted(confidences)
    midpoint = len(sorted_confidences) // 2
    median = (
        sorted_confidences[midpoint]
        if len(sorted_confidences) % 2
        else (sorted_confidences[midpoint - 1] + sorted_confidences[midpoint]) / 2
    )
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(config.input_path),
        "output_path": str(config.output_path),
        "group_output_path": str(config.group_output_path),
        "labels": len(labels),
        "confidence": {
            "min": min(confidences),
            "max": max(confidences),
            "mean": sum(confidences) / len(confidences),
            "median": median,
            "histogram": histogram,
        },
        "groups": {name: {"count": len(rows), "path": group_paths[name]} for name, rows in groups.items()},
        "top_labels": label_frequency[:50],
    }

    with open(config.output_path / "summary.json", "w") as file:
        json.dump(summary, file, indent=2, sort_keys=True)

    write_markdown_summary(config, summary, groups)
    return summary


def write_markdown_summary(
    config: AnalysisConfig,
    summary: dict[str, Any],
    groups: dict[str, list[dict[str, Any]]],
) -> None:
    lines = [
        "# Autointerp Label Analysis",
        "",
        f"labels: `{summary['labels']}`",
        f"input: `{summary['input_path']}`",
        "",
        "## Confidence",
        "",
        (
            f"- min `{summary['confidence']['min']:.2f}`, "
            f"median `{summary['confidence']['median']:.2f}`, "
            f"mean `{summary['confidence']['mean']:.2f}`, "
            f"max `{summary['confidence']['max']:.2f}`"
        ),
    ]
    for row in summary["confidence"]["histogram"]:
        lines.append(f"- `{row['range']}`: {row['count']} ({row['percent']:.2f}%)")

    lines.extend(["", "## Groups", ""])
    for name, info in summary["groups"].items():
        lines.append(f"- `{name}`: {info['count']} -> `{info['path']}`")

    lines.extend(["", "## Candidate Steering Features", ""])
    for row in groups["candidate_steering_features"][: config.max_examples_in_markdown]:
        lines.append(
            f"- feature `{row['feature_id']}` ({row['confidence']:.2f}): "
            f"{row.get('label', '')} - {row.get('reason', '')}"
        )

    lines.extend(["", "## Most Common Exact Labels", ""])
    for row in summary["top_labels"][:25]:
        lines.append(f"- {row['count']}: {row['label']}")

    (config.output_path / "summary.md").write_text("\n".join(lines))


def run_autointerp_analysis(
    config_path: str | Path,
    commit_callback: Callable[[], None] | None = None,
) -> dict[str, Any]:
    config = parse_analysis_config(config_path)
    prepare_output_dirs(config)
    labels = read_labels(config.input_path)
    groups = build_groups(config, labels)
    summary = write_outputs(config, labels, groups)
    if commit_callback is not None:
        commit_callback()
    return summary


def main() -> None:
    import argparse
    import pprint

    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    args = parser.parse_args()
    pprint.pp(run_autointerp_analysis(args.config_path))


if __name__ == "__main__":
    main()
