"""Scoring and report generation for steering outputs."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

from scripts.steering.config import SteeringConfig
from scripts.steering.io import write_csv


def label_keywords(label: str | None, reason: str | None) -> list[str]:
    text = f"{label or ''} {reason or ''}".lower()
    keyword_sets = {
        "cooking": ["cook", "recipe", "mix", "stir", "bowl", "add", "water", "milk", "sugar", "honey"],
        "exercise": ["exercise", "fitness", "running", "run", "gym", "strength", "stretch", "cardio", "workout"],
        "adaptation": ["adapt", "evolve", "animal", "environment", "survive", "species", "body", "movement"],
        "feel": ["feel", "feeling", "emotion", "connected", "self", "heart", "comfort", "motivation"],
        "biblical": ["god", "mercy", "heart", "wisdom", "upright", "sin", "blessed", "lord", "righteous"],
        "medical": ["medical", "health", "care", "diagnosis", "treatment", "disease", "risk"],
    }
    for anchor, keywords in keyword_sets.items():
        if anchor in text:
            return keywords
    tokens = [
        token.strip("'\".,:;()[]{}").lower()
        for token in (label or "").replace("/", " ").replace("-", " ").split()
    ]
    return [token for token in tokens if len(token) >= 4][:8]


def keyword_score(text: str, keywords: list[str]) -> int:
    lower = text.lower()
    return sum(lower.count(keyword) for keyword in keywords)


def repetition_score(text: str) -> float:
    words = [word.strip(".,;:!?()[]{}\"'").lower() for word in text.split()]
    words = [word for word in words if word]
    if len(words) < 8:
        return 0.0
    trigrams = list(zip(words, words[1:], words[2:]))
    if not trigrams:
        return 0.0
    counts = Counter(trigrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(trigrams)


def coherence_label(text: str) -> str:
    score = repetition_score(text)
    if score >= 0.22:
        return "low"
    if score >= 0.08:
        return "medium"
    return "high"


def build_comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["feature_id"])].append(row)

    comparison_rows = []
    for feature_id, feature_rows in sorted(grouped.items()):
        label = str(feature_rows[0].get("label") or "")
        reason = str(feature_rows[0].get("reason") or "")
        keywords = label_keywords(label, reason)
        base_scores = [
            keyword_score(row["completion"], keywords)
            for row in feature_rows
            if row["condition"] == "base"
        ]
        base_score = max(base_scores) if base_scores else 0
        steered_rows = [row for row in feature_rows if row["condition"] == "steered"]
        if not steered_rows:
            continue

        def sort_key(row: dict[str, Any]) -> tuple[int, float]:
            return (
                keyword_score(row["completion"], keywords),
                -abs(float(row["alpha"])),
            )

        best = max(steered_rows, key=sort_key)
        best_score = keyword_score(best["completion"], keywords)
        delta = best_score - base_score
        if delta >= 4:
            visible_effect = "strong"
        elif delta >= 2:
            visible_effect = "medium"
        elif delta >= 1:
            visible_effect = "weak"
        else:
            visible_effect = "unclear"

        coherence = coherence_label(best["completion"])
        notes = f"keywords={','.join(keywords[:6])}; score {base_score}->{best_score}"
        comparison_rows.append(
            {
                "feature_id": feature_id,
                "label": label,
                "best_alpha": best["alpha"],
                "position_mode": best.get("position_mode", ""),
                "normalize_direction": best.get("normalize_direction", ""),
                "visible_effect": visible_effect,
                "coherence": coherence,
                "notes": notes,
            }
        )
    return comparison_rows


def write_comparison_table(config: SteeringConfig, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparison_rows = build_comparison_rows(rows)
    write_csv(config.output_path / "comparison_table.csv", comparison_rows)

    lines = [
        "# Steering Comparison Table",
        "",
        "| feature_id | label | best_alpha | position_mode | normalize_direction | visible_effect | coherence | notes |",
        "|---:|---|---:|---|---|---|---|---|",
    ]
    for row in comparison_rows:
        lines.append(
            "| {feature_id} | {label} | {best_alpha} | {position_mode} | {normalize_direction} | "
            "{visible_effect} | {coherence} | {notes} |".format(**row)
        )
    (config.output_path / "comparison_table.md").write_text("\n".join(lines))
    return comparison_rows


def write_summary(config: SteeringConfig, rows: list[dict[str, Any]], stats: dict[str, Any]) -> None:
    lines = [
        "# SAE Steering Smoke Test",
        "",
        f"created: `{datetime.now(timezone.utc).isoformat()}`",
        f"model: `{config.model_path}`",
        f"sae: `{config.sae_load_path}`",
        f"layer: `{config.hook_layer}`",
        f"features: `{', '.join(str(feature_id) for feature_id in config.feature_ids)}`",
        f"alphas: `{', '.join(str(alpha) for alpha in config.alphas)}`",
        f"position_mode: `{config.position_mode}`",
        f"normalize_direction: `{config.normalize_direction}`",
        "",
        "## Decoder Norms",
        "",
        f"- W_dec shape: `{stats['W_dec_shape']}`",
        f"- mean/min/max: `{stats['decoder_norm_mean']:.4f}` / `{stats['decoder_norm_min']:.4f}` / `{stats['decoder_norm_max']:.4f}`",
    ]
    for feature_id, norm in stats["selected_decoder_norms"].items():
        lines.append(f"- feature `{feature_id}` norm: `{norm:.4f}`")

    lines.extend(["", "## Generations", ""])
    for row in rows:
        lines.extend(
            [
                f"### feature {row['feature_id']} | {row['condition']} | alpha {row['alpha']} | seed {row['seed']}",
                "",
                f"label: `{row.get('label', '')}`",
                "",
                "prompt:",
                "",
                f"> {row['prompt']}",
                "",
                "completion:",
                "",
                row["completion"] or "<empty completion>",
                "",
            ]
        )
    (config.output_path / "summary.md").write_text("\n".join(lines))
