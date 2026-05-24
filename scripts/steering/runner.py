"""Top-level orchestration for steering experiments."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from scripts.steering.config import parse_steering_config, prepare_output_dir
from scripts.steering.generation import (
    generate_completion,
    generate_steered_completion,
    load_decoder_directions,
    load_model_and_tokenizer,
    set_seed,
)
from scripts.steering.io import load_label_map, write_jsonl
from scripts.steering.reporting import write_comparison_table, write_summary


def run_steering(config_path: str | Path, commit_callback: Callable[[], None] | None = None) -> dict[str, Any]:
    config = parse_steering_config(config_path)
    prepare_output_dir(config)
    labels = load_label_map(config.candidate_labels_path)
    model, tokenizer = load_model_and_tokenizer(config)
    w_dec, decoder_stats = load_decoder_directions(config)

    rows: list[dict[str, Any]] = []
    for feature_id in config.feature_ids:
        label = labels.get(feature_id, {})
        raw_direction = w_dec[feature_id]
        direction_norm = raw_direction.norm()
        direction = raw_direction / direction_norm if config.normalize_direction else raw_direction

        for prompt in config.prompts:
            for seed in config.seeds:
                set_seed(seed)
                base_completion = generate_completion(model, tokenizer, prompt, config)
                rows.append(
                    {
                        "condition": "base",
                        "feature_id": feature_id,
                        "label": label.get("label"),
                        "reason": label.get("reason"),
                        "prompt": prompt,
                        "alpha": 0.0,
                        "seed": seed,
                        "completion": base_completion,
                        "decoder_norm": float(direction_norm),
                        "layer": config.hook_layer,
                        "position_mode": "none",
                        "normalize_direction": config.normalize_direction,
                    }
                )

                for alpha in config.alphas:
                    if alpha == 0:
                        continue
                    set_seed(seed)
                    completion = generate_steered_completion(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        config=config,
                        direction=direction,
                        alpha=alpha,
                    )
                    rows.append(
                        {
                            "condition": "steered",
                            "feature_id": feature_id,
                            "label": label.get("label"),
                            "reason": label.get("reason"),
                            "prompt": prompt,
                            "alpha": alpha,
                            "seed": seed,
                            "completion": completion,
                            "decoder_norm": float(direction_norm),
                            "layer": config.hook_layer,
                            "position_mode": config.position_mode,
                            "normalize_direction": config.normalize_direction,
                        }
                    )

    generation_path = config.output_path / "generations.jsonl"
    summary_json_path = config.output_path / "summary.json"
    write_jsonl(generation_path, rows)
    comparison_rows = write_comparison_table(config, rows)
    with open(summary_json_path, "w") as file:
        json.dump(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "config_path": str(config_path),
                "output_path": str(config.output_path),
                "num_generations": len(rows),
                "num_comparison_rows": len(comparison_rows),
                "decoder_stats": decoder_stats,
            },
            file,
            indent=2,
            sort_keys=True,
        )
    write_summary(config, rows, decoder_stats)

    if commit_callback is not None:
        commit_callback()

    return {
        "output_path": str(config.output_path),
        "generations_path": str(generation_path),
        "summary_path": str(config.output_path / "summary.md"),
        "summary_json_path": str(summary_json_path),
        "comparison_table_path": str(config.output_path / "comparison_table.md"),
        "num_generations": len(rows),
        "num_comparison_rows": len(comparison_rows),
        "decoder_stats": decoder_stats,
    }
