"""Top-level orchestration for feature dashboard generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from scripts.dashboard.config import parse_dashboard_config, prepare_output_dir
from scripts.dashboard.examples import build_feature_rows
from scripts.dashboard.loading import (
    load_activation_dataset,
    load_sae,
    load_tokenizer,
    validate_feature_ids,
)
from scripts.dashboard.reporting import write_dashboard_outputs
from scripts.dashboard.topk import select_output_features, stream_top_k_feature_activations


def run_feature_dashboard(
    config_path: str | Path,
    commit_callback: Callable[[], None] | None = None,
) -> dict[str, Any]:
    config = parse_dashboard_config(config_path)
    prepare_output_dir(config)

    dataset = load_activation_dataset(config)
    sae = load_sae(config)
    d_sae = int(sae.cfg.d_sae)
    configured_feature_ids = validate_feature_ids(config.feature_ids, d_sae)
    tracked_feature_ids = configured_feature_ids or list(range(d_sae))

    top_k_result = stream_top_k_feature_activations(
        dataset=dataset,
        sae=sae,
        config=config,
        tracked_feature_ids=tracked_feature_ids,
        commit_callback=commit_callback,
    )

    tokenizer = load_tokenizer(config)
    output_feature_ids = select_output_features(config, tracked_feature_ids, top_k_result.values)
    feature_rows = build_feature_rows(
        dataset=dataset,
        tokenizer=tokenizer,
        config=config,
        tracked_feature_ids=tracked_feature_ids,
        output_feature_ids=output_feature_ids,
        top_k_result=top_k_result,
    )

    return write_dashboard_outputs(
        config_path=config_path,
        config=config,
        sae=sae,
        top_k_result=top_k_result,
        tracked_feature_count=len(tracked_feature_ids),
        feature_rows=feature_rows,
    )
