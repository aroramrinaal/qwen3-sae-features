"""Build decoded activation examples for feature dashboard outputs."""

from __future__ import annotations

from typing import Any

from scripts.dashboard.config import DashboardConfig
from scripts.dashboard.topk import TopKResult


def token_text(tokenizer: Any, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def decode_window(
    tokenizer: Any,
    token_ids: list[int],
    token_position: int,
    window_tokens: int,
) -> dict[str, Any]:
    start = max(0, token_position - window_tokens)
    end = min(len(token_ids), token_position + window_tokens + 1)
    left_ids = token_ids[start:token_position]
    center_ids = token_ids[token_position : token_position + 1]
    right_ids = token_ids[token_position + 1 : end]

    left = token_text(tokenizer, left_ids)
    center = token_text(tokenizer, center_ids)
    right = token_text(tokenizer, right_ids)

    return {
        "window_start": start,
        "window_end": end,
        "token_id": int(center_ids[0]) if center_ids else None,
        "token_text": center,
        "text": f"{left}[[{center}]]{right}",
    }


def build_feature_rows(
    dataset: Any,
    tokenizer: Any,
    config: DashboardConfig,
    tracked_feature_ids: list[int],
    output_feature_ids: list[int],
    top_k_result: TopKResult,
) -> list[dict[str, Any]]:
    local_index_by_feature_id = {
        feature_id: local_idx for local_idx, feature_id in enumerate(tracked_feature_ids)
    }

    feature_rows: list[dict[str, Any]] = []
    for feature_id in output_feature_ids:
        local_idx = local_index_by_feature_id[feature_id]
        examples = build_feature_examples(
            dataset=dataset,
            tokenizer=tokenizer,
            config=config,
            top_indices=top_k_result.indices[local_idx],
            top_values=top_k_result.values[local_idx],
            context_size=top_k_result.context_size,
        )
        feature_rows.append(
            {
                "feature_id": int(feature_id),
                "max_activation": examples[0]["activation"] if examples else None,
                "top_examples": examples,
            }
        )

    feature_rows.sort(
        key=lambda row: row["max_activation"] if row["max_activation"] is not None else -float("inf"),
        reverse=True,
    )
    return feature_rows


def build_feature_examples(
    dataset: Any,
    tokenizer: Any,
    config: DashboardConfig,
    top_indices: Any,
    top_values: Any,
    context_size: int,
) -> list[dict[str, Any]]:
    examples = []
    for rank in range(config.top_k):
        if len(examples) >= config.max_activation_examples_per_feature:
            break
        global_token_index = int(top_indices[rank])
        activation = float(top_values[rank])
        if (
            global_token_index < 0
            or activation == float("-inf")
            or activation < config.min_example_activation
        ):
            continue

        row_idx = global_token_index // context_size
        token_position = global_token_index % context_size
        token_ids = [int(token_id) for token_id in dataset[int(row_idx)]["token_ids"]]
        window = decode_window(tokenizer, token_ids, int(token_position), config.window_tokens)
        examples.append(
            {
                "rank": rank + 1,
                "activation": activation,
                "global_token_index": global_token_index,
                "row_index": int(row_idx),
                "token_position": int(token_position),
                **window,
            }
        )
    return examples


def sample_rank_indices(
    rng: Any,
    start: int,
    end: int,
    count: int,
) -> list[int]:
    if count <= 0 or start >= end:
        return []
    population = list(range(start, end))
    if len(population) <= count:
        return population
    return sorted(rng.sample(population, count))
