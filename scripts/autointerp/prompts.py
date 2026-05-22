"""Prompts and DeepSeek request payloads."""

from __future__ import annotations

import json
from typing import Any

from scripts.autointerp.config import AutointerpConfig
from scripts.autointerp.features import compact_feature


SYSTEM_PROMPT = """You are labeling sparse autoencoder features from Qwen3-4B layer-20 activations.

You will receive JSON evidence for several SAE features. For each feature, the
examples are top-activating text windows; the activating token is marked with
double square brackets like [[this]].

Return valid JSON only. Do not include markdown. Do not include extra keys.
Use short, specific labels. Prefer observable evidence over speculation. If a
feature is unclear, say so with low confidence instead of inventing a story.
"""


def build_batch_prompt(
    batch_id: int,
    rows: list[dict[str, Any]],
    config: AutointerpConfig,
) -> str:
    features = [compact_feature(row, config) for row in rows]
    payload = {
        "task": "Write SAE feature labels from top activating examples. Return JSON.",
        "batch_id": batch_id,
        "output_schema": {
            "batch_id": "integer",
            "labels": [
                {
                    "feature_id": "integer",
                    "label": "short lowercase-ish noun phrase",
                    "confidence": "number from 0.0 to 1.0",
                    "reason": "one short evidence-based sentence",
                }
            ],
        },
        "rules": [
            "Return exactly one label object per input feature_id.",
            "Use only evidence from the provided examples.",
            "Mention uncertainty with lower confidence when examples are mixed.",
            "The activating token is wrapped in [[double brackets]].",
            "Your response must be valid JSON.",
        ],
        "features": features,
    }
    return json.dumps(payload, ensure_ascii=False)


def build_request_payload(
    batch_id: int,
    rows: list[dict[str, Any]],
    config: AutointerpConfig,
) -> dict[str, Any]:
    return {
        "model": config.model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_batch_prompt(batch_id, rows, config)},
        ],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
        "user_id": "qwen3-sae-autointerp",
    }
