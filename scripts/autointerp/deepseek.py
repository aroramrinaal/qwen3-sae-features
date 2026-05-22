"""DeepSeek API calls and response normalization."""

from __future__ import annotations

import asyncio
import json
import random
import re
from datetime import datetime, timezone
from typing import Any

from scripts.autointerp.config import AutointerpConfig
from scripts.autointerp.prompts import build_request_payload


async def call_deepseek_batch(
    client: Any,
    batch_id: int,
    rows: list[dict[str, Any]],
    config: AutointerpConfig,
) -> dict[str, Any]:
    payload = build_request_payload(batch_id, rows, config)
    url = f"{config.api_base_url}/chat/completions"

    last_error = None
    for attempt in range(config.max_retries + 1):
        try:
            response = await client.post(url, json=payload)
            if response.status_code == 429:
                raise RuntimeError(f"HTTP 429 rate limit: {response.text[:500]}")
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            parsed = parse_json_content(content)
            return normalize_batch_response(parsed, batch_id, rows, config, data)
        except Exception as exc:  # noqa: BLE001 - preserve full retry surface.
            last_error = exc
            if attempt >= config.max_retries:
                break
            delay = config.retry_base_seconds * (2**attempt) + random.random()
            print(
                f"[autointerp] batch={batch_id} retry={attempt + 1} "
                f"delay={delay:.1f}s error={exc}",
                flush=True,
            )
            await asyncio.sleep(delay)

    raise RuntimeError(f"Batch {batch_id} failed after retries: {last_error}")


def parse_json_content(content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if match is None:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Expected DeepSeek response JSON object")
    return parsed


def normalize_batch_response(
    parsed: dict[str, Any],
    batch_id: int,
    rows: list[dict[str, Any]],
    config: AutointerpConfig,
    raw_response: dict[str, Any],
) -> dict[str, Any]:
    expected_ids = [int(row["feature_id"]) for row in rows]
    labels = parsed.get("labels")
    if not isinstance(labels, list):
        raise ValueError("DeepSeek response missing labels list")

    by_feature: dict[int, dict[str, Any]] = {}
    for label in labels:
        if not isinstance(label, dict):
            continue
        feature_id = int(label.get("feature_id"))
        if feature_id not in expected_ids:
            continue
        by_feature[feature_id] = normalize_label(label, config)

    missing = [feature_id for feature_id in expected_ids if feature_id not in by_feature]
    if missing:
        raise ValueError(f"DeepSeek response missing feature_ids: {missing}")

    usage = raw_response.get("usage") if isinstance(raw_response, dict) else None
    return {
        "batch_id": batch_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": config.model_id,
        "feature_ids": expected_ids,
        "labels": [by_feature[feature_id] for feature_id in expected_ids],
        "usage": usage,
    }


def normalize_label(label: dict[str, Any], config: AutointerpConfig) -> dict[str, Any]:
    confidence = float(label.get("confidence", 0.0))
    confidence = min(max(confidence, 0.0), 1.0)
    return {
        "feature_id": int(label["feature_id"]),
        "label": str(label.get("label", "unclear feature")).strip()[:160],
        "confidence": confidence,
        "reason": str(label.get("reason", "")).strip()[:600],
        "model_id": config.model_id,
        "num_examples_used": config.examples_per_feature,
    }
