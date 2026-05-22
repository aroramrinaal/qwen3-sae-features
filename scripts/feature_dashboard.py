"""Collect top activating token contexts for a trained SAE.

This is the dashboarding step before autointerp. It streams cached residual
activations through the SAE encoder, keeps only top-k token positions per
feature, then decodes small tokenizer windows around those positions.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = [
    "model_name",
    "activation_path",
    "sae_path",
    "hook_name",
    "output_path",
]


def load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path) as file:
        cfg = yaml.safe_load(file)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")

    missing = [field for field in REQUIRED_FIELDS if field not in cfg]
    if missing:
        raise ValueError(f"Missing required feature dashboard fields: {missing}")
    return cfg


def _parse_feature_ids(cfg: dict[str, Any], d_sae: int) -> list[int] | None:
    raw_feature_ids = cfg.get("feature_ids")
    if raw_feature_ids is None:
        return None
    feature_ids = [int(feature_id) for feature_id in raw_feature_ids]
    bad_ids = [feature_id for feature_id in feature_ids if feature_id < 0 or feature_id >= d_sae]
    if bad_ids:
        raise ValueError(f"feature_ids out of range for d_sae={d_sae}: {bad_ids[:10]}")
    return feature_ids


def _token_text(tokenizer: Any, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def _decode_window(
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

    left = _token_text(tokenizer, left_ids)
    center = _token_text(tokenizer, center_ids)
    right = _token_text(tokenizer, right_ids)

    return {
        "window_start": start,
        "window_end": end,
        "token_id": int(center_ids[0]) if center_ids else None,
        "token_text": center,
        "text": f"{left}[[{center}]]{right}",
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_preview(path: Path, rows: list[dict[str, Any]], preview_features: int) -> None:
    lines = [
        "# SAE Feature Dashboard Preview",
        "",
        "Bracketed text marks the max-activating token for that example.",
        "",
    ]
    for row in rows[:preview_features]:
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


def _load_sae(load_path: Path, device: str, dtype: str):
    from sae_lens import SAE

    return SAE.load_from_disk(load_path, device=device, dtype=dtype)


def _select_output_features(
    cfg: dict[str, Any],
    tracked_feature_ids: list[int],
    top_values: Any,
) -> list[int]:
    requested_count = cfg.get("num_features")
    min_activation = cfg.get("min_activation")
    max_values = top_values[:, 0].detach().cpu()
    local_index_by_feature_id = {
        feature_id: local_idx for local_idx, feature_id in enumerate(tracked_feature_ids)
    }

    candidates = tracked_feature_ids
    if min_activation is not None:
        threshold = float(min_activation)
        candidates = [
            feature_id
            for local_idx, feature_id in enumerate(tracked_feature_ids)
            if float(max_values[local_idx]) >= threshold
        ]

    if requested_count is None:
        return candidates

    count = int(requested_count)
    ranked_local = sorted(
        range(len(candidates)),
        key=lambda idx: float(max_values[local_index_by_feature_id[candidates[idx]]]),
        reverse=True,
    )
    return [candidates[idx] for idx in ranked_local[:count]]


def _safe_remove_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        if not output_dir.is_relative_to(Path("/vol/features")):
            raise ValueError(f"Refusing to overwrite outside /vol/features: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def run_feature_dashboard(config_path: str | Path) -> dict[str, Any]:
    import torch
    from datasets import load_from_disk
    from transformers import AutoTokenizer

    cfg = load_config(config_path)

    activation_path = Path(cfg["activation_path"])
    sae_root = Path(cfg["sae_path"])
    load_path = Path(cfg.get("load_path", sae_root / "final_sae"))
    output_dir = Path(cfg["output_path"])
    if not output_dir.is_absolute():
        raise ValueError("output_path must be an absolute /vol path.")
    if not output_dir.is_relative_to(Path("/vol/features")):
        raise ValueError("Refusing to write feature dashboards outside /vol/features.")

    overwrite = bool(cfg.get("overwrite", False))
    if overwrite:
        _safe_remove_output_dir(output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    hook_name = str(cfg["hook_name"])
    top_k = int(cfg.get("top_k", 20))
    batch_rows = int(cfg.get("batch_rows", 2))
    window_tokens = int(cfg.get("window_tokens", 32))
    max_rows = cfg.get("max_rows")
    max_rows = int(max_rows) if max_rows is not None else None
    device = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    dtype = str(cfg.get("dtype", "float32"))

    dataset = load_from_disk(str(activation_path))
    if hook_name not in dataset.column_names:
        raise ValueError(f"Hook column {hook_name!r} not found. Columns: {dataset.column_names}")
    if "token_ids" not in dataset.column_names:
        raise ValueError("Feature dashboarding needs token_ids in the cached activation dataset.")

    sae = _load_sae(load_path, device=device, dtype=dtype)
    d_sae = int(sae.cfg.d_sae)
    configured_feature_ids = _parse_feature_ids(cfg, d_sae)
    tracked_feature_ids = configured_feature_ids or list(range(d_sae))
    tracked_count = len(tracked_feature_ids)
    feature_id_tensor = torch.tensor(tracked_feature_ids, device=device, dtype=torch.long)

    top_values = torch.full((tracked_count, top_k), -torch.inf, dtype=torch.float32)
    top_indices = torch.full((tracked_count, top_k), -1, dtype=torch.long)

    rows_seen = 0
    tokens_seen = 0

    with torch.inference_mode():
        iterator = dataset.iter(batch_size=batch_rows)
        for batch in iterator:
            batch_acts = torch.tensor(batch[hook_name], dtype=torch.float32)
            if batch_acts.ndim != 3:
                raise ValueError(
                    f"Expected cached activations shaped [batch, context, d_in], got {list(batch_acts.shape)}"
                )
            rows_in_batch, context_size, d_in = batch_acts.shape
            if max_rows is not None and rows_seen >= max_rows:
                break
            if max_rows is not None and rows_seen + rows_in_batch > max_rows:
                keep_rows = max_rows - rows_seen
                batch_acts = batch_acts[:keep_rows]
                rows_in_batch = keep_rows

            flat_acts = batch_acts.reshape(rows_in_batch * context_size, d_in).to(device)
            feature_acts = sae.encode(flat_acts)
            if configured_feature_ids is not None:
                feature_acts = feature_acts.index_select(dim=1, index=feature_id_tensor)

            batch_k = min(top_k, feature_acts.shape[0])
            batch_values, batch_positions = torch.topk(feature_acts.float(), k=batch_k, dim=0)
            batch_values = batch_values.transpose(0, 1).cpu()
            batch_indices = (batch_positions.transpose(0, 1).cpu() + tokens_seen).long()

            combined_values = torch.cat([top_values, batch_values], dim=1)
            combined_indices = torch.cat([top_indices, batch_indices], dim=1)
            new_values, new_positions = torch.topk(combined_values, k=top_k, dim=1)
            top_values = new_values
            top_indices = combined_indices.gather(1, new_positions)

            rows_seen += rows_in_batch
            tokens_seen += rows_in_batch * context_size

    if tokens_seen == 0:
        raise ValueError(f"No activation rows were read from {activation_path}")

    output_feature_ids = _select_output_features(cfg, tracked_feature_ids, top_values)
    local_index_by_feature_id = {
        feature_id: local_idx for local_idx, feature_id in enumerate(tracked_feature_ids)
    }

    tokenizer = AutoTokenizer.from_pretrained(
        str(cfg["model_name"]),
        local_files_only=bool(cfg.get("local_files_only", True)),
        trust_remote_code=bool(cfg.get("trust_remote_code", True)),
    )

    feature_rows: list[dict[str, Any]] = []
    for feature_id in output_feature_ids:
        local_idx = local_index_by_feature_id[feature_id]
        examples = []
        for rank in range(top_k):
            global_token_index = int(top_indices[local_idx, rank])
            activation = float(top_values[local_idx, rank])
            if global_token_index < 0 or activation == float("-inf"):
                continue
            row_idx = global_token_index // context_size
            token_position = global_token_index % context_size
            token_ids = [int(token_id) for token_id in dataset[int(row_idx)]["token_ids"]]
            window = _decode_window(tokenizer, token_ids, int(token_position), window_tokens)
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

    jsonl_path = output_dir / "top_activations.jsonl"
    summary_path = output_dir / "feature_summary.json"
    preview_path = output_dir / "preview.md"
    _write_jsonl(jsonl_path, feature_rows)
    _write_preview(preview_path, feature_rows, int(cfg.get("preview_features", 20)))

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "model_name": str(cfg["model_name"]),
        "activation_path": str(activation_path),
        "sae_path": str(sae_root),
        "load_path": str(load_path),
        "hook_name": hook_name,
        "d_sae": d_sae,
        "tracked_features": tracked_count,
        "written_features": len(feature_rows),
        "top_k": top_k,
        "window_tokens": window_tokens,
        "rows_seen": rows_seen,
        "tokens_seen": tokens_seen,
        "batch_rows": batch_rows,
        "device": device,
        "dtype": dtype,
        "jsonl_path": str(jsonl_path),
        "preview_path": str(preview_path),
    }
    with open(summary_path, "w") as file:
        json.dump(summary, file, indent=2, sort_keys=True)

    return {
        "output_path": str(output_dir),
        "top_activations_path": str(jsonl_path),
        "summary_path": str(summary_path),
        "preview_path": str(preview_path),
        "rows_seen": rows_seen,
        "tokens_seen": tokens_seen,
        "written_features": len(feature_rows),
        "top_feature_ids": [row["feature_id"] for row in feature_rows[:10]],
    }


def main() -> None:
    import argparse
    import pprint

    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    args = parser.parse_args()
    pprint.pp(run_feature_dashboard(args.config_path))


if __name__ == "__main__":
    main()
