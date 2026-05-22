"""Collect top activating token contexts for a trained SAE.

This is the dashboarding step before autointerp. It uses SAELens to load the
trained SAE and run sae.encode() on cached residual-stream activations. It does
not run the base Qwen model again.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
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


@dataclass(frozen=True)
class DashboardConfig:
    model_name: str
    activation_path: Path
    sae_path: Path
    load_path: Path
    hook_name: str
    output_path: Path
    overwrite: bool
    top_k: int
    batch_rows: int
    window_tokens: int
    preview_features: int
    max_rows: int | None
    num_features: int | None
    min_activation: float | None
    min_token_position: int
    max_token_position: int | None
    min_example_activation: float
    max_activation_examples_per_feature: int
    feature_ids: list[int] | None
    device: str
    dtype: str
    local_files_only: bool
    trust_remote_code: bool


@dataclass
class TopKResult:
    values: Any
    indices: Any
    rows_seen: int
    tokens_seen: int
    context_size: int


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


def parse_dashboard_config(path: str | Path) -> DashboardConfig:
    import torch

    cfg = load_config(path)
    sae_path = Path(cfg["sae_path"])
    output_path = Path(cfg["output_path"])

    if not output_path.is_absolute():
        raise ValueError("output_path must be an absolute /vol path.")
    if not output_path.is_relative_to(Path("/vol/features")):
        raise ValueError("Refusing to write feature dashboards outside /vol/features.")

    max_rows = cfg.get("max_rows")
    num_features = cfg.get("num_features")
    min_activation = cfg.get("min_activation")
    max_token_position = cfg.get("max_token_position")
    feature_ids = cfg.get("feature_ids")

    return DashboardConfig(
        model_name=str(cfg["model_name"]),
        activation_path=Path(cfg["activation_path"]),
        sae_path=sae_path,
        load_path=Path(cfg.get("load_path", sae_path / "final_sae")),
        hook_name=str(cfg["hook_name"]),
        output_path=output_path,
        overwrite=bool(cfg.get("overwrite", False)),
        top_k=int(cfg.get("top_k", 20)),
        batch_rows=int(cfg.get("batch_rows", 2)),
        window_tokens=int(cfg.get("window_tokens", 32)),
        preview_features=int(cfg.get("preview_features", 20)),
        max_rows=int(max_rows) if max_rows is not None else None,
        num_features=int(num_features) if num_features is not None else None,
        min_activation=float(min_activation) if min_activation is not None else None,
        min_token_position=int(cfg.get("min_token_position", 0)),
        max_token_position=int(max_token_position) if max_token_position is not None else None,
        min_example_activation=float(cfg.get("min_example_activation", 0.0)),
        max_activation_examples_per_feature=int(
            cfg.get("max_activation_examples_per_feature", cfg.get("top_k", 20))
        ),
        feature_ids=[int(feature_id) for feature_id in feature_ids]
        if feature_ids is not None
        else None,
        device=str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")),
        dtype=str(cfg.get("dtype", "float32")),
        local_files_only=bool(cfg.get("local_files_only", True)),
        trust_remote_code=bool(cfg.get("trust_remote_code", True)),
    )


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if overwrite and output_dir.exists():
        if not output_dir.is_relative_to(Path("/vol/features")):
            raise ValueError(f"Refusing to overwrite outside /vol/features: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def load_activation_dataset(config: DashboardConfig):
    from datasets import load_from_disk

    dataset = load_from_disk(str(config.activation_path))
    if config.hook_name not in dataset.column_names:
        raise ValueError(
            f"Hook column {config.hook_name!r} not found. Columns: {dataset.column_names}"
        )
    if "token_ids" not in dataset.column_names:
        raise ValueError("Feature dashboarding needs token_ids in the cached activation dataset.")
    return dataset


def load_sae(config: DashboardConfig):
    from sae_lens import SAE

    return SAE.load_from_disk(config.load_path, device=config.device, dtype=config.dtype)


def load_tokenizer(config: DashboardConfig):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        config.model_name,
        local_files_only=config.local_files_only,
        trust_remote_code=config.trust_remote_code,
    )


def validate_feature_ids(feature_ids: list[int] | None, d_sae: int) -> list[int] | None:
    if feature_ids is None:
        return None
    bad_ids = [feature_id for feature_id in feature_ids if feature_id < 0 or feature_id >= d_sae]
    if bad_ids:
        raise ValueError(f"feature_ids out of range for d_sae={d_sae}: {bad_ids[:10]}")
    return feature_ids


def stream_top_k_feature_activations(
    dataset: Any,
    sae: Any,
    config: DashboardConfig,
    tracked_feature_ids: list[int],
) -> TopKResult:
    import torch

    tracked_count = len(tracked_feature_ids)
    feature_id_tensor = torch.tensor(tracked_feature_ids, device=config.device, dtype=torch.long)
    top_values = torch.full((tracked_count, config.top_k), -torch.inf, dtype=torch.float32)
    top_indices = torch.full((tracked_count, config.top_k), -1, dtype=torch.long)

    rows_seen = 0
    tokens_seen = 0
    context_size = 0

    with torch.inference_mode():
        for batch in dataset.iter(batch_size=config.batch_rows):
            batch_acts = torch.tensor(batch[config.hook_name], dtype=torch.float32)
            if batch_acts.ndim != 3:
                raise ValueError(
                    "Expected cached activations shaped [batch, context, d_in], "
                    f"got {list(batch_acts.shape)}"
                )

            rows_in_batch, context_size, d_in = batch_acts.shape
            if config.max_rows is not None and rows_seen >= config.max_rows:
                break
            if config.max_rows is not None and rows_seen + rows_in_batch > config.max_rows:
                keep_rows = config.max_rows - rows_seen
                batch_acts = batch_acts[:keep_rows]
                rows_in_batch = keep_rows

            flat_acts = batch_acts.reshape(rows_in_batch * context_size, d_in).to(config.device)
            feature_acts = sae.encode(flat_acts)
            apply_position_filters(feature_acts, rows_in_batch, context_size, config)
            if config.feature_ids is not None:
                feature_acts = feature_acts.index_select(dim=1, index=feature_id_tensor)

            batch_values, batch_indices = top_k_for_batch(
                feature_acts=feature_acts,
                top_k=config.top_k,
                token_offset=tokens_seen,
            )
            top_values, top_indices = merge_top_k(
                old_values=top_values,
                old_indices=top_indices,
                batch_values=batch_values,
                batch_indices=batch_indices,
                top_k=config.top_k,
            )

            rows_seen += rows_in_batch
            tokens_seen += rows_in_batch * context_size

    if tokens_seen == 0:
        raise ValueError(f"No activation rows were read from {config.activation_path}")

    return TopKResult(
        values=top_values,
        indices=top_indices,
        rows_seen=rows_seen,
        tokens_seen=tokens_seen,
        context_size=context_size,
    )


def apply_position_filters(
    feature_acts: Any,
    rows_in_batch: int,
    context_size: int,
    config: DashboardConfig,
) -> None:
    """Mask row-boundary tokens before top-k selection.

    Row-start tokens have missing left context, and row-end tokens can have
    missing right context. Masking here keeps those positions out of the top-k
    heap instead of filtering them after they have already displaced better
    evidence.
    """

    import torch

    positions = torch.arange(context_size, device=feature_acts.device).repeat(rows_in_batch)
    keep = positions >= config.min_token_position
    if config.max_token_position is not None:
        keep = keep & (positions <= config.max_token_position)
    if bool(keep.all()):
        return
    feature_acts.masked_fill_(~keep[:, None], -torch.inf)


def top_k_for_batch(feature_acts: Any, top_k: int, token_offset: int) -> tuple[Any, Any]:
    import torch

    batch_k = min(top_k, feature_acts.shape[0])
    values, positions = torch.topk(feature_acts.float(), k=batch_k, dim=0)
    values = values.transpose(0, 1).cpu()
    indices = (positions.transpose(0, 1).cpu() + token_offset).long()
    return values, indices


def merge_top_k(
    old_values: Any,
    old_indices: Any,
    batch_values: Any,
    batch_indices: Any,
    top_k: int,
) -> tuple[Any, Any]:
    import torch

    combined_values = torch.cat([old_values, batch_values], dim=1)
    combined_indices = torch.cat([old_indices, batch_indices], dim=1)
    new_values, new_positions = torch.topk(combined_values, k=top_k, dim=1)
    return new_values, combined_indices.gather(1, new_positions)


def select_output_features(
    config: DashboardConfig,
    tracked_feature_ids: list[int],
    top_values: Any,
) -> list[int]:
    max_values = top_values[:, 0].detach().cpu()
    local_index_by_feature_id = {
        feature_id: local_idx for local_idx, feature_id in enumerate(tracked_feature_ids)
    }

    candidates = tracked_feature_ids
    if config.min_activation is not None:
        candidates = [
            feature_id
            for local_idx, feature_id in enumerate(tracked_feature_ids)
            if float(max_values[local_idx]) >= config.min_activation
        ]

    if config.num_features is None:
        return candidates

    ranked_positions = sorted(
        range(len(candidates)),
        key=lambda idx: float(max_values[local_index_by_feature_id[candidates[idx]]]),
        reverse=True,
    )
    return [candidates[idx] for idx in ranked_positions[: config.num_features]]


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


def write_dashboard_outputs(
    config_path: str | Path,
    config: DashboardConfig,
    sae: Any,
    top_k_result: TopKResult,
    tracked_feature_count: int,
    feature_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    jsonl_path = config.output_path / "top_activations.jsonl"
    summary_path = config.output_path / "feature_summary.json"
    preview_path = config.output_path / "preview.md"

    write_jsonl(jsonl_path, feature_rows)
    write_preview(preview_path, feature_rows, config.preview_features)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "model_name": config.model_name,
        "activation_path": str(config.activation_path),
        "sae_path": str(config.sae_path),
        "load_path": str(config.load_path),
        "hook_name": config.hook_name,
        "d_sae": int(sae.cfg.d_sae),
        "tracked_features": tracked_feature_count,
        "written_features": len(feature_rows),
        "top_k": config.top_k,
        "window_tokens": config.window_tokens,
        "min_token_position": config.min_token_position,
        "max_token_position": config.max_token_position,
        "min_example_activation": config.min_example_activation,
        "max_activation_examples_per_feature": config.max_activation_examples_per_feature,
        "rows_seen": top_k_result.rows_seen,
        "tokens_seen": top_k_result.tokens_seen,
        "batch_rows": config.batch_rows,
        "device": config.device,
        "dtype": config.dtype,
        "jsonl_path": str(jsonl_path),
        "preview_path": str(preview_path),
    }
    with open(summary_path, "w") as file:
        json.dump(summary, file, indent=2, sort_keys=True)

    return {
        "output_path": str(config.output_path),
        "top_activations_path": str(jsonl_path),
        "summary_path": str(summary_path),
        "preview_path": str(preview_path),
        "rows_seen": top_k_result.rows_seen,
        "tokens_seen": top_k_result.tokens_seen,
        "written_features": len(feature_rows),
        "top_feature_ids": [row["feature_id"] for row in feature_rows[:10]],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_preview(path: Path, rows: list[dict[str, Any]], preview_features: int) -> None:
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


def run_feature_dashboard(config_path: str | Path) -> dict[str, Any]:
    config = parse_dashboard_config(config_path)
    prepare_output_dir(config.output_path, config.overwrite)

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


def main() -> None:
    import argparse
    import pprint

    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    args = parser.parse_args()
    pprint.pp(run_feature_dashboard(args.config_path))


if __name__ == "__main__":
    main()
