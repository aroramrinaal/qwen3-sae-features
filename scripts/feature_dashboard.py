"""Collect top activating token contexts for a trained SAE.

This is the dashboarding step before autointerp. It uses SAELens to load the
trained SAE and run sae.encode() on cached residual-stream activations. It does
not run the base Qwen model again.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


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
    diverse_preview_top: int
    diverse_preview_middle: int
    diverse_preview_tail: int
    diverse_preview_seed: int
    progress_interval_batches: int
    checkpoint_interval_batches: int
    resume_from_checkpoint: bool
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
    batches_seen: int


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
        diverse_preview_top=int(cfg.get("diverse_preview_top", 20)),
        diverse_preview_middle=int(cfg.get("diverse_preview_middle", 20)),
        diverse_preview_tail=int(cfg.get("diverse_preview_tail", 20)),
        diverse_preview_seed=int(cfg.get("diverse_preview_seed", 42)),
        progress_interval_batches=int(cfg.get("progress_interval_batches", 50)),
        checkpoint_interval_batches=int(cfg.get("checkpoint_interval_batches", 250)),
        resume_from_checkpoint=bool(cfg.get("resume_from_checkpoint", True)),
        feature_ids=[int(feature_id) for feature_id in feature_ids]
        if feature_ids is not None
        else None,
        device=str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")),
        dtype=str(cfg.get("dtype", "float32")),
        local_files_only=bool(cfg.get("local_files_only", True)),
        trust_remote_code=bool(cfg.get("trust_remote_code", True)),
    )


def prepare_output_dir(config: DashboardConfig) -> None:
    checkpoint_exists = checkpoint_path(config).exists()
    should_keep_for_resume = config.resume_from_checkpoint and checkpoint_exists
    if config.overwrite and config.output_path.exists() and not should_keep_for_resume:
        if not config.output_path.is_relative_to(Path("/vol/features")):
            raise ValueError(f"Refusing to overwrite outside /vol/features: {config.output_path}")
        shutil.rmtree(config.output_path)
    config.output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path(config).parent.mkdir(parents=True, exist_ok=True)


def load_activation_dataset(config: DashboardConfig):
    from datasets import load_from_disk

    dataset = load_from_disk(str(config.activation_path))
    if config.hook_name not in dataset.column_names:
        raise ValueError(
            f"Hook column {config.hook_name!r} not found. Columns: {dataset.column_names}"
        )
    if "token_ids" not in dataset.column_names:
        raise ValueError("Feature dashboarding needs token_ids in the cached activation dataset.")
    return dataset.with_format("torch", columns=[config.hook_name, "token_ids"])


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
    commit_callback: Callable[[], None] | None = None,
) -> TopKResult:
    import torch

    tracked_count = len(tracked_feature_ids)
    feature_id_tensor = torch.tensor(tracked_feature_ids, device=config.device, dtype=torch.long)
    checkpoint = load_topk_checkpoint(config, tracked_count)
    if checkpoint is None:
        top_values = torch.full((tracked_count, config.top_k), -torch.inf, dtype=torch.float32)
        top_indices = torch.full((tracked_count, config.top_k), -1, dtype=torch.long)
        rows_seen = 0
        tokens_seen = 0
        context_size = 0
        batches_seen = 0
    else:
        top_values = checkpoint["top_values"]
        top_indices = checkpoint["top_indices"]
        rows_seen = int(checkpoint["rows_seen"])
        tokens_seen = int(checkpoint["tokens_seen"])
        context_size = int(checkpoint["context_size"])
        batches_seen = int(checkpoint["batches_seen"])

    total_rows = estimate_total_rows(dataset, config)
    total_tokens_estimate = None
    start_time = time.time()
    last_log_time = start_time
    initial_tokens_seen = tokens_seen
    print(
        "[dashboard] starting scan "
        f"resume={checkpoint is not None} rows_seen={rows_seen} "
        f"tokens_seen={tokens_seen} total_rows={total_rows} "
        f"batch_rows={config.batch_rows} top_k={config.top_k}",
        flush=True,
    )

    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataset.iter(batch_size=config.batch_rows)):
            if batch_idx < batches_seen:
                continue
            batch_acts = activation_batch_to_tensor(batch[config.hook_name])
            if batch_acts.ndim != 3:
                raise ValueError(
                    "Expected cached activations shaped [batch, context, d_in], "
                    f"got {list(batch_acts.shape)}"
                )

            rows_in_batch, context_size, d_in = batch_acts.shape
            total_tokens_estimate = total_rows * context_size
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
            batches_seen = batch_idx + 1

            if should_log_progress(batches_seen, config, last_log_time):
                last_log_time = time.time()
                log_dashboard_progress(
                    rows_seen=rows_seen,
                    tokens_seen=tokens_seen,
                    total_rows=total_rows,
                    total_tokens_estimate=total_tokens_estimate,
                    start_time=start_time,
                    initial_tokens_seen=initial_tokens_seen,
                    batches_seen=batches_seen,
                )

            if should_checkpoint(batches_seen, config):
                save_topk_checkpoint(
                    config=config,
                    top_values=top_values,
                    top_indices=top_indices,
                    rows_seen=rows_seen,
                    tokens_seen=tokens_seen,
                    context_size=context_size,
                    batches_seen=batches_seen,
                    commit_callback=commit_callback,
                )

    if tokens_seen == 0:
        raise ValueError(f"No activation rows were read from {config.activation_path}")

    save_topk_checkpoint(
        config=config,
        top_values=top_values,
        top_indices=top_indices,
        rows_seen=rows_seen,
        tokens_seen=tokens_seen,
        context_size=context_size,
        batches_seen=batches_seen,
        commit_callback=commit_callback,
    )

    return TopKResult(
        values=top_values,
        indices=top_indices,
        rows_seen=rows_seen,
        tokens_seen=tokens_seen,
        context_size=context_size,
        batches_seen=batches_seen,
    )


def checkpoint_path(config: DashboardConfig) -> Path:
    return config.output_path / "checkpoints" / "topk.pt"


def load_topk_checkpoint(config: DashboardConfig, tracked_count: int) -> dict[str, Any] | None:
    import torch

    path = checkpoint_path(config)
    if not config.resume_from_checkpoint or not path.exists():
        return None

    checkpoint = torch.load(path, map_location="cpu")
    expected_shape = (tracked_count, config.top_k)
    values = checkpoint.get("top_values")
    indices = checkpoint.get("top_indices")
    if values is None or indices is None:
        raise ValueError(f"Checkpoint {path} is missing top_values/top_indices")
    if tuple(values.shape) != expected_shape or tuple(indices.shape) != expected_shape:
        raise ValueError(
            f"Checkpoint {path} shape mismatch. Expected {expected_shape}, "
            f"got values={tuple(values.shape)} indices={tuple(indices.shape)}"
        )
    checkpoint_batch_rows = checkpoint.get("batch_rows")
    if checkpoint_batch_rows is not None and int(checkpoint_batch_rows) != config.batch_rows:
        raise ValueError(
            f"Checkpoint {path} was created with batch_rows={checkpoint_batch_rows}, "
            f"but this run uses batch_rows={config.batch_rows}. Delete the checkpoint "
            "or set resume_from_checkpoint: false for a fresh run."
        )

    print(
        "[dashboard] resuming checkpoint "
        f"path={path} rows_seen={checkpoint.get('rows_seen')} "
        f"tokens_seen={checkpoint.get('tokens_seen')} "
        f"batches_seen={checkpoint.get('batches_seen')}",
        flush=True,
    )
    return checkpoint


def save_topk_checkpoint(
    config: DashboardConfig,
    top_values: Any,
    top_indices: Any,
    rows_seen: int,
    tokens_seen: int,
    context_size: int,
    batches_seen: int,
    commit_callback: Callable[[], None] | None,
) -> None:
    import torch

    path = checkpoint_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "top_values": top_values.cpu(),
            "top_indices": top_indices.cpu(),
            "rows_seen": rows_seen,
            "tokens_seen": tokens_seen,
            "context_size": context_size,
            "batches_seen": batches_seen,
            "batch_rows": config.batch_rows,
            "top_k": config.top_k,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        },
        path,
    )
    if commit_callback is not None:
        commit_callback()
    print(
        "[dashboard] checkpoint saved "
        f"rows={rows_seen} tokens={tokens_seen} batches={batches_seen} path={path}",
        flush=True,
    )


def estimate_total_rows(dataset: Any, config: DashboardConfig) -> int:
    total_rows = len(dataset)
    if config.max_rows is not None:
        return min(total_rows, config.max_rows)
    return total_rows


def should_log_progress(batches_seen: int, config: DashboardConfig, last_log_time: float) -> bool:
    if config.progress_interval_batches > 0 and batches_seen % config.progress_interval_batches == 0:
        return True
    return time.time() - last_log_time >= 300


def should_checkpoint(batches_seen: int, config: DashboardConfig) -> bool:
    return (
        config.checkpoint_interval_batches > 0
        and batches_seen % config.checkpoint_interval_batches == 0
    )


def log_dashboard_progress(
    rows_seen: int,
    tokens_seen: int,
    total_rows: int,
    total_tokens_estimate: int | None,
    start_time: float,
    initial_tokens_seen: int,
    batches_seen: int,
) -> None:
    elapsed = max(time.time() - start_time, 1.0)
    tokens_per_second = max(tokens_seen - initial_tokens_seen, 0) / elapsed
    eta_hours = None
    if total_tokens_estimate is not None and tokens_per_second > 0:
        remaining_tokens = max(total_tokens_estimate - tokens_seen, 0)
        eta_hours = remaining_tokens / tokens_per_second / 3600
    percent = rows_seen / max(total_rows, 1) * 100
    eta_text = "unknown" if eta_hours is None else f"{eta_hours:.2f}"
    print(
        "[dashboard] progress "
        f"batches={batches_seen} rows={rows_seen}/{total_rows} "
        f"tokens={tokens_seen} pct={percent:.2f}% "
        f"tok/s={tokens_per_second:.1f} eta_hr={eta_text}",
        flush=True,
    )


def activation_batch_to_tensor(batch_acts: Any):
    import torch

    if isinstance(batch_acts, torch.Tensor):
        return batch_acts.to(dtype=torch.float32)
    return torch.tensor(batch_acts, dtype=torch.float32)


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
    diverse_preview_path = config.output_path / "diverse_preview.md"

    write_jsonl(jsonl_path, feature_rows)
    write_preview(preview_path, feature_rows, config.preview_features)
    write_diverse_preview(diverse_preview_path, feature_rows, config)

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
        "batches_seen": top_k_result.batches_seen,
        "batch_rows": config.batch_rows,
        "progress_interval_batches": config.progress_interval_batches,
        "checkpoint_interval_batches": config.checkpoint_interval_batches,
        "checkpoint_path": str(checkpoint_path(config)),
        "device": config.device,
        "dtype": config.dtype,
        "jsonl_path": str(jsonl_path),
        "preview_path": str(preview_path),
        "diverse_preview_path": str(diverse_preview_path),
    }
    with open(summary_path, "w") as file:
        json.dump(summary, file, indent=2, sort_keys=True)

    return {
        "output_path": str(config.output_path),
        "top_activations_path": str(jsonl_path),
        "summary_path": str(summary_path),
        "preview_path": str(preview_path),
        "diverse_preview_path": str(diverse_preview_path),
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
    preview_rows = [row for row in rows if row["max_activation"] is not None]
    lines = [
        "# SAE Feature Dashboard Preview",
        "",
        "Bracketed text marks the max-activating token for that example.",
        "",
    ]
    for row in preview_rows[:preview_features]:
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


def write_diverse_preview(
    path: Path,
    rows: list[dict[str, Any]],
    config: DashboardConfig,
) -> None:
    import random

    rows_with_examples = [row for row in rows if row["max_activation"] is not None]
    selected: list[tuple[str, int, dict[str, Any]]] = []
    seen_feature_ids: set[int] = set()

    def add_row(section: str, rank: int, row: dict[str, Any]) -> None:
        feature_id = int(row["feature_id"])
        if feature_id in seen_feature_ids:
            return
        seen_feature_ids.add(feature_id)
        selected.append((section, rank, row))

    for rank, row in enumerate(rows_with_examples[: config.diverse_preview_top], start=1):
        add_row("Top max-activation features", rank, row)

    rng = random.Random(config.diverse_preview_seed)
    middle_start = min(200, len(rows_with_examples))
    middle_end = min(2000, len(rows_with_examples))
    tail_start = min(2000, len(rows_with_examples))
    tail_end = min(6000, len(rows_with_examples))

    middle_indices = sample_rank_indices(
        rng=rng,
        start=middle_start,
        end=middle_end,
        count=config.diverse_preview_middle,
    )
    tail_indices = sample_rank_indices(
        rng=rng,
        start=tail_start,
        end=tail_end,
        count=config.diverse_preview_tail,
    )

    for idx in middle_indices:
        add_row("Random features from ranks 200-2000", idx + 1, rows_with_examples[idx])
    for idx in tail_indices:
        add_row("Random features from ranks 2000-6000", idx + 1, rows_with_examples[idx])

    lines = [
        "# Diverse SAE Feature Dashboard Preview",
        "",
        "This preview samples beyond the loudest max-activation features.",
        "Bracketed text marks the max-activating token for that example.",
        "",
    ]

    current_section = None
    for section, rank, row in selected:
        if section != current_section:
            lines.append(f"## {section}")
            lines.append("")
            current_section = section
        lines.append(f"### Rank {rank} - Feature {row['feature_id']}")
        lines.append("")
        lines.append(f"max_activation: `{row['max_activation']:.6g}`")
        lines.append("")
        for idx, example in enumerate(row["top_examples"], start=1):
            activation = example["activation"]
            token_position = example["token_position"]
            text = example["text"].replace("\n", "\\n")
            lines.append(f"{idx}. `{activation:.6g}` pos `{token_position}` {text}")
        lines.append("")

    path.write_text("\n".join(lines))


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


def main() -> None:
    import argparse
    import pprint

    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    args = parser.parse_args()
    pprint.pp(run_feature_dashboard(args.config_path))


if __name__ == "__main__":
    main()
