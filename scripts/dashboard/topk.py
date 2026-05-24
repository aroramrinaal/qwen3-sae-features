"""Top-k SAE activation scanning for feature dashboards."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from scripts.dashboard.checkpointing import load_topk_checkpoint, save_topk_checkpoint
from scripts.dashboard.config import DashboardConfig


@dataclass
class TopKResult:
    values: Any
    indices: Any
    rows_seen: int
    tokens_seen: int
    context_size: int
    batches_seen: int


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
    """Mask row-boundary tokens before top-k selection."""

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
