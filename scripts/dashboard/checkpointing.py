"""Checkpoint helpers for resumable feature dashboard scans."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from scripts.dashboard.config import DashboardConfig


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
