"""Async orchestration for autointerp label generation."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Awaitable, Callable

from scripts.autointerp.config import (
    AutointerpConfig,
    batch_path,
    parse_autointerp_config,
    prepare_output_dir,
)
from scripts.autointerp.deepseek import call_deepseek_batch
from scripts.autointerp.features import make_batches, read_feature_rows
from scripts.autointerp.io import (
    BatchResult,
    merge_batch_files,
    write_failed_batches,
    write_json_atomic,
    write_label_summary,
    write_labels_jsonl,
    write_run_summary,
)


async def run_batches_async(
    rows: list[dict],
    config: AutointerpConfig,
    commit_callback: Callable[[], None] | None,
    async_commit_callback: Callable[[], Awaitable[None]] | None,
) -> list[BatchResult]:
    import httpx

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set. Check the deepseek-secret Modal secret.")

    batches = make_batches(rows, config.batch_features_per_call)
    semaphore = asyncio.Semaphore(config.concurrency)
    completed_since_commit = 0
    completed_lock = asyncio.Lock()

    timeout = httpx.Timeout(config.request_timeout_seconds)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        async def run_one(batch_id: int, batch_rows: list[dict]) -> BatchResult:
            nonlocal completed_since_commit
            path = batch_path(config, batch_id)
            feature_ids = [int(row["feature_id"]) for row in batch_rows]
            if config.skip_existing and path.exists():
                return BatchResult(batch_id=batch_id, feature_ids=feature_ids, ok=True, path=str(path))

            async with semaphore:
                try:
                    response = await call_deepseek_batch(client, batch_id, batch_rows, config)
                    write_json_atomic(path, response)
                    async with completed_lock:
                        completed_since_commit += 1
                        if (
                            (commit_callback is not None or async_commit_callback is not None)
                            and completed_since_commit >= config.commit_every_batches
                        ):
                            await commit_volume(commit_callback, async_commit_callback)
                            completed_since_commit = 0
                    print(
                        f"[autointerp] wrote batch={batch_id} features={feature_ids[0]}..{feature_ids[-1]}",
                        flush=True,
                    )
                    return BatchResult(
                        batch_id=batch_id,
                        feature_ids=feature_ids,
                        ok=True,
                        path=str(path),
                    )
                except Exception as exc:  # noqa: BLE001 - record failed batches.
                    print(f"[autointerp] failed batch={batch_id} error={exc}", flush=True)
                    return BatchResult(
                        batch_id=batch_id,
                        feature_ids=feature_ids,
                        ok=False,
                        error=str(exc),
                    )

        tasks = [run_one(batch_id, batch_rows) for batch_id, batch_rows in enumerate(batches)]
        results = await asyncio.gather(*tasks)

    await commit_volume(commit_callback, async_commit_callback)
    return results


async def commit_volume(
    commit_callback: Callable[[], None] | None,
    async_commit_callback: Callable[[], Awaitable[None]] | None,
) -> None:
    if async_commit_callback is not None:
        await async_commit_callback()
    elif commit_callback is not None:
        commit_callback()


def run_autointerp_labels(
    config_path: str | Path,
    commit_callback: Callable[[], None] | None = None,
    async_commit_callback: Callable[[], Awaitable[None]] | None = None,
) -> dict:
    start_time = time.time()
    config = parse_autointerp_config(config_path)
    prepare_output_dir(config)
    rows = read_feature_rows(config)
    print(
        "[autointerp] starting "
        f"features={len(rows)} batch_size={config.batch_features_per_call} "
        f"concurrency={config.concurrency} output={config.output_path}",
        flush=True,
    )

    results = asyncio.run(
        run_batches_async(
            rows=rows,
            config=config,
            commit_callback=commit_callback,
            async_commit_callback=async_commit_callback,
        )
    )
    failed_path = write_failed_batches(config, results)
    labels = merge_batch_files(config)
    labels_path = write_labels_jsonl(config, labels)
    label_summary_path = write_label_summary(config, labels)
    run_summary_path = write_run_summary(
        config_path=config_path,
        config=config,
        labels=labels,
        results=results,
        elapsed_seconds=time.time() - start_time,
    )
    if async_commit_callback is not None:
        asyncio.run(async_commit_callback())
    elif commit_callback is not None:
        commit_callback()

    failed_count = sum(1 for result in results if not result.ok)
    return {
        "output_path": str(config.output_path),
        "labels_path": str(labels_path),
        "run_summary_path": str(run_summary_path),
        "label_summary_path": str(label_summary_path),
        "failed_batches_path": str(failed_path),
        "labels_written": len(labels),
        "failed_batches": failed_count,
        "total_batches": len(results),
    }
