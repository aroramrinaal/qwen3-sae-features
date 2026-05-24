"""Load datasets, models, and tokenizers for feature dashboards."""

from __future__ import annotations

from scripts.dashboard.config import DashboardConfig


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
