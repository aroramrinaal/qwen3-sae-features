"""Top-level orchestration for SAELens SAE training."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from scripts.training.config import build_training_cfg, load_config
from scripts.training.metadata import save_training_metadata


def run_train(config_path: str | Path) -> dict[str, Any]:
    from datasets import load_from_disk
    from sae_lens import LanguageModelSAETrainingRunner

    cfg = load_config(config_path)
    output_dir = Path(cfg["sae_output_path"])
    if not output_dir.is_absolute():
        raise ValueError("sae_output_path must be an absolute /vol path.")
    if not output_dir.is_relative_to(Path("/vol/saes")):
        raise ValueError("Refusing to overwrite outside /vol/saes.")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_cfg = build_training_cfg(cfg)
    tokenized_dataset = load_from_disk(cfg["dataset_path"])
    sae = LanguageModelSAETrainingRunner(
        training_cfg,
        override_dataset=tokenized_dataset,
    ).run()

    final_sae_path = output_dir / "final_sae"
    inference_sae_path = output_dir / "inference_sae"
    sae.save_model(final_sae_path)
    sae.save_inference_model(inference_sae_path)
    metadata = save_training_metadata(cfg, output_dir, sae)

    return {
        "sae_path": str(output_dir),
        "final_sae_path": str(final_sae_path),
        "inference_sae_path": str(inference_sae_path),
        "metadata_path": str(output_dir / "metadata.json"),
        "cached_activations_path": cfg["cached_activations_path"],
        "hook_name": cfg["hook_name"],
        "d_in": int(cfg["d_in"]),
        "d_sae": int(cfg["d_sae"]),
        "training_tokens": int(cfg["training_tokens"]),
        "final_checkpoint_saved": final_sae_path.exists(),
        "metadata": metadata,
    }
