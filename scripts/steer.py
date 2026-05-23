"""Run SAE decoder-direction steering smoke tests on Qwen3."""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


REQUIRED_FIELDS = [
    "model_path",
    "sae_path",
    "output_path",
    "feature_ids",
    "prompts",
]


@dataclass(frozen=True)
class SteeringConfig:
    model_path: Path
    sae_path: Path
    sae_load_path: Path
    output_path: Path
    candidate_labels_path: Path | None
    feature_ids: list[int]
    prompts: list[str]
    alphas: list[float]
    hook_layer: int
    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: float
    seeds: list[int]
    steer_on_prompt: bool
    normalize_direction: bool
    overwrite: bool
    dtype: str


def load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path) as file:
        cfg = yaml.safe_load(file)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")

    missing = [field for field in REQUIRED_FIELDS if field not in cfg]
    if missing:
        raise ValueError(f"Missing required steering fields: {missing}")
    return cfg


def parse_steering_config(path: str | Path) -> SteeringConfig:
    cfg = load_config(path)
    output_path = Path(cfg["output_path"])
    if not output_path.is_absolute():
        raise ValueError("output_path must be an absolute /vol path.")
    if not output_path.is_relative_to(Path("/vol/features")):
        raise ValueError("Refusing to write steering outputs outside /vol/features.")

    labels_path = cfg.get("candidate_labels_path")
    return SteeringConfig(
        model_path=Path(cfg["model_path"]),
        sae_path=Path(cfg["sae_path"]),
        sae_load_path=Path(cfg.get("sae_load_path", Path(cfg["sae_path"]) / "final_sae")),
        output_path=output_path,
        candidate_labels_path=Path(labels_path) if labels_path else None,
        feature_ids=[int(feature_id) for feature_id in cfg["feature_ids"]],
        prompts=[str(prompt) for prompt in cfg["prompts"]],
        alphas=[float(alpha) for alpha in cfg.get("alphas", [-20, 0, 20, 40])],
        hook_layer=int(cfg.get("hook_layer", 20)),
        max_new_tokens=int(cfg.get("max_new_tokens", 96)),
        do_sample=bool(cfg.get("do_sample", False)),
        temperature=float(cfg.get("temperature", 0.7)),
        top_p=float(cfg.get("top_p", 0.9)),
        seeds=[int(seed) for seed in cfg.get("seeds", [0])],
        steer_on_prompt=bool(cfg.get("steer_on_prompt", False)),
        normalize_direction=bool(cfg.get("normalize_direction", False)),
        overwrite=bool(cfg.get("overwrite", False)),
        dtype=str(cfg.get("dtype", "bfloat16")),
    )


def prepare_output_dir(config: SteeringConfig) -> None:
    if config.overwrite and config.output_path.exists():
        if not config.output_path.is_relative_to(Path("/vol/features")):
            raise ValueError(f"Refusing to overwrite outside /vol/features: {config.output_path}")
        shutil.rmtree(config.output_path)
    config.output_path.mkdir(parents=True, exist_ok=True)


def load_label_map(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None or not path.exists():
        return {}

    labels: dict[int, dict[str, Any]] = {}
    with open(path) as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            labels[int(row["feature_id"])] = row
    return labels


def dtype_from_name(name: str) -> Any:
    import torch

    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype={name!r}")


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(config: SteeringConfig) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        dtype=dtype_from_name(config.dtype),
        device_map="auto",
        local_files_only=True,
    )
    model.eval()
    return model, tokenizer


def load_decoder_directions(config: SteeringConfig) -> tuple[Any, dict[str, Any]]:
    import torch
    from sae_lens import SAE

    sae = SAE.load_from_disk(config.sae_load_path, device="cpu", dtype="float32")
    state = sae.state_dict()
    w_dec = state["W_dec"].detach().to(torch.float32)
    feature_ids = torch.tensor(config.feature_ids, dtype=torch.long)
    selected_norms = w_dec[feature_ids].norm(dim=1)
    norm_all = w_dec.norm(dim=1)
    stats = {
        "sae_path": str(config.sae_path),
        "sae_load_path": str(config.sae_load_path),
        "W_dec_shape": list(w_dec.shape),
        "selected_decoder_norms": {
            str(feature_id): float(norm)
            for feature_id, norm in zip(config.feature_ids, selected_norms, strict=True)
        },
        "decoder_norm_mean": float(norm_all.mean()),
        "decoder_norm_min": float(norm_all.min()),
        "decoder_norm_max": float(norm_all.max()),
        "normalize_direction": config.normalize_direction,
    }
    return w_dec, stats


def generate_completion(
    model: Any,
    tokenizer: Any,
    prompt: str,
    config: SteeringConfig,
) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generation_kwargs = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if config.do_sample:
        generation_kwargs["temperature"] = config.temperature
        generation_kwargs["top_p"] = config.top_p

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)
    generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def generate_steered_completion(
    model: Any,
    tokenizer: Any,
    prompt: str,
    config: SteeringConfig,
    direction: Any,
    alpha: float,
) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_length = int(inputs["input_ids"].shape[-1])
    token_counter = {"seen": 0}

    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        token_counter["seen"] += int(hidden.shape[1])
        if config.steer_on_prompt or token_counter["seen"] > prompt_length:
            steer = direction.to(device=hidden.device, dtype=hidden.dtype)
            hidden = hidden.clone()
            hidden[:, -1, :] = hidden[:, -1, :] + alpha * steer
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    generation_kwargs = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if config.do_sample:
        generation_kwargs["temperature"] = config.temperature
        generation_kwargs["top_p"] = config.top_p

    handle = model.model.layers[config.hook_layer].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            output_ids = model.generate(**inputs, **generation_kwargs)
    finally:
        handle.remove()

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_summary(config: SteeringConfig, rows: list[dict[str, Any]], stats: dict[str, Any]) -> None:
    lines = [
        "# SAE Steering Smoke Test",
        "",
        f"created: `{datetime.now(timezone.utc).isoformat()}`",
        f"model: `{config.model_path}`",
        f"sae: `{config.sae_load_path}`",
        f"layer: `{config.hook_layer}`",
        f"features: `{', '.join(str(feature_id) for feature_id in config.feature_ids)}`",
        f"alphas: `{', '.join(str(alpha) for alpha in config.alphas)}`",
        "",
        "## Decoder Norms",
        "",
        f"- W_dec shape: `{stats['W_dec_shape']}`",
        f"- mean/min/max: `{stats['decoder_norm_mean']:.4f}` / `{stats['decoder_norm_min']:.4f}` / `{stats['decoder_norm_max']:.4f}`",
    ]
    for feature_id, norm in stats["selected_decoder_norms"].items():
        lines.append(f"- feature `{feature_id}` norm: `{norm:.4f}`")

    lines.extend(["", "## Generations", ""])
    for row in rows:
        lines.extend(
            [
                f"### feature {row['feature_id']} | {row['condition']} | alpha {row['alpha']} | seed {row['seed']}",
                "",
                f"label: `{row.get('label', '')}`",
                "",
                "prompt:",
                "",
                f"> {row['prompt']}",
                "",
                "completion:",
                "",
                row["completion"] or "<empty completion>",
                "",
            ]
        )
    (config.output_path / "summary.md").write_text("\n".join(lines))


def run_steering(config_path: str | Path, commit_callback: Callable[[], None] | None = None) -> dict[str, Any]:
    config = parse_steering_config(config_path)
    prepare_output_dir(config)
    labels = load_label_map(config.candidate_labels_path)
    model, tokenizer = load_model_and_tokenizer(config)
    w_dec, decoder_stats = load_decoder_directions(config)

    rows: list[dict[str, Any]] = []
    for feature_id in config.feature_ids:
        label = labels.get(feature_id, {})
        raw_direction = w_dec[feature_id]
        direction_norm = raw_direction.norm()
        direction = raw_direction / direction_norm if config.normalize_direction else raw_direction

        for prompt in config.prompts:
            for seed in config.seeds:
                set_seed(seed)
                base_completion = generate_completion(model, tokenizer, prompt, config)
                rows.append(
                    {
                        "condition": "base",
                        "feature_id": feature_id,
                        "label": label.get("label"),
                        "reason": label.get("reason"),
                        "prompt": prompt,
                        "alpha": 0.0,
                        "seed": seed,
                        "completion": base_completion,
                        "decoder_norm": float(direction_norm),
                        "layer": config.hook_layer,
                    }
                )

                for alpha in config.alphas:
                    if alpha == 0:
                        continue
                    set_seed(seed)
                    completion = generate_steered_completion(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        config=config,
                        direction=direction,
                        alpha=alpha,
                    )
                    rows.append(
                        {
                            "condition": "steered",
                            "feature_id": feature_id,
                            "label": label.get("label"),
                            "reason": label.get("reason"),
                            "prompt": prompt,
                            "alpha": alpha,
                            "seed": seed,
                            "completion": completion,
                            "decoder_norm": float(direction_norm),
                            "layer": config.hook_layer,
                        }
                    )

    generation_path = config.output_path / "generations.jsonl"
    summary_json_path = config.output_path / "summary.json"
    write_jsonl(generation_path, rows)
    with open(summary_json_path, "w") as file:
        json.dump(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "config_path": str(config_path),
                "output_path": str(config.output_path),
                "num_generations": len(rows),
                "decoder_stats": decoder_stats,
            },
            file,
            indent=2,
            sort_keys=True,
        )
    write_summary(config, rows, decoder_stats)

    if commit_callback is not None:
        commit_callback()

    return {
        "output_path": str(config.output_path),
        "generations_path": str(generation_path),
        "summary_path": str(config.output_path / "summary.md"),
        "summary_json_path": str(summary_json_path),
        "num_generations": len(rows),
        "decoder_stats": decoder_stats,
    }


def main() -> None:
    import argparse
    import pprint

    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    args = parser.parse_args()
    pprint.pp(run_steering(args.config_path))


if __name__ == "__main__":
    main()
