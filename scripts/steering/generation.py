"""Model loading and generation helpers for steering runs."""

from __future__ import annotations

import random
from typing import Any

from scripts.steering.config import SteeringConfig


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
        "position_mode": config.position_mode,
    }
    return w_dec, stats


def generation_kwargs(config: SteeringConfig, tokenizer: Any) -> dict[str, Any]:
    kwargs = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if config.do_sample:
        kwargs["temperature"] = config.temperature
        kwargs["top_p"] = config.top_p
    return kwargs


def generate_completion(
    model: Any,
    tokenizer: Any,
    prompt: str,
    config: SteeringConfig,
) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs(config, tokenizer))
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
            if config.position_mode == "last_token":
                hidden[:, -1, :] = hidden[:, -1, :] + alpha * steer
            elif config.position_mode == "all_positions":
                hidden = hidden + alpha * steer
            else:
                raise AssertionError(f"Unhandled position_mode={config.position_mode!r}")
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    handle = model.model.layers[config.hook_layer].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            output_ids = model.generate(**inputs, **generation_kwargs(config, tokenizer))
    finally:
        handle.remove()

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
