"""Activation-capture smoke test utilities for Qwen3."""

from __future__ import annotations

from typing import Any

from scripts.infer import build_model, build_tokenizer


def capture_activation_metadata(
    prompt: str = "The capital of France is",
    layer_idx: int = 20,
) -> dict[str, Any]:
    import torch

    tokenizer = build_tokenizer()
    model = build_model()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    token_count = inputs["input_ids"].shape[-1]
    captured: dict[str, torch.Tensor] = {}

    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        hidden_states = output[0] if isinstance(output, tuple) else output
        captured["activation"] = hidden_states.detach()

    handle = model.model.layers[layer_idx].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            model(**inputs)
    finally:
        handle.remove()

    activation = captured["activation"]
    return {
        "layer": layer_idx,
        "shape": str(activation.shape),
        "dtype": str(activation.dtype),
        "device": str(activation.device),
        "token_count": token_count,
    }
