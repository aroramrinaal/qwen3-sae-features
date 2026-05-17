"""Reusable Hugging Face inference utilities for Qwen3 base causal generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MODEL_NAME = "Qwen/Qwen3-4B-Base"


@dataclass
class InferConfig:
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True


def build_tokenizer(model_name: str = MODEL_NAME) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def build_model(model_name: str = MODEL_NAME) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )


def generate_text(
    prompt: str,
    tokenizer: Any,
    model: Any,
    config: InferConfig | None = None,
) -> str:
    import torch

    cfg = config or InferConfig()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            do_sample=cfg.do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)
