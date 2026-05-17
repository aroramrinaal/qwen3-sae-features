"""Modal remote entrypoint for Qwen3 inference on an NVIDIA H100."""

from __future__ import annotations

import modal

from scripts.infer import InferConfig, build_model, build_tokenizer, generate_text

app = modal.App("qwen3-sae-features")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .add_local_python_source("scripts")
    .pip_install("torch", "transformers", "accelerate", "safetensors")
)


@app.function(
    image=image,
    gpu="H100",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=60 * 20,
)
def run_inference(prompt: str, max_new_tokens: int = 128) -> str:
    tokenizer = build_tokenizer()
    model = build_model()
    config = InferConfig(max_new_tokens=max_new_tokens)
    return generate_text(prompt=prompt, tokenizer=tokenizer, model=model, config=config)


@app.local_entrypoint()
def main(prompt: str = "The capital of France is"):
    print(run_inference.remote(prompt=prompt))
