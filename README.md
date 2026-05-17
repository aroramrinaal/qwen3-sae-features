# qwen3-sae-features

`scripts/infer.py` contains modular inference logic for `Qwen/Qwen3-4B-Base` using Hugging Face Transformers and `model.generate()`.
`modal_app.py` is the Modal remote entry point configured to run inference on an NVIDIA H100 with `huggingface-secret`.
