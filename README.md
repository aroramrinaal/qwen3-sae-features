# qwen3-sae-features

`scripts/infer.py` contains modular inference logic for `Qwen/Qwen3-4B-Base` using Hugging Face Transformers and `model.generate()`. It loads model files from the Modal Volume path `/vol/models/Qwen3-4B-Base`.

`scripts/weights.py` contains the model snapshot download and cleanup logic. Running the save command clears the model subfolder in the `qwen3-sae-features` Modal Volume, then writes the safetensors shards plus the tokenizer/config files needed by Transformers.

`modal_app.py` is the single Modal app entry point for both inference and weight saving.
