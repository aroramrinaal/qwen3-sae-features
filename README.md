# qwen3-sae-features

`scripts/infer.py` contains modular inference logic for `Qwen/Qwen3-4B-Base` using Hugging Face Transformers and `model.generate()`. It loads model files from the Modal Volume path `/vol/models/Qwen3-4B-Base`.

`scripts/weights.py` contains the model snapshot download and cleanup logic. Running the save command clears the model subfolder in the `qwen3-sae-features` Modal Volume, then writes the safetensors shards plus the tokenizer/config files needed by Transformers.

The `Qwen3-4B-Base` model weights are stored in the Modal Volume named `qwen3-sae-features`, under the `models/Qwen3-4B-Base` subfolder. That folder contains the `.safetensors` weight shards plus the necessary Transformers config and tokenizer files for local-volume inference.

`modal_app.py` is the single Modal app entry point for both inference and weight saving.


step-by-step breakdown:

phase 0: environment setup and model verification
- loading weights and running inference on modal.com's cloud infrastructure
- saving Qwen/Qwen3-4B-Base model weights to a Modal Volume at this location; volume name: "qwen3-sae-features" , under the `models/Qwen3-4B-Base` subfolder. That folder contains the `.safetensors` weight shards plus the necessary Transformers config and tokenizer files for local-volume inference.

phase 1: 
- smoke test by hooking one residual stream layer, let's say layer 20 since this qwent model has toal of 36 layers. Capturing activations from model.model.layers[20]
- For Qwen-style huggingFace models, the layer output hidden states are usually the residual stream after that block.