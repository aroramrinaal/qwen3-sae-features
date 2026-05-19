# qwen3-sae-features

Qwen3 SAE feature-learning experiments on Modal using SAELens.

## Phase 0: Model setup and inference

Completed.

- Downloaded `Qwen/Qwen3-4B-Base` into the Modal Volume `qwen3-sae-features`.
- Stored model files under `/vol/models/Qwen3-4B-Base`.
- Verified Hugging Face Transformers inference with `model.generate()`.
- Verified the model loads from local Modal Volume files instead of redownloading from Hugging Face.

## Phase 1: Residual-stream hook smoke test

Completed.

- Hooked `model.layers.20`.
- Captured layer-20 activations from Qwen3-4B-Base.
- Verified activation shape `[batch, seq, d_model]`.
- Confirmed layer-20 residual stream width is `2560`.

Smoke result:

```text
layer: 20
shape: torch.Size([1, 5, 2560])
dtype: torch.bfloat16
device: cuda:0
token_count: 5
```

## Phase 2: Dataset tokenization

Completed for smoke.

- Used FineWeb-Edu text as the source distribution.
- Tokenized text using the Qwen3 tokenizer.
- Saved the tokenized dataset to the Modal Volume as a Hugging Face Arrow dataset.

Smoke tokenized dataset path:

```text
/vol/datasets/fineweb-edu/tokens/smoke
```

## Phase 3: Cached activation collection

Completed for smoke.

- Used SAELens `CacheActivationsRunner`.
- Loaded Qwen3-4B-Base from the Modal Volume.
- Ran a forward pass over the smoke token dataset.
- Cached activations from `model.layers.20`.
- Saved cached activations as a Hugging Face Arrow dataset.

Smoke activation cache path:

```text
/vol/activations/qwen3-4b-base/layer20/smoke
```

Inspection result:

```text
columns: ['model.layers.20', 'token_ids']
row_count: 16
context_size: 512
d_in: 2560
total_activation_tokens: 8192
dtype: float32
no_nan: true
no_inf: true
all_checks_passed: true
```

## Phase 4: Smoke SAE training

Current phase.

The next step is to train a tiny standard SAE using the cached smoke activation dataset. This is not intended to produce scientifically meaningful features. It is only meant to prove that the cached activations can be loaded by SAELens and used for SAE training.

Smoke SAE config:

```text
config/train_sae_smoke.yaml
```

Expected smoke SAE output:

```text
/vol/saes/qwen3-4b-base/layer20/smoke_standard_exp2
```

The trainer saves:

```text
final_sae/
inference_sae/
metadata.json
```

## Phase 5: 1M-token run

Planned next.

The 1M run repeats the smoke pipeline with larger token and activation caches:

```text
tokenized dataset: /vol/datasets/fineweb-edu/tokens/1m
activation cache:  /vol/activations/qwen3-4b-base/layer20/1m
SAE output:        /vol/saes/qwen3-4b-base/layer20/1m_standard_exp2
```

Configs:

```text
config/tokenize_1m.yaml
config/cache_1m.yaml
config/inspect_1m_activations.yaml
config/train_sae_1m.yaml
config/inspect_sae_1m.yaml
```

Modal runs are config-driven through a single entrypoint:

```bash
.venv/bin/modal run --detach modal_app.py config/tokenize_1m.yaml
.venv/bin/modal run --detach modal_app.py config/cache_1m.yaml
.venv/bin/modal run --detach modal_app.py config/train_sae_1m.yaml
.venv/bin/modal run modal_app.py config/inspect_1m_activations.yaml
.venv/bin/modal run modal_app.py config/inspect_sae_1m.yaml
```

The launcher infers the job from the YAML keys. Tokenization, activation caching, and training are spawned by default for detached/background execution; pass `--wait` if you want those jobs to block and print their full result. Inspect configs always run synchronously so their checks print directly.
