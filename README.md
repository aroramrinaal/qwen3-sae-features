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

Completed for the 50M `exp4_l1_5` run.

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

## Phase 5: Feature dashboarding before autointerp

Current phase.

Do not start with LLM labels first. The LLM can only label the evidence you give
it, so the first interpretability milestone is: can we produce clean top
activating examples for SAE features?

Mental model:

```text
cached layer-20 residual vector, shape [2560]
        -> SAE encoder
sparse feature activations, shape [10240]
        -> stream top-k only
decoded token windows with the max token bracketed
```

The base Qwen model does not need to run again for this step. We use:

```text
tokenizer:    /vol/models/Qwen3-4B-Base
activations:  /vol/activations/qwen3-4b-base/layer20/50m
SAE:          /vol/saes/qwen3-4b-base/layer20/50m_standard_exp4_l1_5/final_sae
```

Feature dashboard outputs:

```text
/vol/features/qwen3-4b-base/layer20/50m_standard_exp4_l1_5/smoke_dashboard/top_activations.jsonl
/vol/features/qwen3-4b-base/layer20/50m_standard_exp4_l1_5/smoke_dashboard/feature_summary.json
/vol/features/qwen3-4b-base/layer20/50m_standard_exp4_l1_5/smoke_dashboard/preview.md
```

Each `top_activations.jsonl` row is one feature:

```json
{
  "feature_id": 3172,
  "max_activation": 18.2,
  "top_examples": [
    {
      "rank": 1,
      "activation": 18.2,
      "row_index": 123,
      "token_position": 45,
      "text": "... import torch.nn as [[nn]] ..."
    }
  ]
}
```

Smoke dashboard run:

```bash
.venv/bin/modal run --detach modal_app.py --config config/feature_dashboard_smoke_50m_exp4_l1_5.yaml
```

Full dashboard run:

```bash
MODAL_GPU=H100 .venv/bin/modal run --detach modal_app.py --config config/feature_dashboard_50m_exp4_l1_5.yaml
```

What the smoke config proves:

- the trained SAE loads from `final_sae`
- cached `.arrow` activations load with the expected hook column
- activations can be encoded into SAE feature activations
- top activating token positions can be found without saving all `50M x 10240` activations
- token windows can be decoded from the local Qwen tokenizer

Only after this looks clean should an `autointerp.py` step read
`top_activations.jsonl` and ask an LLM for labels.
