# qwen3-sae-features

Sparse autoencoder feature experiments for `Qwen/Qwen3-4B-Base` layer 20 on Modal.

Run jobs through the config-driven remote entrypoint in [modal_app.py](modal_app.py):

```bash
.venv/bin/modal run --detach modal_app.py --config <config.yaml>
```

Use `--wait` only for short inspection or analysis jobs.

## Phase 0: Model Setup

Completed. Qwen3-4B-Base is cached on the Modal volume and reused by later jobs instead of being downloaded every run.

Volume path:

```text
/vol/models/Qwen3-4B-Base
```

## Phase 1: Layer-20 Activation Hook

Completed. The project hooks `model.layers.20`, whose residual-stream width is `2560`.

This is the activation space the SAE learns over.

## Phase 2: Tokenized Dataset

Completed for `smoke`, `1m`, and `50m`. FineWeb-Edu text is tokenized with the Qwen3 tokenizer.

Volume paths:

```text
/vol/datasets/fineweb-edu/tokens/smoke
/vol/datasets/fineweb-edu/tokens/1m
/vol/datasets/fineweb-edu/tokens/50m
```

## Phase 3: Cached Activations

Completed for `smoke`, `1m`, and `50m`. Layer-20 residual activations are stored as Arrow datasets so SAE training/dashboarding can run without repeatedly running the base model.

Volume paths:

```text
/vol/activations/qwen3-4b-base/layer20/smoke
/vol/activations/qwen3-4b-base/layer20/1m
/vol/activations/qwen3-4b-base/layer20/50m
```

## Phase 4: SAE Training

Completed for the main run: `50m_standard_exp4_l1_5`.

The main SAE has `d_in=2560`, expansion factor `4`, and `d_sae=10240`.

Volume path:

```text
/vol/saes/qwen3-4b-base/layer20/50m_standard_exp4_l1_5
```

## Phase 5: Feature Dashboard

Completed. The dashboard step reads cached activations, runs the SAE encoder, and stores top activating token windows per feature.

Main dashboard path:

```text
/vol/features/qwen3-4b-base/layer20/50m_standard_exp4_l1_5/dashboard
```

## Phase 6: Autointerp Labels

Completed. DeepSeek labeled `9418` SAE features with `0` failed batches.

Main autointerp path:

```text
/vol/features/qwen3-4b-base/layer20/50m_standard_exp4_l1_5/autointerp
```

## Phase 7: Label Analysis

Completed. Labels were grouped into confidence and rough semantic buckets for later feature selection.

Useful outputs:

```text
label_analysis/summary.md
feature_groups/high_confidence.jsonl
feature_groups/low_confidence_or_unclear.jsonl
feature_groups/boring_formatting.jsonl
feature_groups/topic_semantic.jsonl
feature_groups/style_instruction.jsonl
feature_groups/candidate_steering_features.jsonl
```

## Phase 8: Steering Smoke Test

Started. The first causal steering tests add SAE decoder directions to the layer-20 hidden state during generation and compare base vs steered outputs.

Initial steering runs:

```text
feature 3311: cooking instructions
/vol/features/qwen3-4b-base/layer20/50m_standard_exp4_l1_5/steering/feature_3311

selected 5: cooking, exercise, biological adaptation, self-help feel, biblical moral language
/vol/features/qwen3-4b-base/layer20/50m_standard_exp4_l1_5/steering/selected_5

3311 sweep: alpha grid across last-token/all-positions and raw/normalized decoder directions
/vol/features/qwen3-4b-base/layer20/50m_standard_exp4_l1_5/steering/feature_3311_sweep_*

7771 sweep: biblical moral language, normalized decoder direction, last-token/all-positions
/vol/features/qwen3-4b-base/layer20/50m_standard_exp4_l1_5/steering/feature_7771_sweep_*_norm

703 sweep: exercise and physical activity, normalized decoder direction, last-token/all-positions
/vol/features/qwen3-4b-base/layer20/50m_standard_exp4_l1_5/steering/feature_703_sweep_*_norm
```

## Volume Map

Modal volume: `qwen3-sae-features`

```text
qwen3-sae-features/
  models/
    Qwen3-4B-Base/
      config.json
      generation_config.json
      tokenizer_config.json
      tokenizer.json
      vocab.json
      merges.txt
      model.safetensors.index.json
      model-00001-of-00003.safetensors
      model-00002-of-00003.safetensors
      model-00003-of-00003.safetensors

  datasets/
    fineweb-edu/
      tokens/
        smoke/
          data-*.arrow
          dataset_info.json
          state.json
        1m/
          data-*.arrow
          dataset_info.json
          state.json
        50m/
          data-*.arrow
          dataset_info.json
          state.json

  activations/
    qwen3-4b-base/
      layer20/
        smoke/
          data-*.arrow
          dataset_info.json
          state.json
        1m/
          data-*.arrow
          dataset_info.json
          state.json
        50m/
          data-*.arrow
          dataset_info.json
          state.json

  saes/
    qwen3-4b-base/
      layer20/
        smoke_standard_exp2/
          final_sae/
          inference_sae/
          checkpoints/
          output/
          metadata.json
        1m_standard_exp2/
          final_sae/
          inference_sae/
          checkpoints/
          output/
          metadata.json
        50m_standard_exp4_l1_1/
          final_sae/
          inference_sae/
          checkpoints/
          output/
          metadata.json
        50m_standard_exp4_l1_2/
          final_sae/
          inference_sae/
          checkpoints/
          output/
          metadata.json
        50m_standard_exp4_l1_5/
          final_sae/
          inference_sae/
          checkpoints/
          output/
          metadata.json

  features/
    qwen3-4b-base/
      layer20/
        50m_standard_exp4_l1_5/
          smoke_dashboard/
            top_activations.jsonl
            feature_summary.json
            preview.md
          medium_dashboard/
            top_activations.jsonl
            feature_summary.json
            preview.md
          dashboard/
            top_activations.jsonl
            feature_summary.json
            preview.md
          autointerp_smoke/
            batches/
            labels.jsonl
            label_summary.md
            run_summary.json
            failed_batches.jsonl
          autointerp/
            batches/
            labels.jsonl
            label_summary.md
            run_summary.json
            failed_batches.jsonl
            label_analysis/
              summary.md
              summary.json
              confidence_histogram.csv
              label_frequency.csv
            feature_groups/
              high_confidence.jsonl
              low_confidence_or_unclear.jsonl
              boring_formatting.jsonl
              topic_semantic.jsonl
              style_instruction.jsonl
              candidate_steering_features.jsonl
          steering/
            feature_3311/
              generations.jsonl
              summary.md
              summary.json
            selected_5/
              generations.jsonl
              summary.md
              summary.json
            feature_3311_sweep_last_raw/
              generations.jsonl
              summary.md
              summary.json
              comparison_table.md
              comparison_table.csv
            feature_3311_sweep_last_norm/
              generations.jsonl
              summary.md
              summary.json
              comparison_table.md
              comparison_table.csv
            feature_3311_sweep_all_raw/
              generations.jsonl
              summary.md
              summary.json
              comparison_table.md
              comparison_table.csv
            feature_3311_sweep_all_norm/
              generations.jsonl
              summary.md
              summary.json
              comparison_table.md
              comparison_table.csv
            feature_7771_sweep_last_norm/
              generations.jsonl
              summary.md
              summary.json
              comparison_table.md
              comparison_table.csv
            feature_7771_sweep_all_norm/
              generations.jsonl
              summary.md
              summary.json
              comparison_table.md
              comparison_table.csv
            feature_703_sweep_last_norm/
              generations.jsonl
              summary.md
              summary.json
              comparison_table.md
              comparison_table.csv
            feature_703_sweep_all_norm/
              generations.jsonl
              summary.md
              summary.json
              comparison_table.md
              comparison_table.csv
```
