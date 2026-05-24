# Config Layout

YAML configs are grouped by workflow and run through the config-driven Modal entrypoint:

```bash
.venv/bin/modal run --detach modal_app.py --config <config.yaml>
```

Use `--wait` for short inspect/analysis jobs.

## Pipeline

- `tokenize/`: FineWeb-Edu tokenization for `smoke`, `1m`, and `50m`.
- `cache/`: layer-20 activation cache collection.
- `inspect/activations/`: cached activation sanity checks.
- `train/sae/`: SAE training configs.
- `inspect/sae/`: trained SAE artifact inspection.
- `feature_dashboard/`: top activating token-window dashboards.
- `autointerp/`: DeepSeek label generation plus post-label analysis.
- `steer/`: SAE decoder-direction steering runs.

## Steering

Steering configs are split by intent:

```text
steer/
  single/
    feature_3311.yaml
  sets/
    selected_5.yaml
  sweeps/
    3311_last_norm.yaml
    3311_last_raw.yaml
    3311_all_norm.yaml
    3311_all_raw.yaml
    703_last_norm.yaml
    703_all_norm.yaml
    7771_last_norm.yaml
    7771_all_norm.yaml
    1179_last_norm.yaml
    4126_last_norm.yaml
    6227_last_norm.yaml
```

Naming convention:

```text
<feature_id>_<position_mode>_<direction_scale>.yaml
```

- `last`: injects the SAE decoder direction at the current/new token position.
- `all`: injects the direction at all sequence positions.
- `norm`: normalizes the decoder direction before applying alpha.
- `raw`: uses the decoder direction at its saved norm.

Current high-signal steering outputs live on the Modal volume under:

```text
/vol/features/qwen3-4b-base/layer20/50m_standard_exp4_l1_5/steering
```

Useful current runs:

```text
feature_3311_sweep_*          cooking instructions
feature_703_sweep_*_norm      exercise / physical activity
feature_7771_sweep_*_norm     biblical moral language
feature_1179_sweep_last_norm  University of Pennsylvania, weak/mixed
feature_4126_sweep_last_norm  neurological/progressive disease, mixed
feature_6227_sweep_last_norm  legal disclaimer / applicable law
```
