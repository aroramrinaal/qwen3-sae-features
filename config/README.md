# Config layout

YAML configs are grouped by workflow:

- `tokenize/`: FineWeb-Edu tokenization jobs
- `cache/`: activation cache collection jobs
- `inspect/activations/`: cached activation inspection jobs
- `inspect/sae/`: trained SAE artifact inspection jobs
- `train/sae/`: SAE training jobs
- `feature_dashboard/`: feature dashboard generation jobs

Run configs with `modal_app.py`, for example:

```bash
.venv/bin/modal run --detach modal_app.py --config config/tokenize/1m.yaml
.venv/bin/modal run modal_app.py --config config/inspect/activations/1m.yaml --wait
MODAL_GPU=H100 .venv/bin/modal run --detach modal_app.py --config config/train/sae/50m_exp4_l1_5.yaml
```
