# Lymph-node triage assistant — CPU-only, offline after setup

`OpenVINO Optimized` · `CPU-Only` · `License: Apache-2.0 (code)` · `Status: RUO (Research Use Only)`

[![Weights on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Weights-yellow)](https://huggingface.co/DeptCreator/lymph-node-triage-cpu)

Second-read triage aid for breast lymph-node metastasis screening on H&E slides.
Frozen histology encoder + linear probe, OpenVINO inference on commodity CPU, no GPU, no cloud.
**Adjunct only: the verdict is always made by the pathologist.**

## Quickstart

```bash
pip install -r requirements.txt
python download_weights.py   # one-time fetch (~110 MB) into ./models
python app.py                # local demo at http://127.0.0.1:7860
# or batch:
python run_cli.py assets/demo_slide.png --out heatmap_out.png
```

Weights mirror: https://huggingface.co/DeptCreator/lymph-node-triage-cpu (same files + full model card
with metrics, limitations and license split). `download_weights.py` fetches from the GitHub Release;
the HF repo is an identical mirror for those who prefer it.

## Measured results (frozen protocol, not marketing)

| Metric | Value | Source |
|---|---|---|
| Patch AUC (PCam test 1k) | 0.937 [95% CI 0.922–0.951] | held-out, single eval |
| Sensitivity / specificity @ thr 0.0962 | 0.947 [0.925–0.968] / 0.671 | threshold tuned on val, reported on test |
| ITC/micro regions highlighted | 35/35 | Camelyon16 lesion polygons |
| Latency (Intel i5 CPU, OpenVINO FP32) | ~54 ms/tile | measured, parity vs torch 2.4e-07 |

## Scientific integrity & limitations (read before use)

- This is a **triage/region-screening aid**, not autonomous diagnostics. Heatmap peak lands exactly on the
  lesion in only 11/35 cases — it highlights *zones*, it does not point at cells.
- Clean-slide false-positive quantification is an **open validation task**; any flagged zone needs tissue review.
- Input scale for arbitrary uploads is unverified (validated on 96px@10x-class patches).
- Backbone weights are a community mirror under **GPLv3** (see WEIGHTS.md) — research use; commercial
  deployment needs original-author weights + legal review. Our probe weights and all code are Apache-2.0.

## Layout

- `src/tiling.py` — Otsu tissue mask + tile grid · `src/inference.py` — OpenVINO runner + probe
- `src/filters.py` — singleton suppression · `src/heatmap.py` — overlay render
- `app.py` — Gradio demo · `run_cli.py` — batch CLI · `configs/thresholds.json` — frozen operating point
