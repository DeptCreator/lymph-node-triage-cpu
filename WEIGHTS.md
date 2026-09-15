# Weights provenance (read this before shipping anything)

## What is what
- **Backbone `ctranspath_ov.{xml,bin}`** — CTransPath histology encoder (Swin-Tiny hybrid, 27.5M params),
  sourced from the community mirror `1aurent/swin_tiny_patch4_window7_224.CTransPath`.
  License: **GPLv3** (original repo terms). Status here: **measurement/research use only**.
  NOT cleared for commercial product use — that requires original-author weights + legal review.
  OpenVINO IR conversion is numerically exact vs torch (max abs diff 2.4e-07, measured).
- **Probe `probe_m2.pkl`** — logistic regression (768→1) trained by us on public CC0 PatchCamelyon patches.
  Ours, **Apache-2.0**, same as this repo's code. Operating point thr 0.0962 (tuned on validation,
  reported on held-out test — see README table).

## Why this split exists
Code we wrote → Apache-2.0. Weights others trained → their terms travel with the files.
Anything in `./models` not listed above does not belong to this project.
