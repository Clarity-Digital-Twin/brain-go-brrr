**TUSZ OSS Summary (SeizureTransformer + Evaluation)**

- Dataset: Temple University Seizure Corpus (TUSZ) v2.0.1 (requires DUA). Place under `data/datasets/tusz/v2.0.1` with official train/dev/test patient-wise splits intact.
- Access: https://www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml

Key open-source references included in this repo
- `reference_repos/SeizureTransformer` (Wu et al., 2025 preprint): time-step level seizure detection model.
  - Paper: https://arxiv.org/abs/2504.00336
  - Docker: `docker pull yujjio/seizure_transformer`
  - Weights: Google Drive link in their README (expects `model.pth` under `wu_2025/src/wu_2025/`).
  - Core architecture (`wu_2025/src/wu_2025/architecture.py`):
    - Input: `(batch, 19, 15360)` → 19 channels, 15360 samples.
    - Sampling rate: 256 Hz → window = 15360 / 256 = 60 s.
    - Encoder: 1D conv stack + pooling, skip connections.
    - Residual CNN stack → Transformer encoder (d_model=512, nhead=4, 8 layers).
    - Decoder: upsample path + skip connections; final `Conv1d(..., out_channels=1, kernel=11)`.
    - Output: time-step probabilities (per-sample seizure probability).
  - Inference utils (`wu_2025/src/wu_2025/utils.py`):
    - Resamples to 256 Hz if needed; normalizes per-channel (z-score).
    - Window size: 15360 (60s); default overlap_ratio=0.0 in dataset, but dataloader concatenates outputs.
    - Post-processing: threshold 0.8; morphological opening/closing (kernel=5); remove events < 2s @ 256 Hz.
    - Note: utils assume unipolar montage and 19 channels; confirm montage mapping for TUSZ records.

Model I/O and preprocessing (from OSS)
- Montage: UNIPOLAR ONLY (reference asserts unipolar montage; convert if needed).
- Channels: 19-channel TCP montage (10–20 system); we use `CHANNELS_TUAB_19` with aliasing (T3→T7, etc.).
- Sampling rate: 256 Hz SSOT (resample as needed).
- Windowing: 60 s windows (15360 samples). Model emits per-sample logits over the window.
- Exact preprocessing order (matches OSS):
  1) Per-channel z-score
  2) Resample to 256 Hz (if needed)
  3) Bandpass 0.5–120 Hz (3rd‑order Butterworth)
  4) Notch 1 Hz (Q=30)
  5) Notch 60 Hz (Q=30)

Evaluation (clinical metrics)
- NEDC evaluator (v6.0.0) is the community baseline for temporal detection:
  - Metrics of interest: FA/24h (false alarms per 24 hours) at fixed sensitivity (e.g., 0.95), TAES (time-aligned event score), and optional ATWV.
  - Export hypotheses per recording in CSV/XML per NEDC spec; run evaluator to obtain official scores.
- This repo includes a light adapter `src/brain_go_brrr/infra/eval/nedc_wrapper.py` that returns proxy metrics (sensitivity, FA/24h, TAES-like F1) to enable local experiments without the toolkit; for publication-grade results, integrate `nedc_eeg_eval` and keep the same outward API.

OSS output format
- The reference project writes TSV annotations using `epilepsy2bids` (`Annotations.saveTsv`). For parity or evaluator compatibility, export our predicted events to NEDC CSV/XML schemas or replicate the TSV and convert.

Implementation guidance adopted here
- SSOT parameters:
  - Sampling rate: 256 Hz
  - Channels: 19-channel montage (`CHANNELS_TUAB_19`)
  - Windowing: 60 s (SeizureTransformer), optional 12 s/1 s (BiLSTM head experiments)
  - Labeling for window-level tasks: positive if seizure fraction ≥ 0.2 within window
- Overlap aggregation: when sliding windows overlap, maintain a `counts` array and average contributions per sample (avoid fixed divide-by-2 assumptions).
- Post-processing: dual-threshold hysteresis (low/high), gap merge (seconds), min event duration (seconds). Start with (0.3, 0.7), 2.0 s gap, 1.0–2.0 s min duration; tune per dev set.

What we won’t do from OSS (by default)
- `sys.path` hacks to import the model; we expose `SeizureTransformerWrapper` that accepts a `build_fn` or uses `wu_2025.SeizureTransformer` if importable.
- Hard-coded threshold at 0.8; we’ll determine operating points from dev set to meet clinical sensitivity targets.
- Morphological ops as the only post-processing; we provide a configurable, auditable post-processing pipeline.

Quick pointers in this repo
- Dataset: `src/brain_go_brrr/infra/data/tusz_detection_dataset.py` (sliding windows, 256 Hz, binary labels).
- Model: `src/brain_go_brrr/infra/ml_models/seizure_transformer_wrapper.py` (safe DI, overlap averaging, AMP-safe; unipolar montage required; applies OSS post-processing by default and returns binary predictions; set `apply_postprocessing=False` for raw probabilities).
- Post: `src/brain_go_brrr/infra/eval/post_processing.py` (hysteresis + merge + min-duration).
- Eval adapter: `src/brain_go_brrr/infra/eval/nedc_wrapper.py` (proxy metrics; swap internals for official NEDC).
- Docs: `TUSZ_IMPLEMENTATION.md`, `TUSZ_ROADMAP.md`, `TUSZ_SPEC.md`.

Open decisions to confirm during integration
- Exact 19-channel mapping for TUSZ records (aliasing to 10–20 canonical names; drop/substitute policy when missing).
- Whether to replicate OSS filters (bandpass + notch) vs. rely on z-score + robust post-processing.
- Target operating points: sensitivities to report (e.g., 0.80/0.85/0.90/0.95) and FA/24h computation via NEDC.

Licensing & attribution
- SeizureTransformer repo includes a LICENSE; respect its terms for any redistributed weights/code.
- Cite Wu et al. (2025) per their README when reporting results using their model/architecture.
