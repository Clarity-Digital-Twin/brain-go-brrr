# TUSZ Temporal Seizure Detection — Implementation Guide

Status: draft-ready
Owner: seizures/tusz
Scope: End-to-end baseline for temporal seizure detection on TUSZ with NEDC evaluation

---

## 1) Goals and Non-Goals

- Goals
  - Implement a reproducible temporal seizure detection baseline for TUSZ.
  - Support training, inference, and evaluation with NEDC TAES/FA/24h metrics.
  - Provide clean CLIs, configs, and minimal smoke tests.
  - Make the baseline a stepping stone for EEGPT+temporal and other models.
- Non-Goals
  - SOTA performance. We aim for a clear, correct baseline with room to iterate.
  - Full hyperparameter sweeps and large-scale tuning.

## 2) Dataset and Annotations

- Corpus: Temple University Seizure (TUSZ) — use the version available locally per `docs/TUH_CORPUS_GUIDE.md`.
- Input: EDF or TUSZ-native format, read via MNE. Standardize channel naming and montage.
- Sampling rate: Preserve native or resample to a standard rate (e.g., 256 Hz) for consistency; document chosen value in config.
- Annotations: Event-level seizure onsets/offsets. Each recording contains 0–N seizure segments.
  - Convert annotation channels (e.g., `seizure events`, `seiz`, `SZ`) into structured spans `[start, end]` in seconds.
  - Use a consistent label space: binary {nonseizure=0, seizure=1} for the initial baseline.

## 3) Preprocessing Pipeline

- Loading
  - Read each recording with MNE, harmonize channel names, drop or map unknowns.
  - Optionally map to a standard montage; retain a defined subset of channels.
  - Ensure stable dtype and scaling to microvolts or volts; document `volt_norm` if applied.
- Filtering
  - Band-pass (e.g., 0.5–70 Hz) and notch (50/60 Hz) per config; allow bypass for benchmarking raw.
- Resampling
  - Resample to `target_fs` (e.g., 256 Hz). Store original `fs` in metadata for correct time conversion when exporting hypotheses.
- Normalization
  - Options: per-recording z-score, robust scaler, or none. Document choice in config.
- Windowing
  - Fixed windows (e.g., 2.0 s) with stride (e.g., 0.5 s) for training and inference.
  - Labeling rule for a window: positive if overlap with any seizure span exceeds `min_overlap` (e.g., 0.5 s) or fraction threshold (e.g., 25%).
  - Persist `window_length`, `window_stride`, and `label_overlap_policy` in metadata for reproducibility.

## 4) Baselines and Model Wrappers

- Baseline A: SeizureTransformer (reference_repos/SeizureTransformer)
  - Wrap dataset -> dataloader -> model init -> train loop -> inference.
  - If the reference repo provides a training harness, either:
    1) Call it via a thin adapter, or
    2) Re-implement a minimal trainer within our code for consistency and logging.
  - Preferred: local wrapper class `SeizureTransformerWrapper` under `src/brain_go_brrr/infra/ml_models/` that hides third-party details.
- Baseline B (optional next): EEGPT embeddings + temporal head
  - Freeze EEGPT; produce per-window embeddings; train a small temporal classifier (e.g., 1D-Conv/GRU/Transformer encoder) on top.
  - Keep this path off by default until Baseline A is solid.

### Wrapper Responsibilities

- Config handling: hyperparameters, optimizer, scheduler, and seed control.
- Model instantiation: correct input shape (channels × time), positional encodings if needed.
- Loss: binary cross-entropy (logits) or focal loss; class weighting to address imbalance.
- Training: gradient clipping, AMP optional, checkpointing (model + optimizer + scheduler + epoch).
- Inference: produce per-window probabilities aligned to recording time.
- Logging: training/validation metrics, class histograms, ROC-AUC/PR-AUC where relevant.

## 5) Postprocessing to Temporal Events

- Inputs: per-window probabilities `p(t)` and window time alignment.
- Thresholding: select `p >= theta` to mark positive windows (default theta=0.5; allow tuning).
- Merge: contiguous or near-contiguous positive windows are merged into events with `max_gap` tolerance (e.g., 0.5 s).
- Min duration: drop events shorter than `min_event_dur` (e.g., 1.0 s) to reduce fragmentation.
- Hysteresis (optional): use high/low thresholds to stabilize on/off transitions.
- Output: event hypotheses with `[start_time, end_time, score]` in seconds.

## 6) NEDC Evaluation Integration

- Reference: `reference_repos/nedc_eeg_eval_v6.0.0`.
- Export formats: CSV and/or XML per NEDC expectations. Include per-file unique identifiers consistent with the evaluator’s mapping.
- Matching policy: TAES/OVLP/DPALIGN as provided by the library; ensure correct tolerance and time base.
- Metrics: TAES, FA/24h, ATWV (if available), plus summary tables by class (binary here) and by recording.
- CLI: `tusz-eval` takes `--hyp-dir`, `--ref-dir`, `--output metrics.json` and prints a concise table.

## 7) Command-Line Interfaces (CLIs)

All CLIs use hyphenated flags (not underscores) to satisfy CI argument checks.

- Data Prep: `scripts/tusz/prepare_tusz.py`
  - Args: `--root`, `--output-cache`, `--target-fs`, `--channels`, `--bandpass`, `--notch`, `--normalize`.
  - Output: cached tensors and metadata for train/val/test splits.
- Train: `scripts/tusz/train_seizure_transformer.py`
  - Args: `--cache-dir`, `--run-dir`, `--epochs`, `--batch-size`, `--lr`, `--scheduler`, `--grad-clip`, `--seed`.
  - Behavior: checkpointing to `--run-dir/checkpoints/`. Log to `--run-dir/logs/`.
- Infer: `scripts/tusz/infer.py`
  - Args: `--cache-dir`, `--checkpoint`, `--run-dir`, `--threshold`, `--window-config`, `--output-hyp`.
  - Output: hypotheses per-recording in `--output-hyp` (CSV/XML).
- Eval: `scripts/tusz/eval_nedc.py`
  - Args: `--hyp-dir`, `--ref-dir`, `--output`, `--metrics`.
  - Behavior: Runs NEDC evaluator; writes `metrics.json` and prints a summary.

## 8) Configs and Defaults

- Location: `experiments/tusz/configs/`
- Example `baseline.yaml`:
  - Data: `target_fs=256`, `window=2.0s`, `stride=0.5s`, `min_overlap=0.5s`, `min_event_dur=1.0s`.
  - Model: SeizureTransformer base dims and layers (small to medium).
  - Optim: AdamW, `lr=3e-4`, `weight_decay=1e-2`, OneCycle or Cosine scheduler.
  - Train: `epochs=30`, `batch_size=64`, `grad_clip=1.0`, mixed precision off by default.
  - Eval: `threshold=0.5`, `max_gap=0.5s`.

## 9) Checkpointing and Resume

- Checkpoint: `{epoch}-{val_loss:.4f}.ckpt` with state dicts for model, optimizer, scheduler, and RNG seeds.
- Resume: detect the latest checkpoint in `--run-dir/checkpoints/` when `--resume` is passed; restore scheduler step state to avoid OneCycle overstep.

## 10) Logging and Metrics

- Per-epoch: loss, accuracy, ROC-AUC, PR-AUC (optional), class distribution.
- Early signal: non-zero recall on seizure class and rise in validation F1.
- Inference debug: dump a per-recording PDF/PNG with probabilities and predicted spans overlayed (optional).

## 11) Tests

- Unit tests: window labeling, merge/hysteresis logic, CSV/XML export formatting.
- Integration: tiny synthetic dataset (few recordings), run train(1–2 epochs) → infer → eval; check that metrics.json schema is correct and FA/24h is finite.

## 12) Directory Layout

```
docs/
  tusz/
    TUSZ_SPEC.md
    TUSZ_IMPLEMENTATION.md
    archive/
      TUSZ_ARCHITECTURE_DECISION.md
      TUSZ_BRAINSTORMING.md
      TUSZ_TEMPORAL_IMPLEMENTATION.md
      TUSZ_WRAPPER_INTEGRATION_PLAN.md
experiments/
  tusz/
    configs/baseline.yaml
    scripts/
      prepare_tusz.py
      train_seizure_transformer.py
      infer.py
      eval_nedc.py
src/brain_go_brrr/
  infra/data/tusz_dataset.py
  infra/ml_models/seizure_transformer_wrapper.py
  infra/eval/nedc_adapter.py
  utils/postproc/temporal_merge.py
```

## 13) Milestones

1. Dataset adapter + windowing + labeling; cached splits.
2. SeizureTransformer wrapper + training loop (resume-safe, clipped grads).
3. Inference + postprocessing merge; hypothesis export.
4. NEDC evaluator adapter; metrics.json + printed summary.
5. E2E tiny test; docs and quickstart.

## 14) Risks and Mitigations

- Channel/montage inconsistencies → enforce a strict channel set and document drop/map policy.
- Time alignment errors (resample drift) → keep original `fs` and use precise sample-to-time conversion.
- Class imbalance → weighted loss, focal loss option, balanced sampling.
- Over-threshold fragmentation → merge with `max_gap`, enforce `min_event_dur`, consider hysteresis.
- NEDC format mismatch → snapshot a known-good example and schema-test the export before running large evals.

## 15) Quickstart

```
# 1) Prepare cached data
python scripts/tusz/prepare_tusz.py \
  --root /data/tusz \
  --output-cache ./outputs/tusz/cache \
  --target-fs 256 --channels "tusz_small" --bandpass 0.5 70 --notch 60 --normalize zscore

# 2) Train baseline
python scripts/tusz/train_seizure_transformer.py \
  --cache-dir ./outputs/tusz/cache \
  --run-dir ./outputs/tusz/runs/st_baseline \
  --epochs 30 --batch-size 64 --lr 3e-4 --scheduler cosine --grad-clip 1.0 --seed 42

# 3) Infer hypotheses
python scripts/tusz/infer.py \
  --cache-dir ./outputs/tusz/cache \
  --checkpoint ./outputs/tusz/runs/st_baseline/checkpoints/latest.ckpt \
  --run-dir ./outputs/tusz/runs/st_baseline \
  --threshold 0.5 --output-hyp ./outputs/tusz/hyp

# 4) Evaluate with NEDC
python scripts/tusz/eval_nedc.py \
  --hyp-dir ./outputs/tusz/hyp \
  --ref-dir /data/tusz/annotations \
  --output ./outputs/tusz/metrics.json
```

## 16) References

- NEDC Evaluator: `reference_repos/nedc_eeg_eval_v6.0.0`
- SeizureTransformer: `reference_repos/SeizureTransformer`
- Metrics background: `literature/markdown/evaluation-metrics/picone-2021-objective-evaluation-metrics.md`

