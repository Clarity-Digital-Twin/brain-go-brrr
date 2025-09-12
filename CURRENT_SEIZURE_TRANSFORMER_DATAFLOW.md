# 🔴 CURRENT SEIZURE TRANSFORMER DATAFLOW (What We Built)

Status: runnable path exists via experiments, but diverges from OSS parity in training preprocessing and supervision. Last audited: 2025-09-12.

Purpose: snapshot the exact behavior of the current codebase; highlight divergences; list concrete fixes.

—

## Current Implementation Path

1) Model architecture
- External reference: `reference_repos/SeizureTransformer/wu_2025/src/wu_2025/architecture.py` (≈41M params).
- Library wrapper: `src/brain_go_brrr/infra/ml_models/seizure_transformer_wrapper.py` holds I/O, preprocessing, windowing, post-processing; it does NOT define the architecture.
- Import policy:
  - Wrapper prefers dependency injection (pass `model` or `build_fn`).
  - If neither is provided, it attempts `from wu_2025.architecture import SeizureTransformer` as a soft fallback (no sys.path hacks).
  - Experiments import `wu_2025` directly.

2) Wrapper behavior (library path)
- Inputs: `(C, T)` Volts at `fs` Hz (default `C=19`, `fs=256`); unipolar montage required (not enforced).
- Preprocessing (matches OSS order): z‑score → resample to 256 (if needed) → 0.5–120 Hz bandpass (order=3) → notch 1 Hz and 60 Hz (Q=30).
- Windowing: 60 s (15360) with `overlap_ratio=0.0` by default; concatenates window outputs and trims to `T`.
- Post-processing (inside wrapper): threshold 0.8 → morphological opening (kernel=5) → closing (kernel=5) → drop events < 2 s; returns binary array. Set `apply_postprocessing=False` for raw probabilities.
- AdvancedPostProcessor (separate module): offers hysteresis + gap merge + min‑duration and emits (start, end, confidence) events; not used by wrapper unless called externally.

3) Dataset: `src/brain_go_brrr/infra/data/tusz_detection_dataset.py`
- Discovers EDF and sidecar `.tse`/`.csv` annotations.
- Channel policy: standardizes names via aliases (e.g., T3→T7), picks AVAILABLE target channels, then PADS WITH ZEROS to ensure exactly 19 channels (`__getitem__` method).
- Resampling: uses `mne.Raw.resample(cfg.fs)` when needed.
- Labels: window‑level binary label by fraction of seizure time in the window (default `positive_fraction=0.2`). Not per‑timestep labels.
- TSE parsing: permissive — `_parse_tse` accepts ANY line with two numeric fields (even without "seiz" label). This WILL include non‑seizure spans and cause false positives.
- Memory limiting: DOES support `max_windows` parameter to limit dataset size (`_build_index` method).

4) Training script: `experiments/seizure_transformer/train_tusz.py`
- Imports `wu_2025.SeizureTransformer` directly (local editable install required).
- Feeds dataset windows directly to the model (bypasses wrapper preprocessing); OSS filters (bandpass + notch) are NOT applied here.
- Supervision: expands each window's single label to all 15360 time steps and uses BCE per timestep (`train_epoch` function). This diverges from true per‑timestep segmentation labels.
- Technical notes:
  - Successfully uses `max_windows=10000` for train and `max_windows=5000` for validation.
  - Model expects 19 channels; dataset DOES pad to guarantee exactly 19 channels.

5) Post‑processing
- Wrapper: OSS‑matching binary morphology as noted above.
- Event pipeline: `AdvancedPostProcessor` (hysteresis/gap/min‑dur) available for event lists and clinical proxy metrics; not applied by the wrapper by default.

—

## Actual Data Flows

Training path (experiments)
- EDF → TUSZDetectionDataset (channel aliasing; resample to cfg.fs; window‑level labels) → DataLoader → wu_2025.SeizureTransformer (no bandpass/notch preprocessing) → per‑timestep logits → BCE against window label expanded across all timesteps → AUROC computed from mean window probability (in script).

Inference path (library wrapper)
- EDF/array (Volts, unipolar) → z‑score → (resample→) bandpass (0.5–120) → notch(1,60) → 60 s windowing → model (DI or wu_2025 fallback) → per‑sample probs → threshold+open/close+min‑dur → binary predictions.

—

## Key Facts (audited 2025-09-12 with function-level verification)
- Wrapper preprocessing and post‑processing parameters match the OSS reference (threshold 0.8; kernel size 5; min duration 2 s; notch Q=30).
- Dataset DOES pad missing channels with zeros to guarantee exactly 19 channels (`TUSZDetectionDataset.__getitem__`).
- TSE parsing is dangerously permissive - `_parse_tse` accepts ANY 2-field line even without "seiz" label; WILL cause false positives.
- Experiments import `wu_2025` directly and bypass wrapper preprocessing; training supervision differs from OSS segmentation.
- Proxy clinical metrics available (`NEDCClinicalEvaluator` in `infra/eval/nedc_wrapper.py`), but experiments do not integrate NEDC scoring.
- `max_windows` IS implemented and working in `_build_index` method (not a bug).

—

## Gaps vs OSS Parity
- Preprocessing in training: filters not applied (wrapper not used in training path).
- Supervision: window‑level labels expanded to timesteps instead of true per‑timestep labels.
- TSE parsing: accepts ANY 2-field line (not just seizures); introduces massive label noise.
- Architecture sourcing: experiments rely on external `wu_2025` import (local install), not an internal builder.
- Metrics: experiments compute AUROC only; no FA/24h or official TAES scoring.

—

## Required Fixes (to reach parity)
- Apply wrapper (or equivalent preprocessing fn) during training so bandpass/notch are applied.
- Use per‑timestep labels (segmentation) or a dataset that yields timestep masks instead of expanding window labels.
- FIX CRITICAL BUG: Tighten `_parse_tse` function to ONLY accept lines with "seiz" labels (currently accepts ALL 2-field lines!).
- Enforce unipolar montage checking (currently just assumes it).
- Provide an internal model builder (or vendor the architecture with a compatible license) to avoid direct `wu_2025` imports in experiments.
- Wire `NEDCClinicalEvaluator` metrics into evaluation scripts.

—

## Notes
- This document reflects the current code paths exactly (no code edits); it distinguishes library‑grade wrapper inference from experiment‑grade training.
