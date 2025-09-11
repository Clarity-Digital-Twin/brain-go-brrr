# TUEV Training Investigation: Why We’re Stuck at 16.67% BAC

## Executive Summary

Root cause: task mismatch. The EEGPT reference trains a multi-class event-segment classifier on event-centered 5 s segments at 200 Hz; our pipeline trains a sliding-window detector over entire recordings (99.58% background). This guarantees collapse to background (balanced accuracy = 1/6).

## What EEGPT Actually Does (Verified)

- Event-only segments at 200 Hz, 5 seconds each:
  - reference_repos/EEGPT/downstream_tueg/dataset_maker/make_TUEV.py
    - readEDF: bandpass 0.1–75 Hz, notch 50 Hz, resample 200 Hz; uses referential “-REF” channels.
    - BuildEvents: creates fixed 5 s (1000-sample) event-centered segments; writes pickles under processed_{train,eval,test}.
  - reference_repos/EEGPT/downstream_tueg/utils.py: TUEVLoader loads those event pickles.
- No bipolar montage used in training:
  - make_TUEV.py defines convert_signals (bipolar helper) but the call is commented out.
- 23→20 channel mapping via conv:
  - downstream_tueg/Modules/models/EEGPT_mcae_finetune_change_tuev.py implements a learned 23→20 conv stack.
  - downstream_tueg/run_class_finetuning_EEGPT_change_tuev.py sets use_chan_conv=True and img_size=[20,1000] (i.e., 20×1000 input).
- Loss and schedule match paper:
  - Unweighted LabelSmoothingCrossEntropy(smoothing=0.1).
  - downstream_tueg/finetune_TUEV_EEGPT.sh: lr=5e-4, weight_decay=0.05, warmup_epochs=5, epochs=30, layer_decay=0.65, batch_size=400 (distributed).

## What Our Repo Does Now (Implemented)

- Event-only pipeline (no sliding windows):
  - Extractor: `src/brain_go_brrr/infra/preprocessing/tuev_event_extractor.py` (0.1–75 Hz + 50 Hz notch; 200 Hz; REF channels; −2..+3 s to (23,1000)).
  - Dataset: `src/brain_go_brrr/infra/data/tuev_event_dataset.py` (parses TUEV `_ch000.lab`, falls back to `.rec.lab`; caches .pt + index with fs=200, duration=5s, channels=23, samples=1000, unit='V', segment_type='event'). Subject 80/20 split fallback when pre‑split dirs absent.
- Mapper + training (SSOT):
  - Mapper (23→20): `src/brain_go_brrr/infra/ml_models/channel_mapper.py`.
  - EEGPT parity support (native 1000 via patch_stride): `src/brain_go_brrr/infra/ml_models/eegpt_architecture.py` (+ wrapper `model_kwargs`).
  - Trainer (single): `experiments/eegpt_linear_probe/train_tuev_events.py` with `--use_parity` (native 1000) or default padding 1000→1024.
  - Loss/schedule/optimizer: label_smoothing=0.1; warmup=5 + cosine; layer_decay=0.65; lr=5e-4; wd=0.05; effective batch≈400 via accumulation.

## Why We’re At 0.1667 BAC

- Training on 99.58% background windows encourages a trivial solution (predict background).
- Balanced accuracy on a 6-class task = 1/6 when predicting a single class.
- Loss converges without improving BAC (observed), consistent with collapsed predictions.

## Key Divergences (Resolved)

- Task mismatch → Fixed: event-only segments implemented.
- Data distribution → Fixed: only event segments cached; metadata enforced.
- Input dimensions → Fixed: 5 s @ 200 Hz (23×1000) end-to-end; EEGPT supports native 1000 via stride (or padding fallback).
- Hyperparameters → Fixed: label_smoothing=0.1; warmup=5; layer_decay=0.65; cosine schedule; effective batch≈400 via accumulation.

## Remediation Summary (Strict Parity)

- SSOT in `src/`; single thin trainer in `experiments/`.
- Event extractor + dataset implemented; parser supports TUEV `_ch000.lab`.
- Mapper (23→20) + EEGPT parity stride; trainer wires `--use_parity` to `time_steps=1000`, `patch_stride=64`.
- Loss/schedule/optimizer as paper; subject split fallback.

Acceptance gates (unchanged):
- By epoch 2: BAC > 0.20; by epoch 5: BAC > 0.40; final: 0.62 ± 0.02 on eval.
- Enforce cache invariants and shape checks at extraction/dataset layers.
- Gradient flow verified through mapper + head.

## Tests To Add

- Unit: event extraction shape (23,1000), sfreq==200, window −2..+3 s; filter + notch applied.
- Unit: dataset META/index validation; labels in 0..5; channels in reference order.
- Integration: parity pipeline (23,1000) → mapper (20,1000) → head; loss/scheduler configs present; gradient flow.

## Guardrails

- SSOT: all logic in src/; experiments thin wrappers only; no Lightning.
- Safe load: prefer torch.load(..., weights_only=True) on PyTorch ≥2.4; otherwise our safe_load helper; no arbitrary unpickling.

## Bottom Line

We’ve implemented the event-only 5 s @ 200 Hz pipeline, mapper, and a parity‑ready EEGPT path (native 1000 via stride). The remaining step is to build the cache on real data and run training to confirm ≈62% BAC. See TUEV_IMPLEMENTATION_PLAN.md for acceptance criteria and run commands.
