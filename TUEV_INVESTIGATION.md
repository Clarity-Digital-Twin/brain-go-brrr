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

## What Our Repo Currently Does (Measured)

- Sliding 4 s @ 256 Hz windows (not event-only):
  - src/brain_go_brrr/infra/data/tuev_dataset.py + src/brain_go_brrr/infra/preprocessing/tuev_preprocessor.py build fixed-grid windows.
- Extreme imbalance in train cache:
  - data/cache/tuev_23ch_paper_parity/train/index_train_mne-ar-v4.json → class_counts: background = 179,444 of 180,205 (99.58%).
- Experiments use OneCycle warmup and sliding windows (fallback path):
  - experiments/eegpt_linear_probe/train_tuev_mne.py (OneCycleLR pct_start=0.3, unweighted CE with smoothing; optional 23→20 mapper).

## Why We’re At 0.1667 BAC

- Training on 99.58% background windows encourages a trivial solution (predict background).
- Balanced accuracy on a 6-class task = 1/6 when predicting a single class.
- Loss converges without improving BAC (observed), consistent with collapsed predictions.

## Key Divergences (Actionable)

- Task mismatch: sliding-window detection vs event-segment classification (root cause).
- Data distribution: 99.58% background vs event-only segments.
- Input dimensions: our path 4 s @ 256 Hz vs reference 5 s @ 200 Hz (20×1000).
- Hyperparameters: our fallback trainer lacks epoch-based warmup + layer decay; reference uses warmup=5 + layer_decay=0.65.

## Fix Direction (Strict Parity)

Implement the parity path in src/ (SSOT), keep experiments/ thin:
- Event extractor: (23,1000) @ 200 Hz from EDF + .rec/.lab with 0.1–75 Hz + 50 Hz notch, referential order; subject-level splits.
- Event dataset: cached .pt segments + META/index (sr=200, unit=V, samples=1000, channels=23, segment_type=event).
- Channel mapper: reuse TUEVChannelMapper (23→20), then classifier head on 20×1000.
- Trainer: unweighted CE with smoothing=0.1; lr=5e-4, weight_decay=0.05, warmup_epochs=5, layer_decay=0.65, epochs≈30, effective batch≈400 (distributed).

Acceptance gates:
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

We’re solving the wrong task. Switching to event-only 5 s @ 200 Hz segments with the 23→20 mapper and the verified reference hyperparameters is necessary and sufficient to reach paper parity. See TUEV_IMPLEMENTATION_PLAN.md for the ironclad plan and acceptance criteria.

