# TUEV Low BAC Investigation — Root Cause and Fix Plan

Created: 2025-09-08
Owner: Core ML
Status: Root cause identified; fix plan proposed (non‑disruptive to current run)

## Summary

- Symptom: Validation Balanced Accuracy (BAC) ~0.22 at epoch ~40+ during TUEV (6‑class) training, target 0.6232 (EEGPT paper Table 3).
- Impact: Model is effectively predicting “background” for most windows; minority classes not learned.
- Root cause: Training batches are not class‑balanced under extreme dataset imbalance (~99.5% background). Weighted loss alone is insufficient; sampler is unbalanced. Evidence shows collapse to background.

## Evidence (from code and logs)

1) Class imbalance and weights (correct but insufficient)

- Log (tuev_mne_20250908_113817.log):
  - Class counts: [55, 208, 199, 94, 81, 157624]
  - Class weights (inverse frequency): [479.58, 126.81, 132.55, 280.60, 325.64, 0.167]
- Source: experiments/eegpt_linear_probe/train_tuev_mne.py
  - Computes inverse‑frequency weights; background has smallest weight — this is correct.

2) Per‑class performance collapse

- Log (multiple epochs): Per‑class F1 very low for SPSW/GPED/PLED/EYEM/ARTF; BCKG ~0.997.
- Example (13:22–16:06):
  - {'SPSW': ~0.12–0.43, 'GPED': ~0.0–0.21, 'PLED': ~0.0–0.10, 'EYEM': ~0.12–0.28, 'ARTF': ~0.0–0.21, 'BCKG': ~0.996–0.998}
- Interpretation: Head is largely predicting background; minority recalls near zero → BAC ~0.17–0.25.

3) No class‑balanced sampler in TRAIN

- Code: experiments/eegpt_linear_probe/train_tuev_mne.py
  - Uses a deterministic epoch‑wise permutation (torch.randperm) and DataLoader with shuffle=False; no WeightedRandomSampler or balanced batching.
  - Batches reflect natural distribution → minority classes rarely occur → gradients dominated by background.

4) Labels, mapping, metrics, and channels look consistent

- Labels and indices: src/brain_go_brrr/infra/data/tuev_dataset.py
  - CLASS_MAPPING: {'spsw':0,'gped':1,'pled':2,'eyem':3,'artf':4,'bckg':5}
  - __getitem__ returns y as torch.long with correct indices.
  - Class counts persisted in cache index; eval split includes all classes (sparse minorities).
- Metric: balanced_accuracy_score used on full eval set per epoch (evaluate()), consistent with SSOT.
- Channels: src/brain_go_brrr/infra/preprocessing/tuev_preprocessor.py
  - Enforces 23→20 mapping; drops A1/A2; synthesizes missing canonical channels (e.g., Fpz) as zeros; final order from CHANNELS_TUEV_20.
  - Cache META validated against CHANNELS_TUEV_20 (Fp1, Fpz, Fp2, …, O2). No mismatch detected.

Conclusion: The primary defect is lack of class‑balanced sampling under extreme imbalance; secondary items are acceptable and not the cause of 0.22 BAC.

## Root Cause

- Extreme imbalance (≈99.5% background) + unbalanced batches ⇒ minority classes seldom appear within a batch.
- Even with inverse‑frequency weights, gradient signal for minorities is too sparse/noisy; the head collapses to background.
- Evidence: Background F1 ≈ 0.997 while minority F1 ≈ 0–0.2; BAC hovers ≈ 0.2.

## Fix Plan (minimal, high‑ROI)

1) Balanced sampling for TRAIN

- Option A: WeightedRandomSampler with per‑sample weights proportional to inverse class frequency (or “effective number of samples” per Cui et al.). Ensures minority samples appear regularly in batches.
- Option B: Class‑balanced batcher (e.g., fixed number per class per batch) to guarantee diverse batches.

2) Keep weighted loss (but verify numerics)

- Continue using inverse‑frequency class weights (normalized). Confirm dtype float32 and device match.
- Alternatively, try “effective number” weights: w_c ∝ (1−β)/(1−β^{n_c}), β≈0.9999, then normalize.

3) Head/optimization hygiene

- Freeze EEGPT backbone (already true); train linear/small MLP probe.
- AdamW lr≈1e−3, wd≈1e−2; disable label smoothing initially; cosine/step schedule okay.

4) Monitoring/selection

- Log per‑epoch confusion matrix and per‑class recall (already logging per‑class F1; keep it).
- Select best checkpoint strictly by eval BAC.

Optional later: Channel mapper (23→20, 1×1 conv) as a +~1% improvement only after BAC is near target; it won’t fix collapse.

## Acceptance Criteria (next run)

- Eval BAC > 0.30 within a few epochs; > 0.50 mid‑run; trending toward ~0.62 by end.
- Confusion shows non‑zero recall across all minority classes; predictions not dominated by background.
- Train batches confirmed to include minority classes consistently.

## Why We Do NOT Rebuild Cache Now

- Preprocessor validates channel order against CHANNELS_TUEV_20; missing canonical channels (e.g., Fpz) are synthesized as zeros in correct slots; META.json confirms expected properties (sr, unit, window, norm, channels).
- No evidence of channel order or normalization mismatch in logs; cache is consistent.
- Rebuild only if diagnostics show channel order mismatch or inconsistent normalization.

## Pointers (for implementers)

- Training script: experiments/eegpt_linear_probe/train_tuev_mne.py
  - Add a class‑balanced sampler for the train DataLoader.
  - Keep weighted CrossEntropyLoss; confirm printed class weights show background as smallest.
  - Ensure best model selection by eval BAC (already present).
- Dataset & cache: src/brain_go_brrr/infra/data/tuev_dataset.py
  - CLASS_MAPPING and saved labels are aligned with indices 0..5.
  - Cache index includes class_counts; eval split includes all classes (sparse minorities).
- Preprocessing: src/brain_go_brrr/infra/preprocessing/tuev_preprocessor.py
  - 23→20 mapping; A1/A2 dropped; Fpz synthesized if missing; final list CHANNELS_TUEV_20.

## Appendix — Key Log Excerpts

```
2025-09-08 11:53:10,922 - Class counts: [55, 208, 199, 94, 81, 157624]
2025-09-08 11:53:10,922 - Class weights: [479.58, 126.81, 132.55, 280.60, 325.64, 0.16734]
...
Per-class F1 (examples over many epochs):
  {'SPSW': ~0.12–0.43, 'GPED': ~0.0–0.21, 'PLED': ~0.0–0.10, 'EYEM': ~0.12–0.28, 'ARTF': ~0.0–0.21, 'BCKG': ~0.996–0.998}
```

---

If you need a concrete sampler patch and config snippet, ping Core ML — we’ll provide a minimal change set that keeps the current cache and reruns cleanly.

