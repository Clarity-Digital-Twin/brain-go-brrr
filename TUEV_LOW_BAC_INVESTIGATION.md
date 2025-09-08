# TUEV Low BAC Investigation — Root Cause and Fix Plan

Created: 2025-09-08
Owner: Core ML
Status: Root cause evidence assembled; hypotheses validated against EEGPT reference repo

## Summary

- Symptom: Validation Balanced Accuracy (BAC) ~0.22 at epoch ~40+ during TUEV (6‑class) training, target 0.6232 (EEGPT paper Table 3).
- Impact: Model is effectively predicting “background” for most windows; minority classes not learned.
- Root cause: Training batches are not class‑balanced under extreme dataset imbalance (~99.5% background). Weighted loss alone is insufficient; sampler is unbalanced. Evidence shows collapse to background.

## 🔥 CRITICAL DISCOVERY FROM EEGPT REFERENCE REPO

**The EEGPT authors DID NOT use class weights or balanced sampling for TUEV!**

Evidence from `reference_repos/EEGPT/downstream_tueg/`:
- **run_class_finetuning_EEGPT_change_tuev.py:480**: `criterion = torch.nn.CrossEntropyLoss()` — NO weight parameter!
- **run_class_finetuning_EEGPT_change_tuev.py:292-294**: Uses standard `DistributedSampler` with shuffle=True — NO balanced sampling!
- **engine_for_finetuning_EEGPT.py:162**: Plain CrossEntropyLoss, no weights
- **utils.py**: No WeightedRandomSampler imports or usage anywhere

**This means (facts from code):**
1. EEGPT achieves ~62.32% BAC using plain CrossEntropy (no class weights) and standard DistributedSampler (no class balancing).
2. Their TUEV fine-tuning uses lr=5e-4, weight_decay=0.05, layer_decay (default 0.9), and label smoothing (default 0.1).
3. Our current config uses lr=5e-4 and batch_size=64 (matches), but weight_decay=0.01, no layer decay, and no label smoothing; we also add class weights without a balanced sampler.

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

## 🎯 Working Hypotheses (to be resolved by A/B)

### H1: Hyperparameter mismatch vs EEGPT
- EEGPT uses CE (no weights), lr=5e-4, wd=0.05, label_smoothing=0.1, layer_decay=0.9, bs=64. Our current run uses CE with class weights, lr=5e-4, wd=0.01, no smoothing, no layer decay, bs=64.
- Claim: Matching their simpler recipe (no weights; smoothing; higher wd; layer decay) recovers BAC without sampler.

### H2: Batch imbalance dominates under our recipe
- With ~99.5% background and no class‑balanced sampler, batches lack minority samples; weights alone (with large multipliers) are insufficient and can destabilize training.
- Claim: Adding a balanced sampler (or balanced batches) is required for our weighted‑loss recipe.

Evidence common to both: Background F1 ~0.996–0.998 while minority F1 ~0.0–0.2; eval BAC ~0.22 across epochs.

## 🚨 URGENT: Two Divergent Fix Strategies

### Strategy A: Match EEGPT Paper EXACTLY (Recommended)
**Rationale**: They got 62% BAC, we got 22%. Do EXACTLY what they did.

1. **REMOVE class weights** - Use plain `nn.CrossEntropyLoss()`
2. **Match their hyperparameters** (checked in reference repo):
   - lr=5e-4
   - weight_decay=0.05
   - warmup_epochs=5
   - batch_size=64
   - label_smoothing=0.1 (LabelSmoothingCrossEntropy path)
   - layer_decay enabled (e.g., 0.9)
3. **Keep simple sampling** - Just torch.randperm like they do
4. **Add layer decay** - Different LR for each layer (optional but they use it)

### Strategy B: Fix The Imbalance Problem (Our Original Plan)

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

## 🤔 CRITICAL QUESTION: Should We Rebuild Cache?

### Arguments FOR Rebuilding:
1. **Fpz Concern** - If Fpz is missing in raw, we zero‑fill the Fpz slot (not Fz). This matches our CHANNELS_TUEV_20 SSOT. Only rebuild if the slot/order is wrong.
2. **Window/Sample Rate** - Our pipeline uses exactly 4.0s @ 256 Hz (1024 samples) consistently (confirmed). Verify consistency Train/Eval; rebuild only if a mismatch is found.
3. **Normalization** - Double-check our normalization matches theirs
4. **Fresh Start** - Eliminate any cached preprocessing bugs

### Arguments AGAINST Rebuilding:
1. **Cache validated** - META.json confirms correct properties
2. **Channel order matches** - CHANNELS_TUEV_20 aligns with EEGPT
3. **Time cost** - Cache rebuild takes hours
4. **Current run** - Let it finish for baseline comparison

### RECOMMENDATION:
**Don't rebuild YET.** First try Strategy A (match EEGPT hyperparams) with current cache. If that fails, THEN rebuild cache.

## A/B Plan (order, stop rules)

- A1 (EEGPT‑match): CE (no weights), lr=5e-4, wd=0.05, bs=64, label_smoothing=0.1, no sampler, layer_decay on.
- A2 (ablation): A1 but without label_smoothing.
- B1 (balanced): CE with class weights + WeightedRandomSampler, lr=5e-4, wd=0.05, bs=64, no smoothing.
- B2 (hybrid): CE (no weights) + WeightedRandomSampler, same hparams.

Selection: best by eval BAC.
Stop rules:
- If BAC < 0.30 after 10 epochs → stop that arm.
- If no improvement ≥ 0.05 BAC across 5 epochs → stop that arm.
Seeds: {42, 123, 456}; report mean±std for the best arm.

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

## 🎯 BULLETPROOF STATUS (2025-09-08)

### ✅ Code Deltas Applied for Strategy A:
1. **Config Fixed** (`experiments/eegpt_linear_probe/configs/tuev.yaml`):
   - `weighted_loss: false` ✅ (was defaulting to True)
   - `weight_decay: 0.05` ✅ (already correct)
   - `label_smoothing: 0.1` ✅ (in config)

2. **Training Script Fixed** (`train_tuev_mne.py`):
   - Line 487: Reads label_smoothing from config ✅
   - Line 489: Default weighted_loss now False ✅
   - Line 518: `nn.CrossEntropyLoss(label_smoothing=0.1)` ✅
   - NO class weights for Strategy A ✅

3. **Launch Script Fixed** (`scripts/launch_tuev_mne.sh`):
   - Line 51: Uses `tuev_mne_fixed` cache dir ✅
   - Consistent with cache builder script ✅

4. **Preprocessor Previously Fixed** (`tuev_preprocessor.py`):
   - Fpz interpolation: (Fp1+Fp2)/2 when available ✅
   - Falls back to zeros if Fp1/Fp2 missing ✅

### 🚀 Ready to Execute:
```bash
cd experiments/eegpt_linear_probe/scripts
./launch_tuev_cache.sh   # Build cache with Fpz fix
./launch_tuev_mne.sh     # Train with EEGPT hyperparams
```

### Monitoring Criteria:
- **Success**: BAC > 0.30 within 10 epochs
- **Stall**: No +0.05 BAC gain over 5 epochs → switch to Strategy B

### If That Fails:
1. **THEN rebuild cache** with potential Fpz fix
2. **Consider channel mapper** as last resort
3. **Investigate data loading** for systematic issues

### Key Insight:
**EEGPT authors got 62% BAC with NO class balancing techniques.** Either:
- Their simple approach works better than complex weighting
- There's a bug in our implementation
- The channel/normalization is subtly different

---

**PRIORITY: Try Strategy A (match EEGPT) before anything else.**
