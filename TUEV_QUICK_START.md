# 🚀 TUEV Quick Start Guide

## Current Status
- ✅ **TUAB trained**: 0.79 AUROC (working)
- ✅ **TUEV dataset**: FULLY DOWNLOADED (518 EDF files, 11,396 labels)
- ✅ **Infrastructure**: Ready (reuse TUAB pipeline)
- ✅ **Verified**: All 6 classes present, 370 subjects (paper had 288)
- ⚠️ **Split Strategy**: MUST use BIOT split (NOT LOSO, NOT random!)
- 🔴 **CRITICAL**: Deep audit found MAJOR architecture differences! See `TUEV_CRITICAL_ARCHITECTURE.md`

## Immediate Actions Required

### 1️⃣ ~~Download TUEV Dataset~~ ✅ DONE!
Dataset already downloaded at: `data/datasets/external/tuh_eeg/TUEV/v2.0.1/`

### 2️⃣ Verify Dataset ✅ DONE!
```bash
uv run python scripts/verify_tuev_dataset.py

# Results:
# ✅ 370 subjects (290 train, 80 eval)
# ✅ 518 EDF files, 11,396 label files
# ✅ 6 classes confirmed: SPSW, GPED, PLED, EYEM, ARTF, BCKG
# ⚠️ Note: 250 Hz sampling (needs resampling to 256 Hz)
# ⚠️ Note: 26-27 channels (need to select 23)
```

### 3️⃣ Key Differences from TUAB (FINAL - Use Table 13!)

| Parameter | TUAB (Table 12) | TUEV (Table 13) | 
|-----------|-----------------|-----------------|
| **Input Size** | 23 × 2000 | **23 × 1000** |
| **Actual Window** | 7.8 seconds | **3.9 seconds** |
| **Channel Reduction** | 23 → 20 | **23 → 20** |
| **Classes** | 2 | **6** |
| **Dropout** | 0.25 | **0.5** |
| **Batch Size** | 100 | **500** |
| **Temporal Kernel** | 15 | **55** |
| **Temporal Padding** | 7 | **27** |
| **Optimizer** | AdamW @ 5e-4 | **AdamW @ 5e-4** |
| **Output Shape** | 31 × 4 × 512 | **15 × 4 × 512** |

### 4️⃣ What TUEV Classes Mean

1. **SPSW** - Spike & Sharp Wave → **IED (epileptiform)**
2. **GPED** - Generalized Periodic Epileptiform → **IED (severe)**
3. **PLED** - Periodic Lateralized Epileptiform → **IED (focal)**
4. **EYEM** - Eye Movement → Artifact
5. **ARTF** - Other Artifacts → Technical issue
6. **BCKG** - Background → Normal

**The first 3 (SPSW, GPED, PLED) ARE the IEDs!**

## Files to Create

### Must Implement
1. `experiments/eegpt_linear_probe/tuev_dataset.py` - Dataset loader
2. `experiments/eegpt_linear_probe/train_tuev_aligned.py` - Training script
3. `experiments/eegpt_linear_probe/configs/tuev_5s_paper_aligned.yaml` - Config

### Can Reuse from TUAB
- ✅ EEGPT wrapper
- ✅ Memory-mapped dataset pattern
- ✅ Scheduler fixes
- ✅ Checkpoint saving

## Expected Performance (Paper)

```
Balanced Accuracy: 0.6232 ± 0.0114
Weighted F1:       0.8187 ± 0.0063
Cohen's Kappa:     0.6351 ± 0.0134
```

## Why TUEV After TUAB?

```
Patient EEG → TUAB (abnormal?) 
              ↓ if yes
              TUEV (what type?)
              ↓
              Clinical Report:
              "Abnormal EEG with periodic 
               lateralized epileptiform 
               discharges (PLED)"
```

## ⚠️ Common Pitfalls

1. **Wrong window size** - TUEV is 5s, not 4s!
2. **Wrong channels** - TUEV has 23, not 20!
3. **Wrong split** - Use BIOT strategy (existing train/eval), NOT LOSO!
4. **Class imbalance** - Use weighted loss
5. **Memory issues** - 112k × 5s × 23ch = big! Use mmap
6. **Kernel size** - Must be (1, 55) not (1, 15)
7. **Sampling rate** - Resample 250 Hz → 256 Hz!

## Command Summary

```bash
# Step 1: Get dataset
./scripts/download_tuev.sh

# Step 2: Verify
python scripts/verify_tuev_dataset.py

# Step 3: Build cache (after implementing dataset.py)
python experiments/eegpt_linear_probe/build_tuev_cache.py

# Step 4: Train (after implementing train script)
bash experiments/eegpt_linear_probe/LAUNCH_TUEV.sh

# Step 5: Evaluate
python scripts/evaluate_tuev.py
```

---

**READ FIRST**: `TUEV_IMPLEMENTATION_PLAN.md` for complete details!