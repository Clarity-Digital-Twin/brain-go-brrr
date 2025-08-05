# Next Steps to Match Literature AUROC Performance

## Current Status
- Best checkpoint: epoch 5 with 67.9% accuracy (~0.68 AUROC)
- Literature target: 0.87 AUROC (EEGPT paper on TUAB)
- Gap: ~0.19 AUROC

## Key Issues Identified

### 1. **Window Duration Mismatch** ⚠️
- **Paper uses: 4-second windows**
- **We're using: 8-second windows**
- This is the PRIMARY issue - EEGPT was pretrained on 4s windows!

### 2. **Training Configuration**
- Paper: OneCycle LR (2.5e-4 → 5e-4 → 3.13e-5) for 200 epochs
- We used: StepLR with aggressive jumps causing instability

### 3. **Preprocessing Differences**
- Paper mentions 0-38Hz bandpass for MI tasks
- We're using 0.5-50Hz (might be fine for TUAB)

## Action Plan

### Step 1: Rebuild Dataset with 4-second Windows
```bash
# Rebuild TUAB cache with correct window size
python src/brain_go_brrr/data/build_tuab_cache.py \
    --window_duration 4.0 \
    --window_stride 2.0 \
    --sampling_rate 256 \
    --cache_dir data/cache/tuab_4s_windows
```

### Step 2: Create New Training Config
```yaml
# configs/tuab_4s_paper_aligned.yaml
data:
  window_duration: 4.0    # MATCH PAPER!
  window_stride: 2.0      # 50% overlap
  sampling_rate: 256
  bandpass_low: 0.5
  bandpass_high: 38.0     # Match paper for consistency
  
training:
  optimizer:
    name: "AdamW"
    lr: 2.5e-4
    weight_decay: 0.01
  
  scheduler:
    name: "OneCycleLR"
    max_lr: 5e-4
    epochs: 200
    pct_start: 0.3
    anneal_strategy: "cos"
    
  max_epochs: 200
  gradient_clip_val: 1.0
```

### Step 3: Resume Training with Best Practices
```bash
python experiments/eegpt_linear_probe/train_aligned.py \
    --config configs/tuab_4s_paper_aligned.yaml \
    --checkpoint output/nan_safe_run_20250804_081330/checkpoint_epoch_5.pt \
    --resume_weights_only  # Only load probe weights, not optimizer state
```

### Step 4: Additional Improvements
1. **Data Augmentation**: Add temporal shifts, amplitude scaling
2. **Regularization**: Add L2 penalty, increase dropout
3. **Ensemble**: Train multiple seeds, average predictions
4. **Class Balancing**: Ensure balanced sampling

## Expected Timeline
- Dataset rebuild: ~2 hours
- Training (200 epochs): ~6-8 hours
- Should achieve >0.85 AUROC with proper setup

## Quick Win Alternative
If you need results NOW:
1. Use the epoch 5 checkpoint (0.68 AUROC)
2. Apply test-time augmentation
3. Calibrate predictions with temperature scaling
4. This might get you to ~0.72-0.75 AUROC

## References
- EEGPT paper: 4s windows, 256Hz, 64-sample patches
- TUAB results: Table 9, AUROC 0.8718±0.0050
- Training: Section 3.2, OneCycle LR, 200 epochs