# EEGPT Linear Probe Training Analysis

## Current Status (2025-08-01)

### Training Progress
- **Current Epoch**: 6/20
- **Best AUROC**: 0.7936 (Epoch 1)
- **Current AUROC**: ~0.76-0.78 (plateaued)
- **Target AUROC**: ≥0.87 (paper baseline), ideally ≥0.93

### Key Achievements
1. ✅ Fixed NaN issues with robust normalization
2. ✅ Stable training with no gradient explosions
3. ✅ Model learning (improved from random 0.5 to 0.79)
4. ✅ Checkpoints saved at each validation

### Validation History
- Epoch 0: AUROC = 0.7881
- Epoch 1: AUROC = 0.7936 ⭐ (best)
- Epoch 2: AUROC = 0.7797
- Epoch 3: AUROC = 0.7806
- Epoch 4: AUROC = 0.7802
- Epoch 5: AUROC = 0.7607 (degradation)

### Observations
1. Model peaked early (Epoch 1) and plateaued
2. Slight overfitting signs (validation metrics degrading)
3. Not reaching paper baseline performance (0.87)

## Potential Issues & Solutions

### 1. Learning Rate Schedule
**Issue**: Fixed learning rate (5e-4) might be too high after initial epochs
**Solution**: Implement learning rate scheduling
```python
# Add to training script
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3, verbose=True
)
```

### 2. Model Architecture
**Issue**: Simple linear probe might be insufficient
**Solutions**:
- Add dropout layer (p=0.3-0.5)
- Try 2-layer MLP probe instead of single linear layer
- Add batch normalization

### 3. Data Augmentation
**Issue**: Limited augmentation might cause overfitting
**Solutions**:
- Add temporal augmentation (time shifting)
- Add noise injection
- Try mixup or cutmix strategies

### 4. Hyperparameter Tuning
**Current Settings**:
- Batch size: 64
- Learning rate: 5e-4
- Weight decay: 0.01
- Window size: 8 seconds
- Stride: 8 seconds (no overlap)

**Potential Adjustments**:
- Increase batch size to 128 or 256
- Reduce learning rate to 1e-4 or 5e-5
- Try different window stride (4s for 50% overlap)
- Adjust weight decay

### 5. EEGPT Feature Quality
**Issue**: Frozen EEGPT features might not be optimal for TUAB
**Solutions**:
- Fine-tune last few layers of EEGPT
- Use features from multiple EEGPT layers
- Apply feature normalization/standardization

## Recommended Next Steps

### Immediate Actions
1. Let current training complete or early stop
2. Analyze best checkpoint (epoch 1) performance
3. Implement learning rate scheduling

### Short-term Improvements
1. Create enhanced training script with:
   - Learning rate scheduling
   - Better model architecture (2-layer MLP)
   - Dropout regularization
   - Validation patience adjustment

2. Experiment with hyperparameters:
   - Batch size sweep: [64, 128, 256]
   - Learning rate sweep: [1e-4, 5e-5, 1e-5]
   - Window overlap: [0%, 50%, 75%]

### Long-term Considerations
1. Fine-tune EEGPT backbone (carefully, to avoid catastrophic forgetting)
2. Ensemble multiple models
3. Try different foundation models or architectures
4. Investigate data quality and label distribution

## Training Commands

### Current Training
```bash
tmux attach -t eegpt_fast  # Monitor current training
```

### Best Checkpoint Location
```
logs/fast_robust_20250801_103902/checkpoints/tuab-epoch=01-val_auroc=0.7936.ckpt
```

### Next Training Run (with improvements)
```bash
# Create enhanced training script first
python experiments/eegpt_linear_probe/train_enhanced.py
```

## Conclusion

While we haven't reached the target AUROC of 0.87, we've successfully:
1. Fixed the critical NaN issue
2. Achieved stable training
3. Improved from random performance to ~0.79 AUROC

The model has plateaued, suggesting we need architectural or hyperparameter changes rather than just more epochs. The next step should focus on implementing the recommended improvements, particularly learning rate scheduling and model architecture enhancements.