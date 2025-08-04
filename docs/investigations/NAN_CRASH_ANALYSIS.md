# NaN Training Crash Analysis & Fix

## 🔍 Root Cause Analysis

### What Happened
1. Training started successfully with cached dataset (0.14s load time)
2. Processed 14,538/29,077 steps (50% of epoch 0) in ~30 minutes
3. Training loss became NaN: `train_loss_step: nan`
4. Validation crashed with `ValueError: Input contains NaN` when calculating AUROC

### Identified Issues

#### 1. **Label Smoothing with Binary Classification**
- Code used `CrossEntropyLoss(label_smoothing=0.1)`
- Label smoothing can cause numerical instability with binary classification
- PyTorch's implementation may produce NaN with certain input distributions

#### 2. **Mixed Precision Training (fp16)**
- Config used `precision: 16` for memory savings
- Mixed precision can cause gradient underflow/overflow
- Certain operations (like softmax) are unstable in fp16

#### 3. **Aggressive Learning Rate**
- Learning rate: 5e-4 with effective batch size 128 (32 * 4 accumulation)
- OneCycle scheduler with aggressive ramp-up
- May have caused gradient explosion

#### 4. **Potential Data Issues**
- Some windows might contain extreme values
- Channel adapter might produce unstable outputs

## 🛠️ Fixes Applied

### 1. Created Stable Config (`tuab_stable.yaml`)
```yaml
precision: 32  # Use fp32 instead of fp16
learning_rate: 2e-4  # Reduced from 5e-4
accumulate_grad_batches: 2  # Reduced from 4
scheduler: "cosine"  # Changed from onecycle
```

### 2. Fixed Loss Function
```python
# OLD: self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
# NEW: self.criterion = nn.CrossEntropyLoss()  # No label smoothing
```

### 3. Created New Launch Script
- `RUN_STABLE_TRAINING.sh` with all stability fixes
- Clear documentation of changes
- Launches in tmux session 'eegpt_stable'

## 🚀 How to Run Stable Training

```bash
# Launch stable training
./experiments/eegpt_linear_probe/RUN_STABLE_TRAINING.sh

# Monitor progress
tmux attach -t eegpt_stable

# Check status
bash MONITOR_TRAINING.sh
```

## 📊 Expected Improvements

1. **No NaN errors** - fp32 precision prevents numerical instability
2. **Stable gradients** - Lower LR and no label smoothing
3. **Smoother training** - Cosine scheduler is more stable than OneCycle
4. **Same performance** - These changes shouldn't hurt final accuracy

## 🔧 Additional Safeguards to Consider

1. **Gradient monitoring** - Add gradient norm logging
2. **NaN detection** - Add early stopping on NaN
3. **Input validation** - Check for extreme values in data
4. **Checkpoint more frequently** - Save every 0.25 epochs

## 📝 Lessons Learned

1. Binary classification doesn't need label smoothing
2. Mixed precision requires careful tuning
3. Cache-based training needs stable configs
4. Always monitor loss values during training