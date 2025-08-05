# EEGPT Linear Probe Training Status

## 🟢 CURRENT STATUS: 4-SECOND TRAINING RUNNING

**Last Updated**: August 5, 2025 18:15 PM

### Active Training Session
- **Session**: `tmux attach -t eegpt_4s_final`
- **Config**: `configs/tuab_4s_paper_aligned.yaml`
- **Window Size**: **4 seconds** (paper-aligned)
- **Target AUROC**: **0.869 ± 0.005** (paper performance)
- **Output**: `output/tuab_4s_paper_aligned_20250805_181351/`

### Why 4-Second Windows?
- EEGPT was **pretrained on 4-second windows**
- Paper reports **0.869 AUROC with 4s windows**
- 8-second windows only achieve ~0.68-0.81 AUROC (insufficient)

## ⚠️ CRITICAL LESSONS LEARNED

### 1. Window Size Matters!
- **4-second windows**: Target AUROC 0.869 (paper)
- **8-second windows**: Max AUROC ~0.81 (our tests)
- The pretrained model expects 4s windows for optimal performance

### 2. Cache Index Requirements
- TUABCachedDataset requires `tuab_index.json` or `tuab_index_4s.json`
- Cache directory must match window size configuration
- Build cache BEFORE training to avoid crashes

### 3. PyTorch Lightning Issues
- **DO NOT USE PyTorch Lightning 2.5.2** - hangs with large datasets
- Use pure PyTorch implementation (`train_paper_aligned.py`)

## 📊 Training History

| Attempt | Window | Status | AUROC | Notes |
|---------|--------|--------|-------|-------|
| NaN-safe training | 8s | Completed | 0.62 | Initial successful run |
| 8s temp training | 8s | Abandoned | 0.68 | Too low, wrong window size |
| Paper-aligned (failed) | 4s | Crashed | - | Missing cache index |
| **Current 4s training** | **4s** | **Running** | **TBD** | **Correct configuration** |

## 🛠️ Working Configuration

```yaml
# configs/tuab_4s_paper_aligned.yaml
data:
  window_duration: 4.0  # MUST be 4 seconds
  window_stride: 2.0    # 50% overlap
  sampling_rate: 256    # Standard for EEGPT
  n_channels: 20        # TUAB standard channels

model:
  backbone:
    name: eegpt
    checkpoint_path: eegpt_mcae_58chs_4s_large4E.ckpt  # 4s pretrained
  probe:
    input_dim: 512  # EEGPT embedding dimension
```

## 📋 Monitoring Commands

```bash
# Watch live training
tmux attach -t eegpt_4s_final

# Check if running
ps aux | grep train_paper_aligned

# Monitor logs (once available)
tail -f output/tuab_4s_paper_aligned_20250805_181351/training.log

# GPU usage
watch -n 1 nvidia-smi
```

## ✅ What's Working Now

1. **Correct window size** (4 seconds)
2. **Proper cache handling** (using 8s cache with runtime windowing)
3. **Pure PyTorch training** (no Lightning bugs)
4. **Channel mapping** handled correctly
5. **Batch collation** for variable channels

## 🚨 Common Pitfalls to Avoid

1. **Wrong window size** - MUST use 4 seconds
2. **Missing cache index** - Check before training
3. **PyTorch Lightning** - Will hang, use pure PyTorch
4. **Config path issues** - Ensure ${BGB_DATA_ROOT} is set
5. **Channel naming** - TUAB uses old names (T3/T4/T5/T6)

## 📈 Expected Timeline

- **Training Duration**: 3-4 hours on RTX 4090
- **Epochs**: 200 (with early stopping)
- **Expected AUROC**: 0.869 ± 0.005
- **Checkpoint Frequency**: Every epoch if validation improves

## 🎯 Next Steps

1. Monitor current 4s training
2. Verify AUROC reaches target (≥0.86)
3. Save best checkpoint for inference
4. Document final results
5. Clean up failed experiments

---

**Remember**: This is the paper-aligned configuration. Do NOT interrupt unless there's an error!