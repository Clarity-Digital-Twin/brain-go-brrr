# MNE + Autoreject Integration for EEGPT Training

## ⚠️ CRITICAL TECHNICAL DEBT - 20-CHANNEL ISSUE

**Current Cache Issue (Aug 27, 2025)**: 
- The existing cache (`mne-ar-v2`) contains 304 windows (0.081%) with 20 channels instead of 19
- Affected files: `aaaaakfo_s004_t000.edf` and `aaaaakfo_s005_t000.edf`
- **Active Workaround**: Collate function drops channel 4 (Fz) when encountering 20-channel data
- **Fixed for Future**: Preprocessor now enforces exactly 19 channels for all new cache builds

See `TECH_DEBT_CRITICAL.md` for full details and remediation plan.

## 🎯 Goal: Improve TUAB accuracy from 56% to 87% AUROC

This implementation adds MNE-Python and Autoreject preprocessing to the EEGPT training pipeline, addressing the critical gap where we were training on noisy data but inferring on clean data.

## 📁 New Files Created

```
experiments/eegpt_linear_probe/
├── mne_integration/              # NEW: MNE preprocessing module
│   ├── __init__.py
│   ├── preprocessor.py          # TUABPreprocessor with verified pipeline
│   └── cache_builder.py         # Script to build preprocessed cache
├── datasets/
│   └── tuab_mne_dataset.py      # NEW: Dataset with MNE preprocessing
├── train_tuab_mne.py            # NEW: Parallel training script
└── scripts/
    ├── build_mne_cache.sh       # Build preprocessed cache
    ├── launch_tuab_mne.sh       # Launch MNE training
    └── monitor_mne_training.sh  # Monitor progress
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# If not already installed
pip install mne autoreject
```

### 2. Build MNE-Preprocessed Cache

```bash
cd experiments/eegpt_linear_probe
./scripts/build_mne_cache.sh

# This will:
# - Load TUAB EDF files
# - Apply MNE preprocessing (filters, artifact detection)
# - Apply Autoreject (adaptive thresholds)
# - Save clean epochs to cache
# - Expected: 20-40% rejection rate (normal for clinical data)
```

### 3. Launch Training with Clean Data

```bash
./scripts/launch_tuab_mne.sh

# Monitor progress:
./scripts/monitor_mne_training.sh
```

## 🔬 Preprocessing Pipeline (Verified)

1. **Load EDF** with MNE
2. **Channel Mapping**: T3→T7, T4→T8, T5→P7, T6→P8
3. **Resample** to 256 Hz (EEGPT requirement)
4. **Bandpass Filter**: 0.5-45 Hz
5. **Notch Filter**: 60 Hz (line noise)
6. **RANSAC**: Detect bad channels
7. **Interpolation**: Fix bad channels
8. **Re-reference**: Common average
9. **Epoching**: 4-second windows
10. **Autoreject**: Adaptive artifact rejection

### TUAB-Specific Parameters

```python
# Autoreject parameters (optimized for 20-channel clinical data)
n_interpolate = [1, 2, 3, 4]  # Can interpolate up to 4 channels
consensus = [0.3, 0.5, 0.7]    # More aggressive for clinical data
cv = 5                         # Reduced from default=10 for speed
```

## 📊 Expected Improvements

| Metric | Without MNE | With MNE | Target |
|--------|------------|----------|--------|
| AUROC | 56% | 75-85% | 87% |
| Training Stability | Poor | Good | Excellent |
| Convergence Speed | Slow | Fast | Fast |

## ⚙️ Configuration

The preprocessing parameters are defined in:
- `mne_integration/preprocessor.py`: Default parameters
- Can be overridden via config dict when initializing `TUABPreprocessor`

## 🔍 Monitoring Training

Watch for these indicators of success:

1. **Rejection Rate**: Should see 20-40% epochs rejected (healthy cleaning)
2. **AUROC Improvement**: Should jump from ~56% to 70%+ quickly
3. **Stable Training**: Less noisy loss curves
4. **Faster Convergence**: Should reach good performance earlier

## 🐛 Troubleshooting

### Cache Build Fails
- Check TUAB data exists at `/data/datasets/external/tuab/`
- Verify EDF files are valid
- Check MNE/Autoreject installed: `pip install mne autoreject`

### Low Rejection Rate (<10%)
- Data might already be clean
- Check preprocessing parameters

### High Rejection Rate (>50%)
- Data quality issues
- Adjust Autoreject consensus parameter (make less aggressive)

### Training Not Improving
- Verify cache was built with preprocessing
- Check that `train_tuab_mne.py` is using MNE cache
- Ensure EEGPT backbone is frozen

## 📈 A/B Testing

Run both pipelines in parallel to compare:

```bash
# Original (56% AUROC)
./scripts/launch_tuab.sh

# With MNE (75-87% AUROC)
./scripts/launch_tuab_mne.sh

# Compare results
grep "AUROC" logs/tuab_*.log
grep "AUROC" logs/tuab_mne_*.log
```

## 🎯 Success Criteria

Training is successful when:
- ✅ Eval AUROC reaches 75%+ (vs 56% baseline)
- ✅ Stable training with smooth loss curves
- ✅ Consistent improvements across epochs
- ✅ Rejection statistics show 20-40% cleaning

## 📚 References

- MNE-Python: https://mne.tools/stable/
- Autoreject: https://autoreject.github.io/stable/
- EEGPT Paper: Expected 87% AUROC on TUAB

## 🔄 Next Steps

After validating improvements:
1. Apply same preprocessing to TUEV dataset
2. Optimize hyperparameters for best performance
3. Consider two-stage Autoreject (light → ICA → final)
4. Add spectral features if needed

---

**Implementation verified against official documentation and source code.**