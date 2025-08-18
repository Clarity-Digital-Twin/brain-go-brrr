# 🔴 TUEV CRITICAL SPECIFICATIONS - MUST FOLLOW EXACTLY

## 📊 DATASET FACTS - VERIFIED ✅

### Core Specifications (Paper vs Actual)
- **Samples**: 112,491 5-second segments (paper) | ~62,000 estimated (actual)
- **Subjects**: 288 (paper) | **370 actual** (290 train, 80 eval) ✅
  - **NOTE**: EEGPT paper used 288 subjects. Our v2.0.1 has 370 subjects (more data, potentially better)
- **Classes**: 6 (multi-class, NOT multi-label) ✅ CONFIRMED
- **Channels**: 23 @ 256 Hz (paper) | 26-27 @ 250 Hz (actual) ⚠️
  - **CRITICAL**: Must resample 250→256 Hz and select exactly 23 channels to match paper
- **Window**: 5 seconds (1280 samples @ 256Hz)
- **Source**: TUEV v2.0.1 (newer version than paper used)

### Dataset Reality Check
- **EDF Files**: 518 total (359 train, 159 eval)
- **Label Files**: 11,396 .lab files (per-channel annotations)
- **Annotation Format**: Microsecond timestamps in .lab files
- **Actual Sampling**: 250 Hz (will need resampling to 256 Hz)

### The 6 Classes (CRITICAL!)
1. **SPSW** - Spike and Sharp Wave (epileptiform)
2. **GPED** - Generalized Periodic Epileptiform Discharges
3. **PLED** - Periodic Lateralized Epileptiform Discharges
4. **EYEM** - Eye Movement (artifact)
5. **ARTF** - Artifact (other)
6. **BCKG** - Background (normal activity)

**KEY INSIGHT**: Classes 1-3 (SPSW, GPED, PLED) ARE the IEDs!

## 🎯 EXACT TRAINING CONFIGURATION

### Model Architecture (from Appendix C.2.6)
```python
# Two-layer convolution adapter
spatial_conv = Conv1d(
    in_channels=23,      # TUEV has 23 channels
    out_channels=58,     # EEGPT expects 58
    kernel_size=1        # 1x1 spatial convolution
)

temporal_conv = Conv1d(
    in_channels=58,
    out_channels=58,
    kernel_size=55,      # CRITICAL: (1, 55) for TUEV!
    groups=58,           # Depthwise convolution
    padding='same'
)
```

### Training Parameters (EXACT from paper)
- **Batch Size**: 500 (not 100 like TUAB!)
- **Learning Rate**: 5e-4
- **Optimizer**: AdamW (same as TUAB)
- **Method**: Linear-probing (frozen EEGPT backbone)
- **GPU Memory**: Batch 500 requires ~8GB VRAM

### Critical Differences from TUAB

| Parameter | TUAB | TUEV | Why Different |
|-----------|------|------|---------------|
| Window | 10s → 4s | 5s | Different annotation granularity |
| Kernel | (1, 15) | **(1, 55)** | Longer temporal context needed |
| Batch | 100 | **500** | More samples available |
| Classes | 2 | **6** | Event classification |
| Samples | 409,455 | 112,491 | Smaller but more specific |

## 📈 PAPER PERFORMANCE (Target)

### EEGPT Results (Table 3)
- **Balanced Accuracy**: 0.6232 ± 0.0114
- **Weighted F1**: 0.8187 ± 0.0063
- **Cohen's Kappa**: 0.6351 ± 0.0134
- **Improvement over BIOT**: +9.5% BAC, +6.9% F1

### Comparison
| Model | Size | BAC | Weighted F1 | Kappa |
|-------|------|-----|-------------|-------|
| BIOT | 3.2M | 0.5281 | 0.7492 | 0.5273 |
| EEGPT-Tiny | 4.7M | 0.5670 | 0.7535 | 0.5085 |
| **EEGPT** | **25M** | **0.6232** | **0.8187** | **0.6351** |

## ⚠️ CRITICAL IMPLEMENTATION NOTES

### 1. Data Processing Requirements
- **Resample**: 250 Hz → 256 Hz (actual data is 250 Hz!)
- **Channel selection**: Pick 23 from 26-27 available
- **Window extraction**: Parse .lab files for 5-second segments
- **Label mapping**: Use .lab files (microsecond precision)

### 2. Data Split Strategy (CRITICAL FOR COMPARISON!)
- **MUST Follow BIOT strategy** - Paper states: "For the data splitting of TUAB and TUEV, we strictly follow the same strategy as BIOT"
- **NOT LOSO** - LOSO is only for BCIC-2A, BCIC-2B, etc. NOT for TUEV!
- **Use existing split**: 290 train / 80 eval subjects (subject-level, not random)
- Train/eval split preserves subject boundaries
- **WARNING**: Using any other split strategy will make results incomparable to paper!

### 2. Channel Mapping
TUEV uses 23 channels, EEGPT expects 58:
- Must use 1x1 conv adapter
- TCP montage (Temporal Central Parasagittal)
- Different from 10-20 system

### 3. Temporal Kernel Size (CRITICAL!)
```python
# WRONG for TUEV:
kernel_size = 15  # This is for TUAB

# CORRECT for TUEV:
kernel_size = 55  # Much larger receptive field
```

### 4. Why Kernel 55?
- 5-second windows = 1280 samples
- Kernel 55 ≈ 214ms receptive field
- Captures event morphology (spikes are 20-200ms)
- Larger than TUAB because events span longer

## 🔧 EXACT IMPLEMENTATION

### Dataset Loader
```python
class TUEVDataset(Dataset):
    """TUEV with EXACT paper specifications."""
    
    def __init__(self, root_dir, split='train'):
        self.window_size = 5.0  # MUST be 5 seconds
        self.sample_rate = 256  # MUST be 256 Hz
        self.n_channels = 23    # MUST be 23
        self.n_classes = 6      # MUST be 6
        
        # CRITICAL: 5 * 256 = 1280 samples per window
        self.n_samples = 1280
```

### Model Configuration
```python
config = {
    'data': {
        'window_size': 5.0,      # NOT 4.0!
        'n_channels': 23,        # NOT 20!
        'batch_size': 500,       # NOT 100!
    },
    'model': {
        'spatial_kernel': 1,     # 1x1 conv
        'temporal_kernel': 55,   # NOT 15!
        'n_classes': 6,          # NOT 2!
    },
    'training': {
        'lr': 5e-4,              # Same as paper
        'optimizer': 'AdamW',
        'method': 'linear_probe', # Frozen backbone
    }
}
```

## 🚨 COMMON MISTAKES TO AVOID

1. ❌ Using 4-second windows (that's TUAB)
2. ❌ Using kernel size 15 (that's TUAB)
3. ❌ Using batch size 100 (memory limited for TUAB)
4. ❌ Using 20 channels (that's TUAB processed)
5. ❌ Using LOSO validation (that's for BCIC)
6. ❌ Binary classification (TUEV is 6-class!)
7. ❌ Using 10-second raw windows (need 5s)

## ✅ VALIDATION CHECKLIST

Before training, verify:
```python
assert window_size == 5.0, "TUEV uses 5s windows"
assert n_channels == 23, "TUEV has 23 channels"
assert kernel_size == 55, "TUEV needs kernel 55"
assert batch_size == 500, "TUEV uses batch 500"
assert n_classes == 6, "TUEV has 6 event types"
assert samples_per_window == 1280, "5s * 256Hz = 1280"
```

## 📐 METRICS TO TRACK

### Primary (from paper)
1. **Balanced Accuracy** (main metric)
2. **Weighted F1** (handles imbalance)
3. **Cohen's Kappa** (agreement measure)

### Secondary (useful)
4. Per-class F1 scores
5. Confusion matrix
6. IED vs non-IED accuracy (SPSW+GPED+PLED vs others)

## 🎯 SUCCESS CRITERIA

You've succeeded when:
- ✅ Balanced Accuracy ≥ 0.62
- ✅ Weighted F1 ≥ 0.81
- ✅ Cohen's Kappa ≥ 0.63
- ✅ Better than BIOT baseline (0.53 BAC)

## 📝 PAPER QUOTES (Direct Evidence)

> "TUEV is a subset of TUEG that contains annotations of EEG segments as one of six classes" (p.19)

> "The EEG signals contain 23 channels at 256 Hz and are segmented into 112,491 5-second samples" (p.20)

> "The convolution kernel size for TUAB was (1, 15), and for TUEV, it was (1, 55)" (p.20)

> "Due to GPU memory limitations, the batch size for TUAB was 100, and for TUEV, it was 500" (p.20)

> "We achieved a 9.5% performance improvement" compared to BIOT (p.8)

---

**FOLLOW THIS EXACTLY OR TRAINING WILL FAIL!**