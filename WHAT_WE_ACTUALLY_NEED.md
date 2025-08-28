# 🎯 WHAT WE ACTUALLY FUCKING NEED (FROM THE LITERATURE)

## EEGPT PAPER REQUIREMENTS (What ACTUALLY Works)

### Model Architecture (From Paper)
- **Input**: 4-second windows @ 256Hz = 1024 samples
- **Channels**: 19-20 standard 10-20 channels
- **Preprocessing**: Bandpass 0.5-45Hz, z-score normalization
- **Output**: 2048-dim features (4 × 512 summary tokens)
- **Training**: Frozen encoder + linear probe

### TUAB Dataset Requirements
- **Channels**: 19 (no Fz) - Standard 10-20 without Fz
- **Normalization**: MUST BE N(0,1) - z-score per window
- **Window**: 4 seconds (1024 samples @ 256Hz)
- **Classes**: Binary (normal=0, abnormal=1)
- **Target**: 0.87 AUROC (paper benchmark)

### TUEV Dataset Requirements  
- **Channels**: 20 (with Fz, no Fpz)
- **Classes**: 6 (SPSW, GPED, PLED, EYEM, ARTF, BCKG)
- **Target**: 62.32% balanced accuracy (Table 13)
- **Same**: 4s windows, 256Hz, z-score normalization

## WHAT WE ACTUALLY NEED (MINIMAL)

### 1. ONE Dataset Class
```python
class EEGDataset:
    - Load EDF files
    - Resample to 256Hz
    - Extract 4s windows
    - Z-score normalize
    - Return (x, y) tensors
```

### 2. ONE EEGPT Model Wrapper
```python
class EEGPTModel:
    - Load checkpoint
    - Forward pass through encoder
    - Extract 2048-dim features
    - That's it!
```

### 3. ONE Linear Probe
```python
class LinearProbe:
    - Input: 2048-dim features
    - Output: n_classes
    - Just nn.Linear(2048, n_classes)
```

### 4. ONE Preprocessing Function
```python
def preprocess_eeg(raw):
    - Bandpass 0.5-45Hz
    - Resample to 256Hz
    - Z-score normalize
    - Return numpy array
```

## WHAT WE HAVE (THE DISASTER)

### In src/infra/data/:
- `tuab_dataset.py` - BASE VERSION (probably works)
- `tuab_cached_dataset.py` - WHY?!
- `tuab_enhanced_dataset.py` - WHAT IS ENHANCED?!

### In src/infra/ml_models/:
- `eegpt_wrapper.py` - Used by experiments
- `eegpt_compat.py` - Used by API
- `eegpt_model.py` - Another one?!
- `eegpt_probe_unified.py` - WTF is unified?!
- `eegpt_architecture.py` - Just architecture definition?
- Plus more shit!

### In experiments/:
- REIMPLEMENTED EVERYTHING
- FORGOT NORMALIZATION
- DOESN'T USE SRC

## THE TRUTH: We Need 4 Files Total

1. **dataset.py** - Load data, normalize, window
2. **model.py** - EEGPT wrapper
3. **train.py** - Training loop
4. **config.yaml** - Hyperparameters

**THAT'S IT. EVERYTHING ELSE IS BULLSHIT.**

## What's Probably Wrong in Our Code

### 1. Normalization Issues
- Some normalize, some don't
- Some normalize wrong (per-channel vs per-window)
- Some forget entirely

### 2. Channel Confusion
- TUAB needs 19 (no Fz)
- TUEV needs 20 (with Fz)
- Some code assumes 19, some 20, some doesn't care

### 3. Feature Extraction
- Paper says 2048 features (4×512)
- Some code uses 512
- Some uses raw 32768
- Nobody knows what's right

### 4. Window Size
- MUST be 4 seconds
- Some code uses 30 seconds
- Some uses variable
- Only experiments/ got it right

## CRITICAL FACTS FROM LITERATURE

1. **EEGPT was pretrained on 4-second windows**
   - Using different window sizes BREAKS IT
   
2. **Normalization is CRITICAL**
   - Without it, model sees zeros
   - Must be z-score (mean=0, std=1)
   
3. **Channel order matters**
   - Model expects specific order
   - Mixing channels = garbage output
   
4. **Features are 2048-dim**
   - NOT 512 (that's per token)
   - NOT 32768 (that's raw patches)
   - It's 4 × 512 = 2048

## THE ONE TRUE PIPELINE

```
EDF File
    ↓
Resample to 256Hz
    ↓
Bandpass 0.5-45Hz
    ↓
Extract 4s windows
    ↓
Z-score normalize
    ↓
EEGPT encoder
    ↓
2048 features
    ↓
Linear probe
    ↓
Predictions
```

## FILES TO KEEP

### In src/:
1. `tuab_dataset.py` (if it works)
2. `eegpt_wrapper.py` (being used)
3. That's it

### In experiments/:
1. Training scripts
2. Configs
3. Nothing else

## FILES TO DELETE (IMMEDIATELY)

### In src/:
- tuab_cached_dataset.py
- tuab_enhanced_dataset.py
- eegpt_model.py
- eegpt_probe_unified.py
- eegpt_compat.py (maybe)
- All other EEGPT variants

### In experiments/:
- All datasets (use src/)
- All preprocessing (use src/)
- 9 of 10 documentation files

## THE BOTTOM LINE

We need 4 files. We have 40+. 
90% is redundant garbage.
The 10% that works is scattered across two parallel universes.

This is what happens when you don't read the fucking paper first.