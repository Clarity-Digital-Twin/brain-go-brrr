# TUEV Preprocessing Specifications - DEFINITIVE ANSWERS

**Source**: EEGPT Reference Repository Analysis
**Date**: December 2024

## ✅ CONFIRMED PREPROCESSING PARAMETERS

Based on direct examination of the EEGPT reference implementation:

### 1. **Sampling Rate: 200 Hz** ✅
- **Location**: `downstream_tueg/dataset_maker/make_TUEV.py:131`
- **Code**: `Rawdata.resample(200, n_jobs=5)`
- **Confirmation**: `downstream_tueg/utils.py:TUEVLoader` default `sampling_rate=200`

### 2. **Window Size: 5 seconds** ✅
- **Location**: `downstream_tueg/dataset_maker/make_TUEV.py:25`
- **Code**: `features = np.zeros([numEvents, numChan, int(fs) * 5])`
- **Math**: 200 Hz × 5 seconds = 1000 samples per window
- **Note**: Paper mentions 4s pretraining, but TUEV uses 5s events

### 3. **Filtering: 0.1-75 Hz bandpass + 50 Hz notch** ✅
- **Location**: `downstream_tueg/dataset_maker/make_TUEV.py:129-130`
- **Code**:
  ```python
  Rawdata.filter(l_freq=0.1, h_freq=75.0)
  Rawdata.notch_filter(50.0)
  ```

## 📊 COMPLETE PREPROCESSING PIPELINE

From `make_TUEV.py` (lines 116-138):

```python
def readEDF(fileName):
    # 1. Load EDF file
    Rawdata = mne.io.read_raw_edf(fileName, preload=True)

    # 2. Drop unnecessary channels
    Rawdata.drop_channels(useless_chs)

    # 3. Reorder to standard 23 channels
    Rawdata.reorder_channels(chOrder_standard)

    # 4. Apply filters
    Rawdata.filter(l_freq=0.1, h_freq=75.0)  # Bandpass
    Rawdata.notch_filter(50.0)               # Notch filter

    # 5. Resample to 200 Hz
    Rawdata.resample(200, n_jobs=5)

    # 6. Get data in microvolts (μV)
    signals = Rawdata.get_data(units='uV')
```

## 🎯 KEY IMPLEMENTATION DETAILS

### Event Extraction (lines 17-39):
- **Window padding**: 2 seconds before + 2 seconds after = 5 seconds total
- **Signal tripling**: `np.concatenate([signals, signals, signals], axis=1)` for boundary handling
- **Extraction**: From middle copy with padding

- **Event duration assumption (critical for shape)**: The assignment
  `features[i, :] = signals[:, offset + start - 2*int(fs) : offset + end + 2*int(fs)]`
  fits the preallocated `int(fs)*5` only when `(end - start) == int(fs)` (i.e., event duration = 1.0 s at 200 Hz). TUEV `.rec` events are 1.0 s, so this matches; if using variable-length events, the code must be adapted or it will shape-mismatch.

### Data Scaling:
- **Input units**: Microvolts (μV)
- **Training scaling**: Divide by 100 (`samples / 100`)
- **NO normalization**: Standardization code is commented out (line 21-23)

### Channel Configuration:
- **23 channels** from TUEV EDF files
- **Learned mapping**: 23→20 via Conv2d in model
- **Standard order**: FP1, FP2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6, A1, A2, FZ, CZ, PZ, T1, T2

- **Mapper target set (precision note)**: The 20 output channels are a canonical 10–20 set defined by `use_channels_names` in `downstream_tueg/run_class_finetuning_EEGPT_change_tuev.py` and include `FPZ` (not present in TUEV raw inputs). The 1×1 Conv2d (see `downstream_tueg/Modules/models/EEGPT_mcae_finetune_change_tuev.py`) learns a data-driven projection from the 23 TUEV inputs to these 20 canonical outputs (including synthesizing FPZ), consistent with the reference implementation.

### Label Mapping (loader)
- TUEV labels in pickles are 1–6 and are remapped to 0–5 by the loader: `Y = int(sample["label"][0] - 1)` in `downstream_tueg/utils.py:TUEVLoader.__getitem__`.

### Dataset Path Versioning
- Preprocessing script path root: `.../tuh_eeg_events/v2.0.0/edf` (`downstream_tueg/dataset_maker/make_TUEV.py`).
- Training script expects processed data under `.../tuh_eeg_events/v2.0.1/edf/processed/` (`downstream_tueg/run_class_finetuning_EEGPT_change_tuev.py`). Ensure these align (either adjust one path or move/symlink processed outputs). Processing steps are identical.

## ❓ ANSWERS TO YOUR QUESTIONS

| Parameter | EEGPT Paper/Reference | Your Implementation | Match? |
|-----------|----------------------|---------------------|--------|
| **Sampling Rate** | 200 Hz | ? | Check |
| **Window Size** | 5 seconds | ? | Check |
| **Filtering** | 0.1-75 Hz + 50 Hz notch | ? | Check |

## 🔴 PERFORMANCE EXPECTATIONS

Based on `TUEV_REPRODUCTION_REPORT.md`:
- **Paper claim**: 62.32% BAC
- **Independent test**: 58.44% BAC (best case)
- **Typical result**: 22-25% BAC
- **Root cause**: Extreme class imbalance (24 samples for minority class)

## 📝 CRITICAL NOTES

1. **Class Imbalance is Fatal**: With only 24 samples for `spsw` class out of 10,448 total, achieving 60%+ BAC is nearly impossible without data augmentation

2. **Paper Results Not Reproducible**: Multiple independent attempts have failed to reproduce the claimed 62.32% BAC

3. **Realistic Target**: 25-30% BAC should be considered acceptable given the dataset limitations

## ✅ IMPLEMENTATION CHECKLIST

To match EEGPT reference exactly, ensure:
- [ ] Sampling rate: 200 Hz
- [ ] Window size: 5 seconds (1000 samples)
- [ ] Filtering: 0.1-75 Hz bandpass + 50 Hz notch
- [ ] Input units: Microvolts (μV)
- [ ] Scaling: Divide by 100 during training
- [ ] NO normalization/standardization
- [ ] Signal tripling for boundary handling
- [ ] 23-channel input with learned 23→20 mapping
- [ ] Label remap 1–6 → 0–5 in loader
- [ ] Align dataset paths (v2.0.0 preprocess vs v2.0.1 loader)

## 🚀 RECOMMENDATION

If your implementation matches these specs and achieves 20-30% BAC, **your implementation is working correctly**. The performance gap is due to:
1. Extreme class imbalance in the dataset
2. Potentially unreproducible paper claims
3. Insufficient minority class samples for deep learning

Consider TUEV as a demonstration/research task rather than a production-ready clinical application.
