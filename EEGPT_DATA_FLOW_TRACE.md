# EEGPT Data Flow Trace - Complete Pipeline Analysis

**Purpose**: Trace exact data transformations from raw TUEV input to final predictions to identify any processing issues.

## 📊 Data Flow Overview

```
Raw TUEV EDF → MNE Loading → Preprocessing → Event Extraction → 
Channel Mapping → Scaling → Model Input → EEGPT → Predictions
```

## Stage 1: Raw Data Loading
**File**: `src/brain_go_brrr/infra/preprocessing/tuev_event_extractor.py`

### Input
- **Format**: EDF files from `data/datasets/tuev/v2.0.1/edf/train/`
- **Channels**: 23 channels (old 10-20 system naming)
- **Sampling**: Variable (usually 256 Hz)
- **Scale**: Microvolts (μV) in EDF

### MNE Loading (`extract_events()` line 108)
```python
raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
```
- **Output scale**: Volts (MNE auto-converts from μV)
- **Verification**: `raw.get_data()` returns values ~1e-5 to 1e-4 (Volts)

## Stage 2: Preprocessing Pipeline

### 2.1 Bandpass Filter (line 112-115)
```python
raw.filter(l_freq=0.1, h_freq=75, method='iir', verbose=False)
```
- **Type**: Butterworth IIR
- **Range**: 0.1-75 Hz
- **Purpose**: Remove DC drift and high-frequency noise

### 2.2 Notch Filter (line 118-119)
```python
raw.notch_filter(freqs=50, method='iir', verbose=False)
```
- **Target**: 50 Hz powerline noise
- **Method**: IIR notch

### 2.3 Resampling (line 122-125)
```python
if raw.info['sfreq'] != 200:
    raw.resample(200, verbose=False)
```
- **Target**: 200 Hz (not 256 Hz as paper states!)
- **Window**: 5 seconds = 1000 samples

### 2.4 Channel Selection & Mapping (line 128-149)
```python
CHANNEL_MAPPING = {
    'EEG T3-REF': 'T7', 'EEG T4-REF': 'T8',
    'EEG T5-REF': 'P7', 'EEG T6-REF': 'P8',
    # ... etc
}
TARGET_CHANNELS = ['FP1','FP2','F7','F3','FZ','F4','F8',
                   'T7','C3','CZ','C4','T8',
                   'P7','P3','PZ','P4','P8','O1','O2','OZ']
```
- **Issue Found**: Maps 23→20 channels
- **Missing**: A1, A2, FPZ from original

### 2.5 Event Window Extraction (line 166-189)
```python
# Buffer calculation
buffer_duration = 10.0  # seconds
start_buffer = start_time - buffer_duration
end_buffer = end_time + buffer_duration

# Triple concatenation (line 185-189)
signal_before = raw_buffer.get_data()[:, :signal_data.shape[1]]
signal_after = raw_buffer.get_data()[:, -signal_data.shape[1]:]
triple_signal = np.concatenate([signal_before, signal_data, signal_after], axis=1)
```
- **Window**: 5s event + 10s buffer each side = 25s total
- **Triple concat**: Creates 15s (3×5s) context window
- **Final**: Center 5s extracted = 1000 samples @ 200Hz

## Stage 3: Data Scaling & Format

### 3.1 Scale Conversion (line 204-207)
```python
# Convert V to μV then divide by 100
signal = signal * 1e6 / 100  # V → μV → μV/100
```
- **Input**: ~1e-5 V (from MNE)
- **Output**: ~0.1 (μV/100 scale)
- **Range**: Typically -1 to +1 after scaling

### 3.2 Final Tensor Shape
```python
return torch.tensor(signal, dtype=torch.float32), label
```
- **Shape**: [20, 1000] (20 channels, 1000 time points)
- **Type**: float32
- **Scale**: μV/100

## Stage 4: Model Input Preparation
**File**: `experiments/eegpt_linear_probe/train_tuev_events.py`

### 4.1 Dataset Creation (line 232-242)
```python
class SimpleEEGDataset(Dataset):
    def __getitem__(self, idx):
        signal, label = self.data[idx]
        # signal shape: [20, 1000]
        return signal, label
```

### 4.2 Batch Collation (line 317)
```python
DataLoader(dataset, batch_size=32, shuffle=True)
# Batch shape: [32, 20, 1000]
```

## Stage 5: EEGPT Model Processing

### 5.1 Channel Mapper (line 127-142)
```python
self.channel_mapper = nn.Sequential(
    Conv2dWithConstraint(23, 20, kernel_size=1),  # 23→20 mapping
    nn.BatchNorm2d(20),
    nn.GELU(),
    nn.Conv2d(20, 20, kernel_size=(1, 55), groups=20),  # Temporal
    nn.BatchNorm2d(20),
    nn.Dropout(0.8)
)
```
- **Issue**: Expects 23 channels but gets 20!
- **Fix Applied**: Changed to Conv2d(20, 20, ...)

### 5.2 EEGPT Encoder Input
```python
# After channel mapper
x = x.view(batch_size, 20, -1)  # [B, 20, 1000]
features = self.eegpt_encoder.forward_encoder_only(x)
# Output: [B, 15, 4, 512] summary tokens
```

### 5.3 Feature Processing
```python
features = features.flatten(1)  # [B, 30720]
output = self.classifier(features)  # [B, 6]
```

## 🔴 CRITICAL ISSUES IDENTIFIED

### Issue 1: Sampling Rate Mismatch
- **Paper claims**: 256 Hz
- **Our code**: 200 Hz
- **Impact**: Different temporal resolution

### Issue 2: Channel Count Confusion
- **Extractor outputs**: 20 channels
- **Channel mapper expects**: 23 channels
- **Fixed but suspicious**: Why the mismatch?

### Issue 3: Window Size Discrepancy
- **Paper mentions**: 10s windows
- **Our implementation**: 5s windows (1000 samples @ 200Hz)
- **Could affect**: Temporal pattern learning

### Issue 4: Scale Verification Needed
```python
# TODO: Verify at each stage
print(f"After MNE load: {raw.get_data().mean():.6f} ± {raw.get_data().std():.6f}")
print(f"After scaling: {signal.mean():.6f} ± {signal.std():.6f}")
print(f"Model input: {x.mean():.6f} ± {x.std():.6f}")
```

## 📝 Validation Checklist

- [ ] Verify 200 Hz vs 256 Hz impact
- [ ] Check if 20 vs 23 channels matters
- [ ] Confirm μV/100 scaling is correct
- [ ] Test with paper's exact 10s windows
- [ ] Validate channel ordering matches paper
- [ ] Check if triple concatenation is correct

## 🎯 Next Steps

1. **Add logging at each stage** to verify data values
2. **Compare with reference implementation** when available
3. **Test with 256 Hz resampling** instead of 200 Hz
4. **Try 10s windows** as paper suggests

## Summary

The pipeline is mostly correct but has several discrepancies with the paper:
1. 200 Hz instead of 256 Hz sampling
2. 5s instead of 10s windows  
3. 20 channels throughout (not 23→20 conversion)
4. Triple concatenation behavior unclear

These differences could explain the 22% vs 62% BAC gap.