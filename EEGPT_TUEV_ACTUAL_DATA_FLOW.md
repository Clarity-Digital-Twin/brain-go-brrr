# EEGPT TUEV ACTUAL Data Flow - VERIFIED Current Implementation

**Created**: September 11, 2025  
**Purpose**: Document the EXACT data flow in our CURRENT TUEV implementation  
**Status**: VERIFIED against actual running code

## 📊 Complete Data Flow Pipeline

```
Raw EDF → TUEVEventExtractor → Cache → TUEVEventDataset → 
DataLoader → TUEVModel → EEGPT → Predictions
```

## Stage 1: Data Extraction & Caching
**File**: `src/brain_go_brrr/infra/preprocessing/tuev_event_extractor.py`

### Raw Data Loading
```python
# Load EDF with MNE (Volts)
raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
```
- **Input**: EDF files from `data/datasets/tuev/v2.0.1/edf/{train,eval}/`
- **Output**: MNE Raw object in VOLTS (MNE auto-converts from μV)

### Preprocessing Pipeline
```python
# Bandpass + notch (EEGPT reference)
raw.filter(l_freq=0.1, h_freq=75.0, verbose=False)
raw.notch_filter(freqs=50.0, verbose=False)

# Resample to 200 Hz (NOT 256!)
if raw.info['sfreq'] != self.target_fs:  # self.target_fs=200
    raw.resample(self.target_fs, verbose=False)
```

### Channel Handling (23 channels)
```python
# Select referential channels in exact order; zero-pad if missing
available_channels = [ch for ch in self.TUEV_CHANNELS_REF if ch in raw.ch_names]
raw.pick_channels(available_channels, ordered=True)

data = raw.get_data()
if len(available_channels) < 23:
    full = np.zeros((23, data.shape[1]), dtype=np.float32)
    for i, ch in enumerate(available_channels):
        full[self.TUEV_CHANNELS_REF.index(ch)] = data[i].astype(np.float32)
    data = full
else:
    data = data.astype(np.float32)
```
- **Target**: 23 referential channels (FP1, FP2, ... A1, A2)
- **NO channel remapping** (T3→T7 etc) at this stage

### Event Extraction with Triple Concatenation
```python
fs = int(self.target_fs)              # 200
samples_per_segment = 5 * fs          # 1000
times = np.arange(data.shape[1]) / fs
offset = data.shape[1]
extended = np.concatenate([data, data, data], axis=1)

start_idx = int(np.searchsorted(times, start_sec, side='left'))
end_idx = int(np.searchsorted(times, end_sec, side='left'))
if (end_idx - start_idx) != fs:       # enforce 1s event length
    end_idx = start_idx + fs

cut_start = offset + start_idx - 2 * fs
cut_end = offset + end_idx + 2 * fs
if (cut_end - cut_start) != samples_per_segment:  # enforce exact 5s
    cut_end = cut_start + samples_per_segment

segment = extended[:, cut_start:cut_end]         # (23, 1000) in Volts
```
- **Window**: 5 seconds @ 200Hz = 1000 samples
- **Shape**: [23, 1000]
- **Scale**: Still in VOLTS from MNE

## Stage 2: Dataset Loading
**File**: `src/brain_go_brrr/infra/data/tuev_event_dataset.py`

### Cache Structure
```python
# Default cache location (overridable via --cache_dir)
self.cache_dir = cache_dir or root_dir / 'cache' / 'tuev_event_segments'
```

### Class Mapping
```python
# Line 47-54: TUEV 6-class mapping
self.class_mapping = {
    'spsw': 0,  # spike and slow wave
    'gped': 1,  # generalized periodic epileptiform discharge
    'pled': 2,  # periodic lateralized epileptiform discharge
    'eyem': 3,  # eye movement
    'artf': 4,  # artifact
    'bckg': 5,  # background
}
```

### Data Loading from Cache
```python
# __getitem__: safe load with weights_only=True
data = torch.load(cache_file, weights_only=True)
return data['x'], data['y']  # x: (23, 1000) in Volts
```

### Split Handling
- Uses official directories `edf/train` and `edf/eval` if present.
- Otherwise applies an 80/20 subject-level split with fixed seed `4523` to avoid leakage.

## Stage 3: Model Input Pipeline
**File**: `experiments/eegpt_linear_probe/train_tuev_events.py`

### DataLoader Creation
```python
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,  # default 64 (configurable)
    shuffle=True,  # Natural distribution, NO balancing
    num_workers=args.num_workers,
    pin_memory=args.pin_memory,
    persistent_workers=(args.num_workers > 0) and args.persistent_workers,
)
```
- **Batch shape**: `[batch_size, 23, 1000]` in Volts

### TUEVModel Processing
```python
class TUEVModel(nn.Module):
    def forward(self, x):
        # x: (batch, 23, 1000) in Volts from dataset
        
        # Critical scaling: V → μV → μV/100
        x = x * 1e6 / 100
        
        x = x.unsqueeze(2)  # [B, 23, 1, 1000]
        x = self.mapper(x)  # TUEVChannelMapper: 23→20
        x = x.squeeze(2)    # [B, 20, 1000]
        
        x = self.classifier(x)  # [B, 6]
        return x
```

## Stage 4: TUEVChannelMapper
**File**: `src/brain_go_brrr/infra/ml_models/channel_mapper.py`

### Architecture (as implemented)
```python
# Spatial conv then temporal depthwise conv
self.spatial_conv = nn.Sequential(
    Conv2dWithConstraint(23, 20, kernel_size=1, bias=True),
    nn.BatchNorm2d(20),
    nn.GELU(),
)
self.temporal_conv = nn.Sequential(
    nn.Conv2d(20, 20, kernel_size=(1, 55), groups=20, padding=(0, 27), bias=False),
    nn.BatchNorm2d(20),
    nn.Dropout(0.8),
)
```

## Stage 5: TUEVClassifierHead
**File**: `experiments/eegpt_linear_probe/train_tuev_events.py`

### Processing Pipeline
```python
def forward(self, x):
    # x: [B, 20, 1000] in μV/100 scale
    
    # Parity reshape (functional no-op)
    b = x.shape[0]
    x = x.reshape(b, 20, 5, 200).reshape(b, 20, 1000)
    
    # EEGPT feature extraction (return all temporal)
    features_all = self.eegpt(x, chan_ids=self.chan_ids, return_all_temporal=True)
    # Shape: [B, 15, 4, 512] for 1000 samples with stride=64
    
    # Flatten and classify
    features = features_all.reshape(b, -1)  # [B, 30720]
    logits = self.classifier(features)  # [B, 6]
```

### EEGPT Wrapper Settings
- **Normalize**: FALSE (explicitly disabled to match reference)
- **Time steps**: 1000 (true parity, no padding)
- **Patch stride**: 64 (for 1000 samples → 15 patches)
- **Channels (order)**: `['FP1','FPZ','FP2','F7','F3','FZ','F4','F8','T7','C3','CZ','C4','T8','P7','P3','PZ','P4','P8','O1','O2']`
- **DropPath**: forced `0.0` in model kwargs

## 🔍 Critical Implementation Details VERIFIED

### ✅ Correct Implementation
1. **200 Hz sampling** - Matches reference
2. **5s windows (1000 samples)** - Matches reference
3. **μV/100 scaling** - Implemented inside `TUEVModel.forward`
4. **23→20 channel mapping** - TUEVChannelMapper
5. **Triple concatenation** - Implemented in extractor via offset buffer
6. **No normalization** - Raw μV/100 to model
7. **No class balancing** - Natural distribution
8. **30720 features** - 15×4×512 flattened

### 📊 Data Scale Verification
- **After MNE load**: ~1e-5 to 1e-4 V (typical EEG)
- **After μV/100 scaling**: ~0.1 to 1.0 range
- **To EEGPT**: Raw μV/100 (normalization disabled)

## 🎯 Summary

The pipeline is CORRECTLY implemented and matches the reference:
1. Extract 5s @ 200Hz events with triple concatenation
2. Load cached segments in VOLTS
3. Scale to μV/100 in model forward pass
4. Map 23→20 channels via Conv2dWithConstraint
5. Extract 30720 features via EEGPT
6. Classify with LinearWithConstraint head

**The 22% BAC is the REAL result with this exact implementation.**
