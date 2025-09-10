# TUEV Paper-Parity Implementation Plan

## 🚨 CRITICAL UPDATE (Dec 10, 2024): CLASS IMBALANCE FOUND 🚨

**Current Status**: Implementation complete but BAC=0.18 due to severe class imbalance
**Root Cause**: Class 0 (spsw) has only 19/2695 samples (0.7%), Class 5 (bckg) has 1168/2695 (43%)
**Required Fix**: Add WeightedRandomSampler to training script

## Executive Summary

This document provides the **exact** implementation plan to achieve EEGPT paper parity for TUEV (62.32% BAC) using our existing Brain-Go-Brrr infrastructure. We will NOT create a parallel universe - everything integrates with our current `src/` components.

Verified reference behavior (paths):
- No bipolar montage: `reference_repos/EEGPT/downstream_tueg/dataset_maker/make_TUEV.py` → `convert_signals` exists but is commented out.
- 23→20 channel mapping via conv: `reference_repos/EEGPT/downstream_tueg/Modules/models/EEGPT_mcae_finetune_change_tuev.py` (chan conv stack) and `run_class_finetuning_EEGPT_change_tuev.py` (`use_chan_conv=True`, `img_size=[20,1000]`).
- 5 s @ 200 Hz segments: `make_TUEV.py` (`readEDF` filters+notch+resample to 200; `BuildEvents` features length `int(fs)*5`).
- Event-only extraction: maker writes event pickles in `processed_{train,eval,test}`, loaded by `TUEVLoader` in `downstream_tueg/utils.py`.
- Unweighted loss with smoothing=0.1: `run_class_finetuning_EEGPT_change_tuev.py` uses `LabelSmoothingCrossEntropy(smoothing=0.1)`.
- Warmup/layer decay used: `downstream_tueg/finetune_TUEV_EEGPT.sh` sets `warmup_epochs=5`, `layer_decay=0.65`, `lr=5e-4`, `weight_decay=0.05`, `batch_size=400`, `epochs=30`.

## CRITICAL CHANNEL MISMATCH DISCOVERED (Sep 10, 2025)

### THE BUG THAT BROKE TRAINING
We were passing 20 channels to EEGPT but the model was configured for 19 or 58 channels!
- **Input**: 23 TUEV channels → mapper → 20 channels
- **EEGPT Config**: Was using default channel count (not 20!)
- **Result**: Patch embedding dimension mismatch crash

### THE FIX
Configure EEGPT with exactly 20 channel names for TUEV:
```python
# TUEV-specific channel config
TUEV_20_CHANNELS = ['FP1','FPZ','FP2','F7','F3','FZ','F4','F8',
                    'T7','C3','CZ','C4','T8','P7','P3','PZ','P4','P8','O1','O2']

# Pass to EEGPT
model = EEGPTModel(n_channels=TUEV_20_CHANNELS, time_steps=1000)
```

## Critical Corrections from Senior Review

### ✅ VERIFIED: No Bipolar Montage
```python
# In make_TUEV.py line 143:
#signals = convert_signals(signals, Rawdata)  # COMMENTED OUT!
```
- EEGPT uses **referential "-REF" channels**, NOT bipolar
- The bipolar function exists but is unused

### ✅ VERIFIED: 23→20 Channel Mapping
```python
# From run_class_finetuning_EEGPT_change_tuev.py
ch_names = ['EEG FP1-REF', 'EEG FP2-REF', ..., 'EEG T1-REF', 'EEG T2-REF']  # 23 channels
# Later uses learned conv1d(23, 20) to map to model input
```

### ✅ VERIFIED: Unweighted Loss
```python
# No class weights, just:
criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
```

## Implementation Architecture (Using Our Existing Code)

Integration map (implemented):
- Data (event-only): `src/brain_go_brrr/infra/data/tuev_event_dataset.py` (23×1000 @ 200 Hz, `_ch000.lab` parser, subject split fallback).
- Preproc (extractor): `src/brain_go_brrr/infra/preprocessing/tuev_event_extractor.py` (filters/notch/resample; REF channels; −2..+3 s).
- Models: `src/brain_go_brrr/infra/ml_models/channel_mapper.py` (23→20 mapper), `src/brain_go_brrr/infra/ml_models/eegpt_architecture.py` (parity stride support).
- Experiments (single trainer): `experiments/eegpt_linear_probe/train_tuev_events.py` with `--use_parity` (native 1000) or fallback (pad 1000→1024).
- Tests: unit/integration updated to validate extractor/dataset/mapper and dataset cache shape/metadata.

### Phase 1: Event Segment Extractor (DONE)

**Location**: `src/brain_go_brrr/infra/preprocessing/tuev_event_extractor.py`

```python
from pathlib import Path
import numpy as np
import mne
import torch
from typing import List, Dict, Tuple

class TUEVEventExtractor:
    """Extract 5-second segments around annotated events per EEGPT paper."""
    
    # EXACT channel order from reference
    TUEV_CHANNELS_REF = [
        'EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 
        'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 
        'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 
        'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 
        'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 
        'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF'
    ]
    
    def __init__(self, 
                 target_fs: int = 200,  # EEGPT uses 200Hz not 256Hz!
                 segment_duration: float = 5.0,  # 5 seconds
                 tmin: float = -2.0,  # 2 seconds before event
                 tmax: float = 3.0):  # 3 seconds after event
        self.target_fs = target_fs
        self.segment_duration = segment_duration
        self.tmin = tmin
        self.tmax = tmax
        
    def extract_segments(self, 
                        edf_path: Path, 
                        annotations: List[Dict]) -> List[Tuple[np.ndarray, int]]:
        """Extract event-centered segments from EDF file.
        
        Returns:
            List of (segment, label) tuples
            segment shape: (23, 1000) - 23 channels, 5 seconds @ 200Hz
        """
        # Load with MNE (reuse our existing infrastructure)
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        
        # Filter per EEGPT: 0.1-75 Hz bandpass, 50 Hz notch
        raw.filter(l_freq=0.1, h_freq=75.0, verbose=False)
        raw.notch_filter(freqs=50.0, verbose=False)
        
        # Resample to 200 Hz (NOT 256!)
        if raw.info['sfreq'] != self.target_fs:
            raw.resample(self.target_fs, verbose=False)
            
        # Select and reorder channels to match reference
        available_channels = [ch for ch in self.TUEV_CHANNELS_REF if ch in raw.ch_names]
        raw.pick_channels(available_channels, ordered=True)
        
        # Pad with zeros if channels missing (maintain 23 channel format)
        data = raw.get_data()  # Shape: (n_available_channels, n_samples)
        if len(available_channels) < 23:
            # Create full 23-channel array with zeros
            full_data = np.zeros((23, data.shape[1]), dtype=np.float32)
            for i, ch in enumerate(available_channels):
                idx = self.TUEV_CHANNELS_REF.index(ch)
                full_data[idx] = data[i]
            data = full_data
            
        segments = []
        for annot in annotations:
            # Extract 5-second segment centered on event
            event_center = (annot['start'] + annot['end']) / 2
            start_sample = int((event_center + self.tmin) * self.target_fs)
            end_sample = int((event_center + self.tmax) * self.target_fs)
            
            # Ensure we have exactly 1000 samples
            if end_sample - start_sample != 1000:
                end_sample = start_sample + 1000
                
            if start_sample >= 0 and end_sample <= data.shape[1]:
                segment = data[:, start_sample:end_sample]  # (23, 1000)
                segments.append((segment, annot['label']))
                
        return segments
```

### Phase 2: Event Segment Dataset (DONE)

**Location**: `src/brain_go_brrr/infra/data/tuev_event_dataset.py`

```python
class TUEVEventDataset(Dataset):
    """TUEV event segment dataset for paper parity.
    
    This is DIFFERENT from our sliding window TUEVMNEDataset.
    This extracts ONLY event segments for classification.
    """
    
    def __init__(self, 
                 root_dir: Path,
                 split: str = 'train',
                 cache_dir: Path = None,
                 force_rebuild: bool = False):
        self.root_dir = Path(root_dir)
        self.split = split
        self.cache_dir = cache_dir or self.root_dir / 'cache' / 'tuev_event_segments'
        
        # Use our existing class mapping
        self.class_mapping = {
            'spsw': 0, 'gped': 1, 'pled': 2, 
            'eyem': 3, 'artf': 4, 'bckg': 5
        }
        
        if force_rebuild or not self._cache_exists():
            self._build_cache()
            
        self._load_cache()
        
    def _build_cache(self):
        """Build cache using our event extractor."""
        extractor = TUEVEventExtractor()
        
        # Get all EDF files for this split
        split_dir = self.root_dir / 'edf' / self.split
        edf_files = list(split_dir.rglob('*.edf'))
        
        # Process each file
        all_segments = []
        for edf_path in tqdm(edf_files, desc=f"Building {self.split} cache"):
            # Parse annotations (reuse our existing parser)
            annotations = self._parse_annotations(edf_path)
            
            # Extract segments
            segments = extractor.extract_segments(edf_path, annotations)
            
            # Save each segment
            for i, (segment, label) in enumerate(segments):
                segment_id = f"{edf_path.stem}_{i}"
                cache_file = self.cache_dir / self.split / f"{segment_id}.pt"
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Convert to tensor in Volts (SI units per our SSOT)
                segment_tensor = torch.from_numpy(segment).float()
                torch.save({
                    'x': segment_tensor,  # (23, 1000)
                    'y': label,
                    'id': segment_id
                }, cache_file)
                
                all_segments.append({
                    'file': cache_file.name,
                    'label': label,
                    'subject': edf_path.stem.split('_')[0]  # Extract subject ID
                })
                
        # Save index with subject-level split info
        index_file = self.cache_dir / self.split / 'index.json'
        with open(index_file, 'w') as f:
            json.dump({
                'segments': all_segments,
                'n_segments': len(all_segments),
                'class_counts': dict(Counter([s['label'] for s in all_segments])),
                'n_subjects': len(set([s['subject'] for s in all_segments])),
                'fs': 200,
                'duration': 5.0,
                'channels': 23,
                'samples': 1000
            }, f, indent=2)
            
    def __getitem__(self, idx):
        """Return (x, y) where x is (23, 1000) and y is label."""
        segment_info = self.segments[idx]
        cache_file = self.cache_dir / self.split / segment_info['file']
        data = torch.load(cache_file, weights_only=True)
        return data['x'], data['y']
```

### Phase 3: Channel Mapper Integration (DONE)

**Location**: Already exists at `src/brain_go_brrr/infra/ml_models/channel_mapper.py`

```python
# We already have this! Just need to use it properly
class TUEVChannelMapper(nn.Module):
    """Learnable 23→20 channel mapping (matches EEGPT reference)."""
    
    def __init__(self, dropout: float = 0.8):
        super().__init__()
        # Conv2d to map 23 input channels to 20 EEGPT channels
        # Note: EEGPT uses Conv2d(23, 20, kernel_size=1) for channel mapping
        self.channel_conv = nn.Conv2d(
            in_channels=23,
            out_channels=20,
            kernel_size=1,
            bias=True  # EEGPT uses bias in their Conv2dWithConstraint
        )
        self.bn = nn.BatchNorm2d(20)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch, 23, 1, 1000) - add spatial dim for Conv2d
        if x.dim() == 3:
            x = x.unsqueeze(2)  # Add height dimension
        x = self.channel_conv(x)  # → (batch, 20, 1, 1000)
        x = self.bn(x)
        x = self.gelu(x)
        x = self.dropout(x)
        return x.squeeze(2)  # → (batch, 20, 1000)
```

### Phase 4: Training (Single thin wrapper) — DONE

- Trainer: `experiments/eegpt_linear_probe/train_tuev_events.py`.
- Modes:
  - Strict parity: `--use_parity` → EEGPT configured with `time_steps=1000`, `patch_stride=64` (no padding).
  - Compatibility fallback (default): pad 1000→1024 for standard EEGPT checkpoints.
- Loss: CrossEntropy(label_smoothing=0.1), unweighted.
- Schedule: warmup_epochs=5 + cosine anneal.
- Optimizer: AdamW with layer_decay=0.65 (block‑aware); lr=5e-4; weight_decay=0.05.
- Effective batch≈400 via gradient accumulation; epochs≈30.

Option A — Strict Parity (paper‑exact)

```
experiments/eegpt_linear_probe/train_tuev_events_parity.py
  - Load TUEVEventDataset (23×1000)
  - Apply TUEVChannelMapper (23→20)
  - Feed 20×1000 into TUEVClassifierHead (thin head in src)
- Loss: CrossEntropy(label_smoothing=0.1), unweighted
- Scheduler/optim: warmup_epochs=5, layer_decay=0.65, lr=5e-4, wd=0.05, epochs≈30; effective batch≈400 (distributed)
```

Option B — EEGPT‑Feature (fallback)

```
experiments/eegpt_linear_probe/train_tuev_events_probe.py
  - Load TUEVEventDataset (23×1000)
  - Apply TUEVChannelMapper (23→20)
  - Center‑crop 4.0 s (−2.0..+2.0) and resample to 256 Hz → 20×1024
  - EEGPTWrapper.extract_features() → (B, 4, 512) or flattened
  - TwoLayerProbe → logits (B, 6)
  - Loss: CrossEntropy(label_smoothing=0.1), unweighted; OneCycleLR with warmup
```

## Key Integration Points with Our Codebase

### What We REUSE:
1. **MNE preprocessing** - Just different parameters (200Hz, 0.1-75Hz filter)
2. **EEGPTWrapper** - Reused only in Option B (4s@256Hz path)
3. **TUEVChannelMapper** - Already built for 23→20 mapping
4. **TwoLayerProbe** - Our existing probe architecture
5. **Cache infrastructure** - Same .pt format with metadata

### What's NEW:
1. **TUEVEventExtractor** - Extracts 5s segments around events
2. **TUEVEventDataset** - Loads event segments (not sliding windows)
3. **train_tuev_events_parity.py / train_tuev_events_probe.py** - Thin training wrappers

### What We DON'T Need:
- Bipolar montage conversion
- Class weights or balancing
- Complex augmentation
- Sliding windows

## Validation Criteria

### Paper Parity Success Metrics (Acceptance):
```python
# From EEGPT Table 3:
target_metrics = {
    'balanced_accuracy': 0.6232 ± 0.0114,
    'weighted_f1': 0.8187 ± 0.0063,
    'cohen_kappa': 0.6351 ± 0.0134
}
```

### Cache Validation (must hold):
```python
# Each segment must be:
assert x.shape == (23, 1000)  # 23 channels, 5 seconds @ 200Hz
assert x.dtype == torch.float32
assert y in range(6)  # Valid class label
# Enforce META:
# sr==200, unit=='V', segment_type=='event', channels==23, samples==1000
```

### Training Validation (must hold):
- Warmup working (loss should be stable first 5 epochs)
- No class collapse (all 6 classes predicted)
- BAC improving steadily (not stuck at 16.67%)
- Layer decay active (per-parameter group scales present)

### Test Suite Alignment
- Unit: event extractor (filters, notch, sr=200, shape 23×1000)
- Unit: event dataset META/index and label ranges
- Integration: dataset → mapper → EEGPT features; shape + grad flow

## Monitoring and Rollback
- Checkpoints: pre-extraction cache snapshot; post-extraction event counts vs `.rec`; training checkpoints every N epochs
- Alerts: if epoch > 5 and max(BAC) < 0.30, raise for investigation (likely extraction/splits)
- Targets: BAC > 0.20 by epoch 2; > 0.40 by epoch 5; final 0.62 ± 0.02

## Guardrails
- SSOT: all logic in `src/`; experiments stay thin; no Lightning
- Safe load: prefer `torch.load(..., weights_only=True)` (PyTorch ≥ 2.4) or our safe loader; never unpickle arbitrary code
- No duplication: experiments import from `src/` only

## CRITICAL QUESTIONS ANSWERED DEFINITIVELY

### Q1: .rec/.lab Parser Status
**ANSWER: Parser EXISTS and WORKS** ✅
- Location: `src/brain_go_brrr/infra/data/tuev_dataset.py:_load_annotations()`
- Lines 316-323 correctly parse microseconds → seconds
- Action: Reuse parser, just change from sliding windows to event extraction

### Q2: Subject-Level Splits Implementation  
**ANSWER: 80/20 Random Split** ✅
- Reference: `make_TUEV.py` lines 224-226
- Method: `np.random.choice(train_sub, size=int(len(train_sub) * 0.2))`
- Action: Extract subject ID from filename, apply 80/20 split with seed=42

### Q3: Single GPU vs Distributed Training
**ANSWER: Single GPU with Smaller Batch** ✅
- Reference: 2 GPUs, batch_size=400 total (200 per GPU)
- Single GPU: Use batch_size=64–100; if memory-bound, implement manual gradient accumulation in PyTorch to match effective batch size. Do not use Lightning.

### Q4: Fallback Path
**ANSWER: Parity is the default; sliding-window is research-only**
- Use Strict Parity for replication (paper numbers).
- Keep existing sliding-window script as a non-parity fallback for research; do not use it to claim paper parity.

## Current Status & Next Steps

- Code: COMPLETE per plan (see files above).
- Data: Build event cache for train/eval using `TUEVEventDataset`.
- Verify metrics:
  - Fallback (pads to 1024): smoke test training for pipeline sanity.
  - Strict parity (`--use_parity`): run full training; acceptance if BAC ≈ 0.62 ± 0.02 on eval.

Verification commands:
- Parser for TUEV annotations:
  - `sed -n '77,86p' src/brain_go_brrr/infra/data/tuev_event_dataset.py`
- Parity stride support:
  - `rg -n "patch_stride|time_steps" src/brain_go_brrr/infra/ml_models/eegpt_architecture.py`
- Train (fallback):
  - `python experiments/eegpt_linear_probe/train_tuev_events.py --data_dir data/datasets/tuev --eegpt_checkpoint <ckpt> --epochs 1 --batch_size 32`
- Train (strict parity):
  - `python experiments/eegpt_linear_probe/train_tuev_events.py --data_dir data/datasets/tuev --eegpt_checkpoint <ckpt> --use_parity --epochs 1 --batch_size 32`

## Risk Mitigation

### If BAC Still Low:
1. Check we're using 200Hz (not 256Hz)
2. Verify warmup is working
3. Ensure all 23 channels present
4. Validate event annotations are correct

### If Cache Too Large:
1. Use HDF5 instead of individual .pt files
2. Compress with zarr
3. Store only unique segments (no overlap)

## Conclusion

This implementation:
- **Uses our existing infrastructure** (no parallel universe)
- **Matches EEGPT exactly** where it matters
- **Can achieve 62% BAC** with correct approach
- **Takes 2-3 days** to implement fully

The key insight: We don't need to rebuild everything. We just need a different data loader that extracts event segments instead of sliding windows. Everything else in our codebase is ready to go.
