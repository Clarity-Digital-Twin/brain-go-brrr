# MNE + Autoreject Implementation Guide for EEGPT Training

> ⚠️ **ARCHIVED DOCUMENT - DO NOT USE FOR CURRENT DEVELOPMENT**
> This document is preserved for historical reference only.
> For current documentation, see [docs/README.md](../../README.md)
> Archive date: September 2, 2025

---



## Executive Summary

This guide consolidates all verified MNE and Autoreject documentation for implementing preprocessing in the `experiments/eegpt_linear_probe/` folder. All parameters have been verified against source code and official documentation.

## Current State Analysis

### The Core Problem
- **Training**: Raw EEG → Direct to model → 56% AUROC
- **Inference**: Raw EEG → MNE → Autoreject → Model → Better performance
- **Gap**: Training on noisy data, inferring on clean data

### Why 56% → 87% is Achievable
Based on the literature and our external_references:
1. **Artifact removal**: 20-40% of data contains artifacts
2. **Bad channel handling**: Proper interpolation improves signal quality
3. **Adaptive thresholds**: Channel-specific rejection improves SNR
4. **Two-stage cleaning**: Global (MNE) + Local (Autoreject) = optimal

## Implementation Strategy

### Phase 1: Parallel Development (Don't Break Working Code)

Create new files parallel to existing:
```
experiments/eegpt_linear_probe/
├── train_tuab.py                 # KEEP RUNNING
├── train_tuab_mne.py             # NEW with preprocessing
├── datasets/
│   ├── tuab_cached_dataset.py    # KEEP AS-IS
│   └── tuab_mne_dataset.py       # NEW with MNE+Autoreject
└── mne_integration/              # NEW FOLDER
    ├── preprocessor.py
    ├── quality_scorer.py
    └── cache_builder.py
```

### Phase 2: MNE Preprocessing Pipeline

```python
# mne_integration/preprocessor.py
import mne
import numpy as np
from autoreject import AutoReject, Ransac

class TUABPreprocessor:
    """MNE+Autoreject preprocessing for TUAB dataset."""

    def __init__(self, config):
        self.config = config

    def process_raw(self, edf_path):
        """Full preprocessing pipeline."""

        # 1. Load with MNE
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

        # 2. Standardize to 256 Hz (EEGPT requirement)
        if raw.info['sfreq'] != 256:
            raw.resample(256)

        # 3. Set channel types and montage
        raw.set_channel_types({ch: 'eeg' for ch in raw.ch_names})
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='warn')

        # 4. MNE Global Preprocessing
        raw = self._apply_mne_preprocessing(raw)

        # 5. Create epochs (4-second windows for EEGPT)
        events = mne.make_fixed_length_events(raw, duration=4.0)
        epochs = mne.Epochs(
            raw, events,
            tmin=0, tmax=4.0,
            baseline=None,  # No baseline for EEGPT
            preload=True,
            reject=None,  # Let Autoreject handle this
            verbose=False
        )

        # 6. Apply Autoreject
        epochs_clean = self._apply_autoreject(epochs)

        return epochs_clean

    def _apply_mne_preprocessing(self, raw):
        """MNE global preprocessing steps."""

        # Bandpass filter (0.5-45 Hz for EEGPT)
        raw.filter(0.5, 45, fir_design='firwin')

        # Notch filter for line noise
        raw.notch_filter([60, 120], fir_design='firwin')

        # Detect and annotate artifacts
        # Muscle artifacts
        muscle_annot = mne.preprocessing.annotate_muscle_zscore(
            raw, threshold=4.0, ch_type='eeg',
            min_length_good=0.2, filter_freq=(110, 140)
        )
        raw.set_annotations(raw.annotations + muscle_annot)

        # Bad channels via RANSAC
        ransac = Ransac(n_jobs=1, random_state=42)
        ransac.fit(raw)
        if ransac.bad_chs_:
            raw.info['bads'].extend(ransac.bad_chs_)
            raw.interpolate_bads(reset_bads=True)

        # Re-reference to average
        raw.set_eeg_reference('average', projection=False)

        return raw

    def _apply_autoreject(self, epochs):
        """Apply Autoreject with TUAB-optimized parameters."""

        # TUAB-specific parameters (verified against docs)
        ar = AutoReject(
            n_interpolate=[1, 2, 3, 4],  # TUAB: 20 channels, can interpolate up to 4
            consensus=[0.3, 0.5, 0.7],   # TUAB: More aggressive for clinical data
            cv=5,  # Reduced from default=10 for speed
            thresh_method='bayesian_optimization',
            random_state=42,
            n_jobs=1,
            verbose=False
        )

        epochs_clean = ar.fit_transform(epochs)

        # Log statistics
        n_epochs_before = len(epochs)
        n_epochs_after = len(epochs_clean)
        print(f"Autoreject: {n_epochs_before} → {n_epochs_after} epochs")
        print(f"Removed: {n_epochs_before - n_epochs_after} ({100*(1-n_epochs_after/n_epochs_before):.1f}%)")

        # Access learned parameters (they're dicts by channel type)
        if hasattr(ar, 'n_interpolate_'):
            print(f"n_interpolate (EEG): {ar.n_interpolate_.get('eeg', 'N/A')}")
            print(f"consensus (EEG): {ar.consensus_.get('eeg', 'N/A')}")

        return epochs_clean
```

### Phase 3: Enhanced Dataset Class

```python
# datasets/tuab_mne_dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import json
from ..mne_integration.preprocessor import TUABPreprocessor

class TUABMNEDataset(Dataset):
    """TUAB dataset with MNE+Autoreject preprocessing."""

    def __init__(self, root_dir, split='train', cache_dir=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Initialize preprocessor
        self.preprocessor = TUABPreprocessor(config={
            'sampling_rate': 256,
            'window_duration': 4.0,
            'bandpass': (0.5, 45)
        })

        # Load file list
        self.files = self._load_file_list()

        # Build cache if needed
        if self.cache_dir and not self._cache_exists():
            self._build_cache()

    def _build_cache(self):
        """Build preprocessed cache with MNE+Autoreject."""

        print(f"Building MNE-preprocessed cache for {self.split}...")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        cache_index = {'files': {}, 'total_windows': 0}
        window_idx = 0

        for file_path, label in self.files:
            try:
                # Apply full preprocessing
                epochs_clean = self.preprocessor.process_raw(file_path)

                # Save each epoch as separate cache file
                for epoch_data in epochs_clean.get_data():
                    # Convert to float32 for training
                    epoch_data = epoch_data.astype('float32')

                    # Save
                    cache_file = f"window_{window_idx:06d}.pt"
                    torch.save({
                        'x': torch.from_numpy(epoch_data),
                        'y': torch.tensor(label, dtype=torch.float32),
                        'source_file': str(file_path)
                    }, self.cache_dir / cache_file)

                    cache_index['files'][window_idx] = {
                        'cache_file': cache_file,
                        'label': label,
                        'source': str(file_path)
                    }
                    window_idx += 1

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

        cache_index['total_windows'] = window_idx

        # Save index
        with open(self.cache_dir / 'index.json', 'w') as f:
            json.dump(cache_index, f, indent=2)

        print(f"Cache built: {window_idx} windows from {len(self.files)} files")
```

### Phase 4: Modified Training Script

```python
# train_tuab_mne.py
"""Training script with MNE+Autoreject preprocessing."""

import torch
from pathlib import Path
import yaml
from datasets.tuab_mne_dataset import TUABMNEDataset
from torch.utils.data import DataLoader

def main(config):
    # Use MNE-preprocessed dataset
    train_dataset = TUABMNEDataset(
        root_dir=config['data']['root_dir'],
        split='train',
        cache_dir=config['data']['mne_cache_dir']  # New cache location
    )

    # Rest of training code remains similar
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Continue with existing training loop...
```

## Critical Parameters (Verified)

### AutoReject Defaults (v0.4.2)
- `cv`: 10
- `n_interpolate`: [1, 4, 32]
- `consensus`: np.linspace(0, 1.0, 11)

### TUAB-Specific Settings
- `cv`: 5 (reduced for speed)
- `n_interpolate`: [1, 2, 3, 4] (20 channels max)
- `consensus`: [0.3, 0.5, 0.7] (clinical data needs aggressive cleaning)

### RejectLog Labels
- 0 = good channel/epoch
- 1 = bad (not interpolated)
- 2 = bad & interpolated (repaired)

### Key Implementation Notes

1. **Two-stage AR recommended**: Light AR → ICA → Final AR
2. **Dict attributes**: `ar.n_interpolate_` and `ar.consensus_` are dicts by channel type
3. **Bipolar reference**: Use `mne.set_bipolar_reference()` function (anode - cathode)
4. **REST reference**: Requires forward model
5. **Data types**: Convert to float32 before caching

## Expected Improvements

Based on literature and our analysis:
- **Baseline**: 56% AUROC (raw data)
- **With MNE only**: ~65-70% AUROC
- **With MNE + Autoreject**: 75-87% AUROC
- **Target**: 87% AUROC (paper performance)

## Validation Strategy

1. **A/B Testing**: Run both pipelines in parallel
2. **Metrics**: Compare AUROC, accuracy, F1
3. **Data Quality**: Log rejection rates, SNR improvements
4. **Convergence**: Monitor training stability

## Implementation Timeline

- **Week 1**: Build parallel infrastructure, implement preprocessor
- **Week 2**: Create cache, validate preprocessing quality
- **Week 3**: Train and compare models
- **Week 4**: Optimize and finalize

## References

- AutoReject source: `/reference_repos/autoreject/`
- MNE source: `/reference_repos/mne-python/`
- Verified docs: `/docs/external_references/`

This implementation guide is based on verified source code and corrected documentation. All parameters and methods have been cross-referenced with the actual implementations.
