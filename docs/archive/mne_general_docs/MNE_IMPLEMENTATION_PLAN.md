# MNE Integration Implementation Plan - Senior Engineer Approach

## Executive Summary

After auditing the current `experiments/eegpt_linear_probe/` implementation, I've identified critical gaps that explain the 56% accuracy plateau. The current pipeline loads raw cached data without any preprocessing, artifact rejection, or quality filtering. This document outlines a pragmatic, incremental approach to integrate MNE preprocessing while minimizing disruption to the existing training infrastructure.

## Current State Audit

### What's Working
✅ **Stable Training Loop** - `train_tuab.py` runs without NaN losses
✅ **Cached Dataset** - `.pt` files in `data/cache/tuab_4s_final/`
✅ **Crash Guards** - BCE loss shape/dtype fixes implemented
✅ **Memory Efficient** - Handles large dataset on WSL2

### Critical Gaps Identified
❌ **No Preprocessing** - Loading raw EEG directly from cache
❌ **No Quality Filtering** - Training on all data regardless of artifacts
❌ **No Artifact Rejection** - Muscle/movement artifacts in training data
❌ **No Feature Engineering** - Only using raw EEG, missing spectral features
❌ **No Data Augmentation** - Limited training diversity

### Why Current Accuracy is 56%
```python
# Current pipeline (oversimplified):
raw_eeg → cache → model → 56% AUROC

# What's missing:
raw_eeg → [PREPROCESSING] → [QUALITY CHECK] → [FEATURES] → cache → model → 87% AUROC
```

## Senior Engineer Recommendation: Parallel Development

**DO NOT** modify the existing working pipeline immediately. Instead:

1. **Keep current training running** - It's stable and provides baseline
2. **Build parallel MNE pipeline** - New scripts that don't break existing
3. **A/B test incrementally** - Compare improvements scientifically
4. **Merge when proven** - Only replace after demonstrating improvement

## Implementation Architecture

### Phase 1: Parallel Infrastructure (Week 1)

Create new parallel structure WITHOUT touching existing files:

```
experiments/eegpt_linear_probe/
├── train_tuab.py                 # KEEP RUNNING (current)
├── train_tuab_mne.py             # NEW - MNE enhanced version
├── datasets/
│   ├── tuab_cached_dataset.py    # KEEP AS-IS
│   └── tuab_mne_dataset.py       # NEW - with preprocessing
├── mne_integration/              # NEW FOLDER
│   ├── __init__.py
│   ├── preprocessor.py           # MNE preprocessing pipeline
│   ├── quality_scorer.py         # Artifact detection & scoring
│   ├── feature_extractor.py      # Spectral/connectivity features
│   └── cache_builder.py          # Build enhanced cache
└── configs/
    ├── tuab.yaml                  # KEEP AS-IS
    └── tuab_mne.yaml             # NEW - MNE config
```

### Phase 2: MNE Cache Builder (Week 1)

Build a NEW cache with MNE preprocessing, parallel to existing:

```python
# mne_integration/cache_builder.py
import mne
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import json

class MNECacheBuilder:
    """Build enhanced cache with MNE preprocessing."""

    def __init__(self, config):
        self.config = config
        self.preprocessor = MNEPreprocessor(config)
        self.quality_scorer = QualityScorer(config)

    def process_file(self, edf_path: Path) -> dict:
        """Process single EDF file with full MNE pipeline."""

        # 1. Load with MNE
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

        # 2. Quality scoring BEFORE preprocessing
        pre_quality = self.quality_scorer.score_raw(raw)

        # Skip if quality too low
        if pre_quality['overall_score'] < self.config['min_quality']:
            return None

        # 3. MNE preprocessing pipeline
        raw = self.preprocessor.process(raw)

        # 4. Extract windows (4s as per paper)
        windows = self.extract_windows(raw)

        # 5. Extract complementary features
        features = self.extract_features(windows, raw)

        # 6. Post-preprocessing quality
        post_quality = self.quality_scorer.score_windows(windows)

        return {
            'windows': windows,
            'features': features,
            'pre_quality': pre_quality,
            'post_quality': post_quality,
            'file_path': str(edf_path),
            'label': self.get_label(edf_path)
        }

    def build_cache(self, data_dir: Path, output_dir: Path):
        """Build complete MNE-enhanced cache."""

        output_dir.mkdir(parents=True, exist_ok=True)

        # Get all EDF files
        edf_files = list(data_dir.glob('**/*.edf'))

        index = {
            'files': {},
            'config': self.config,
            'stats': {
                'total_files': len(edf_files),
                'processed': 0,
                'rejected': 0,
                'total_windows': 0
            }
        }

        for edf_path in tqdm(edf_files, desc="Building MNE cache"):
            result = self.process_file(edf_path)

            if result is None:
                index['stats']['rejected'] += 1
                continue

            # Save each window
            file_key = edf_path.stem
            for i, window in enumerate(result['windows']):
                cache_file = output_dir / f"{file_key}_w{i:03d}.pt"

                # Save window with features
                torch.save({
                    'x': torch.from_numpy(window).float(),
                    'features': result['features'][i],
                    'y': result['label'],
                    'quality': result['post_quality'][i]
                }, cache_file)

                index['files'][str(cache_file)] = {
                    'original': result['file_path'],
                    'window_idx': i,
                    'label': result['label'],
                    'quality': result['post_quality'][i]
                }

            index['stats']['processed'] += 1
            index['stats']['total_windows'] += len(result['windows'])

        # Save index
        with open(output_dir / 'index.json', 'w') as f:
            json.dump(index, f, indent=2)

        print(f"Cache built: {index['stats']}")
```

### Phase 3: Enhanced Dataset (Week 1-2)

Create dataset that uses MNE cache with quality filtering:

```python
# datasets/tuab_mne_dataset.py
class TUABMNEDataset(Dataset):
    """TUAB dataset with MNE preprocessing and quality filtering."""

    def __init__(
        self,
        cache_dir: Path,
        split: str = 'train',
        min_quality: float = 0.5,
        use_features: bool = True,
        augment: bool = True
    ):
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.min_quality = min_quality
        self.use_features = use_features
        self.augment = augment and (split == 'train')

        # Load index
        with open(self.cache_dir / 'index.json', 'r') as f:
            self.index = json.load(f)

        # Filter by quality and split
        self.samples = []
        for file_path, info in self.index['files'].items():
            if info['quality'] >= min_quality:
                if self.should_include_in_split(info['original'], split):
                    self.samples.append({
                        'path': file_path,
                        'label': info['label'],
                        'quality': info['quality']
                    })

        # Quality-based sample weighting for training
        if split == 'train':
            self.sample_weights = [s['quality'] for s in self.samples]

        print(f"Loaded {len(self.samples)} high-quality samples for {split}")

    def __getitem__(self, idx):
        sample = self.samples[idx]
        data = torch.load(self.cache_dir / sample['path'])

        x = data['x']  # EEG data
        features = data['features'] if self.use_features else None
        y = data['y']

        # Augmentation for training
        if self.augment:
            x = self.augment_window(x)

        # Combine EEG with features if requested
        if features is not None:
            # Concatenate or return separately based on model needs
            return {'eeg': x, 'features': features}, y
        else:
            return x, y

    def augment_window(self, x):
        """Apply data augmentation."""
        # Random amplitude scaling
        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.8, 1.2)
            x = x * scale

        # Random temporal shift
        if np.random.rand() > 0.5:
            shift = np.random.randint(-50, 50)
            x = torch.roll(x, shift, dims=-1)

        # Add noise
        if np.random.rand() > 0.5:
            noise = torch.randn_like(x) * 0.05
            x = x + noise

        return x
```

### Phase 4: Modified Training Script (Week 2)

Create `train_tuab_mne.py` that uses enhanced dataset:

```python
# train_tuab_mne.py
"""TUAB training with MNE preprocessing - targeting 87% AUROC."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.eegpt_linear_probe.datasets.tuab_mne_dataset import TUABMNEDataset
from experiments.eegpt_linear_probe.train_tuab import LinearProbe, load_config
from src.brain_go_brrr.models.eegpt_wrapper import EEGPTWrapper

class EnhancedLinearProbe(nn.Module):
    """Linear probe that can use additional features."""

    def __init__(self, config):
        super().__init__()

        # EEGPT features path
        self.eegpt_probe = nn.Sequential(
            nn.LazyLinear(config["probe"]["hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(config["probe"]["dropout"]),
        )

        # Optional: Additional features path
        if config["probe"].get("use_additional_features", False):
            self.feature_probe = nn.Sequential(
                nn.LazyLinear(64),
                nn.ReLU(),
                nn.Dropout(0.1),
            )

            # Fusion layer
            self.fusion = nn.Sequential(
                nn.Linear(config["probe"]["hidden_dim"] + 64, 64),
                nn.ReLU(),
                nn.Dropout(config["probe"]["dropout"]),
                nn.Linear(64, 1)  # Binary output
            )
        else:
            self.feature_probe = None
            self.fusion = nn.Linear(config["probe"]["hidden_dim"], 1)

    def forward(self, eegpt_features, additional_features=None):
        batch_size = eegpt_features.shape[0]

        # Process EEGPT features
        x = eegpt_features.reshape(batch_size, -1)
        x = self.eegpt_probe(x)

        # Optionally fuse with additional features
        if self.feature_probe is not None and additional_features is not None:
            f = self.feature_probe(additional_features)
            x = torch.cat([x, f], dim=-1)

        return self.fusion(x).squeeze(-1)

def train_mne():
    """Main training function with MNE enhancements."""

    # Load config
    config = load_config("configs/tuab_mne.yaml")

    # Create MNE-enhanced datasets
    train_dataset = TUABMNEDataset(
        cache_dir=Path(config["data"]["mne_cache_dir"]),
        split="train",
        min_quality=config["data"]["min_quality"],
        use_features=config["model"]["probe"]["use_additional_features"],
        augment=config["data"]["augment"]
    )

    val_dataset = TUABMNEDataset(
        cache_dir=Path(config["data"]["mne_cache_dir"]),
        split="eval",
        min_quality=config["data"]["min_quality"],
        use_features=config["model"]["probe"]["use_additional_features"],
        augment=False  # No augmentation for validation
    )

    # Use weighted sampler for imbalanced classes + quality weighting
    from torch.utils.data import WeightedRandomSampler

    # Combine class weights with quality weights
    class_weights = compute_class_weights(train_dataset)
    quality_weights = train_dataset.sample_weights
    combined_weights = [c * q for c, q in zip(class_weights, quality_weights)]

    sampler = WeightedRandomSampler(
        weights=combined_weights,
        num_samples=len(train_dataset),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        sampler=sampler,  # Use weighted sampling
        num_workers=0,  # WSL2 compatibility
        pin_memory=False,
        collate_fn=collate_mne_batch  # Custom collate for features
    )

    # ... rest of training loop similar to train_tuab.py
    # but with enhanced model and metrics tracking
```

### Phase 5: Preprocessing Pipeline (Week 2)

Implement the actual MNE preprocessing:

```python
# mne_integration/preprocessor.py
class MNEPreprocessor:
    """Production-ready MNE preprocessing pipeline."""

    def __init__(self, config):
        self.config = config

    def process(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Apply full preprocessing pipeline."""

        # 1. Channel standardization
        raw = self.standardize_channels(raw)

        # 2. Artifact annotation
        raw = self.annotate_artifacts(raw)

        # 3. Filtering
        raw = self.apply_filters(raw)

        # 4. Re-referencing
        raw = self.apply_reference(raw)

        # 5. Bad channel interpolation
        raw = self.interpolate_bads(raw)

        return raw

    def standardize_channels(self, raw):
        """Ensure standard 10-20 channel names."""
        # Rename old TUAB names to modern
        mapping = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
        raw.rename_channels({old: new for old, new in mapping.items()
                            if old in raw.ch_names})

        # Set standard montage
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='warn')

        return raw

    def annotate_artifacts(self, raw):
        """Detect and annotate artifacts."""
        # Muscle artifacts
        muscle_annot = mne.preprocessing.annotate_muscle_zscore(
            raw,
            threshold=self.config['muscle_threshold'],
            ch_type='eeg',
            min_length_good=0.2,
            filter_freq=(110, 140)
        )
        raw.set_annotations(raw.annotations + muscle_annot)

        # Movement artifacts (custom implementation)
        movement_annot = self.detect_movement_artifacts(raw)
        raw.set_annotations(raw.annotations + movement_annot)

        # Flat segments
        flat_annot = mne.preprocessing.annotate_flat(
            raw,
            threshold=1e-15,
            min_duration=0.5
        )
        raw.set_annotations(raw.annotations + flat_annot)

        return raw

    def apply_filters(self, raw):
        """Apply optimal filtering."""
        # Notch filter for line noise
        raw.notch_filter(
            freqs=np.arange(60, 241, 60),
            picks='eeg',
            method='spectrum_fit',
            mt_bandwidth=2
        )

        # Bandpass filter
        raw.filter(
            l_freq=self.config['highpass'],
            h_freq=self.config['lowpass'],
            picks='eeg',
            method='fir',
            phase='zero-double'
        )

        return raw

    def apply_reference(self, raw):
        """Apply referencing scheme."""
        if self.config['reference'] == 'average':
            raw.set_eeg_reference('average', projection=False)
        elif self.config['reference'] == 'REST':
            sphere = mne.make_sphere_model('auto', 'auto', raw.info)
            raw.set_eeg_reference('REST', forward=sphere)

        return raw

    def interpolate_bads(self, raw):
        """Detect and interpolate bad channels."""
        # Use RANSAC from autoreject
        from autoreject import Ransac

        # Create temporary epochs for RANSAC
        events = mne.make_fixed_length_events(raw, duration=4.0)
        epochs = mne.Epochs(raw, events, tmin=0, tmax=4.0,
                           baseline=None, preload=True, verbose=False)

        # Detect bad channels
        ransac = Ransac(n_jobs=1, random_state=42)
        ransac.fit(epochs)

        # Mark bad channels
        raw.info['bads'] = ransac.bad_chs_

        # Interpolate if not too many
        if len(raw.info['bads']) < len(raw.ch_names) * 0.3:
            raw.interpolate_bads(reset_bads=True)

        return raw
```

### Phase 6: Quality Scoring (Week 2)

Implement quality metrics:

```python
# mne_integration/quality_scorer.py
class QualityScorer:
    """Score EEG quality for filtering training data."""

    def score_raw(self, raw: mne.io.Raw) -> dict:
        """Compute quality metrics for raw recording."""

        metrics = {}

        # 1. SNR
        psd, freqs = mne.time_frequency.psd_welch(
            raw, fmin=0.5, fmax=50, n_fft=2048
        )

        signal_power = np.mean(psd[:, (freqs >= 8) & (freqs <= 12)])
        noise_power = np.mean(psd[:, (freqs >= 30) & (freqs <= 50)])
        metrics['snr'] = 10 * np.log10(signal_power / noise_power)

        # 2. Artifact percentage
        total_time = raw.times[-1]
        bad_time = sum([a['duration'] for a in raw.annotations
                       if a['description'].startswith('BAD')])
        metrics['artifact_pct'] = (bad_time / total_time) * 100

        # 3. Channel quality
        n_good = len([ch for ch in raw.ch_names
                     if ch not in raw.info.get('bads', [])])
        metrics['good_channel_pct'] = (n_good / len(raw.ch_names)) * 100

        # 4. Overall score (weighted combination)
        metrics['overall_score'] = (
            0.4 * np.clip(metrics['snr'] / 20, 0, 1) +
            0.3 * (1 - metrics['artifact_pct'] / 100) +
            0.3 * (metrics['good_channel_pct'] / 100)
        )

        return metrics

    def score_windows(self, windows: list) -> list:
        """Score individual windows."""
        scores = []

        for window in windows:
            score = self.score_window(window)
            scores.append(score)

        return scores

    def score_window(self, window: np.ndarray) -> float:
        """Score single window quality."""
        # Check for flat channels
        flat_channels = np.sum(np.std(window, axis=1) < 1e-6)

        # Check for extreme values
        has_extreme = np.any(np.abs(window) > 200e-6)

        # Check variance
        variance_ok = 1e-7 < np.var(window) < 1e-3

        # Combine into score
        score = 1.0
        score -= 0.1 * flat_channels
        score -= 0.3 if has_extreme else 0
        score -= 0.2 if not variance_ok else 0

        return max(0, score)
```

## Configuration Strategy

### New Config File: `tuab_mne.yaml`

```yaml
# configs/tuab_mne.yaml
experiment:
  name: tuab_mne_enhanced
  description: "MNE-enhanced training targeting 0.87 AUROC"
  seed: 42

data:
  # Original data paths (for cache building)
  root_dir: ${BGB_DATA_ROOT}/datasets/external/tuab

  # MNE cache (separate from original)
  mne_cache_dir: ${BGB_DATA_ROOT}/cache/tuab_mne_4s

  # Quality filtering
  min_quality: 0.5  # Only use windows with quality > 0.5

  # Training settings
  batch_size: 32  # Smaller due to additional features
  augment: true  # Enable augmentation

  # Preprocessing parameters
  preprocessing:
    # Filtering
    highpass: 0.5
    lowpass: 50
    notch_freqs: [60, 120, 180]

    # Artifact detection
    muscle_threshold: 4.0
    movement_threshold: 100e-6

    # Reference
    reference: average  # or REST

    # Channels
    required_channels: ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4']
    max_bad_channels: 4

model:
  probe:
    use_additional_features: true  # Use spectral features
    feature_dim: 128  # Dimension of additional features
    fusion_type: concatenate  # or 'attention'

training:
  # Start conservative, increase if stable
  max_epochs: 20

  # Quality-weighted loss
  use_quality_weighting: true

  # More aggressive early stopping
  early_stopping:
    patience: 5
    min_delta: 0.01
    monitor: val_auroc
```

## Validation & Testing Strategy

### 1. A/B Testing Framework

```python
# scripts/compare_pipelines.py
"""Compare original vs MNE-enhanced pipelines."""

def compare_models():
    # Load both models
    original_model = load_checkpoint('output/tuab_original/best.pt')
    mne_model = load_checkpoint('output/tuab_mne/best.pt')

    # Test on same validation set
    val_dataset = load_validation_set()

    # Compute metrics
    original_metrics = evaluate(original_model, val_dataset)
    mne_metrics = evaluate(mne_model, val_dataset)

    # Statistical significance test
    from scipy.stats import mcnemar
    p_value = mcnemar_test(original_metrics, mne_metrics)

    print(f"Original AUROC: {original_metrics['auroc']:.3f}")
    print(f"MNE AUROC: {mne_metrics['auroc']:.3f}")
    print(f"Improvement: {mne_metrics['auroc'] - original_metrics['auroc']:.3f}")
    print(f"Significant? p={p_value:.4f}")
```

### 2. Ablation Studies

Test each component's contribution:

```bash
# Test configurations
configs=(
    "tuab.yaml"                    # Baseline
    "tuab_mne_filter_only.yaml"    # Just filtering
    "tuab_mne_artifact_only.yaml"  # Just artifact rejection
    "tuab_mne_quality_only.yaml"   # Just quality filtering
    "tuab_mne_features_only.yaml"  # Just additional features
    "tuab_mne_full.yaml"           # Everything
)

for config in "${configs[@]}"; do
    python train_tuab_mne.py --config "configs/$config"
done
```

## Risk Mitigation

### Memory Management

```python
# For WSL2 memory constraints
class MemoryEfficientMNEDataset(TUABMNEDataset):
    """Load and preprocess on-the-fly if cache too large."""

    def __getitem__(self, idx):
        # Option 1: Load from cache if exists
        if self.use_cache:
            return super().__getitem__(idx)

        # Option 2: Process on-the-fly
        else:
            edf_path = self.get_edf_path(idx)
            raw = mne.io.read_raw_edf(edf_path, preload=False)

            # Load only needed segment
            start, end = self.get_segment_times(idx)
            raw.crop(start, end).load_data()

            # Quick preprocessing (lighter than full)
            raw = self.quick_preprocess(raw)

            return self.extract_window(raw), self.labels[idx]
```

### Debugging Tools

```python
# mne_integration/debug.py
def diagnose_preprocessing():
    """Visualize preprocessing effects."""

    # Load sample
    raw_original = load_sample_raw()
    raw_processed = preprocess(raw_original.copy())

    # Plot comparison
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    # Original
    raw_original.plot(ax=axes[0], show=False)
    axes[0].set_title('Original')

    # Processed
    raw_processed.plot(ax=axes[1], show=False)
    axes[1].set_title('After MNE Preprocessing')

    # Difference
    diff = raw_processed.get_data() - raw_original.get_data()
    axes[2].plot(diff.T[:1000])  # First 1000 samples
    axes[2].set_title('Difference')

    plt.savefig('preprocessing_diagnostic.png')
```

## Timeline & Milestones

### Week 1: Foundation
- [ ] Set up parallel folder structure
- [ ] Implement cache builder
- [ ] Build small test cache (100 files)
- [ ] Verify cache integrity

### Week 2: Core Implementation
- [ ] Implement MNEPreprocessor
- [ ] Implement QualityScorer
- [ ] Create TUABMNEDataset
- [ ] Test on small subset

### Week 3: Training Integration
- [ ] Implement train_tuab_mne.py
- [ ] Add feature extraction
- [ ] Run first training
- [ ] Compare with baseline

### Week 4: Optimization
- [ ] Tune hyperparameters
- [ ] Run ablation studies
- [ ] Full-scale training
- [ ] Performance validation

### Week 5: Production
- [ ] Document findings
- [ ] Merge if improved
- [ ] Update configs
- [ ] Deploy

## Expected Outcomes

### Metrics Improvements
- **AUROC**: 56% → 75-87%
- **Training Time**: Similar or slightly longer
- **Memory Usage**: +20% due to features
- **Stability**: Better (quality filtering)

### Deliverables
1. Enhanced cache with quality scores
2. MNE preprocessing pipeline
3. Improved model checkpoint
4. Ablation study results
5. A/B test statistics

## Monitoring & Debugging

### Key Metrics to Track

```python
# Add to train_tuab_mne.py
metrics_to_log = {
    # Quality metrics
    'avg_window_quality': np.mean([s['quality'] for s in batch]),
    'rejected_samples_pct': rejected / total * 100,

    # Preprocessing metrics
    'avg_snr_improvement': snr_after - snr_before,
    'interpolated_channels': len(interpolated),

    # Training metrics
    'loss': loss.item(),
    'auroc': auroc_score,
    'learning_rate': scheduler.get_last_lr()[0],

    # System metrics
    'gpu_memory_mb': torch.cuda.max_memory_allocated() / 1e6,
    'batch_processing_time': time.time() - start
}
```

## Decision Points

### When to Switch to MNE Pipeline

Switch from original to MNE pipeline when:

1. **AUROC improvement > 10%** (e.g., 56% → 66%+)
2. **Training is stable** (no NaN losses for 3 epochs)
3. **Validation on held-out data confirms improvement**
4. **Memory usage is acceptable** (< 16GB)
5. **Processing time is reasonable** (< 2x original)

### When to Keep Original

Stay with original if:

1. **No significant improvement** (< 5% AUROC gain)
2. **Training becomes unstable**
3. **Memory issues on WSL2**
4. **Processing time > 3x original**

## Conclusion

This implementation plan provides a **safe, incremental path** to integrate MNE preprocessing without disrupting your current training. The parallel development approach allows you to:

1. **Keep baseline running** - No risk to current progress
2. **Test scientifically** - A/B testing with statistical validation
3. **Debug easily** - Separate pipelines for comparison
4. **Roll back if needed** - Original pipeline untouched

The modular design means you can also **cherry-pick improvements** - if only artifact rejection helps but features don't, you can adopt just that component.

**Senior Engineer Wisdom**: Don't rebuild what's working. Enhance it incrementally, measure everything, and only adopt what proves its worth.

---

*Ready for implementation review*
*Estimated effort: 3-4 weeks with current training running in parallel*
*Risk level: Low (parallel development)*
*Expected outcome: 15-30% AUROC improvement*
