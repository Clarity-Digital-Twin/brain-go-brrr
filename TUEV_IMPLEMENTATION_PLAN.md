# 🎯 TUEV Implementation Plan - Complete Guide

## 📊 Current Status Assessment

### ✅ What We Have Working
1. **TUAB Dataset & Training**
   - 2993 EDF files downloaded and cached
   - Memory-mapped dataset implementation working
   - Best model: 0.7897 AUROC (91% of paper target 0.869)
   - Saved at: `experiments/eegpt_linear_probe/output/tuab_4s_paper_target_BULLETPROOF_20250809_073159/best_model.pt`

2. **Infrastructure Ready**
   - EEGPT wrapper functioning
   - Training pipeline battle-tested (10 bugs fixed)
   - Clean architecture with ports/adapters
   - YASA sleep staging integrated

### ✅ What We Have Now
1. **TUEV Dataset**: FULLY DOWNLOADED at `data/datasets/external/tuh_eeg/TUEV/v2.0.1/`
   - 370 subjects (290 train, 80 eval) - **More than paper's 288** (v2.0.1 is newer)
   - 518 EDF files (359 train, 159 eval)
   - 11,396 label files with 6-class annotations
   - All 6 event classes confirmed: SPSW, GPED, PLED, EYEM, ARTF, BCKG
   - **IMPORTANT**: Paper used 288 subjects, we have 370 - results may differ slightly
2. **Dataset Structure Verified**: Ready for loader implementation
3. **Annotation Format**: .lab files with microsecond timestamps + .rec files
4. **Split Strategy CRITICAL**: Must use BIOT strategy (existing train/eval split), NOT LOSO!

## 🚨 CRITICAL UPDATE: Deep Audit Found Major Discrepancies!

**SEE `TUEV_CRITICAL_ARCHITECTURE.md` for full details!**

Key findings from Table 13 (page 20):
- Input is 23 × 1000 (NOT 23 × 1280!)
- Reduces to 20 channels (NOT expands to 58!)
- Dropout is 0.5 (NOT 0.25!)
- No learning rate schedule (just constant 5e-4)
- Output shape is 15 × 4 × 512 (NOT 31 × 4 × 512)

## 📚 TUEV Specifications (From EEGPT Paper - CORRECTED)

### Dataset Characteristics
- **Size**: 112,491 5-second samples
- **Classes**: 6 event types
  1. SPSW - Spike and Sharp Wave (epileptiform)
  2. GPED - Generalized Periodic Epileptiform Discharges
  3. PLED - Periodic Lateralized Epileptiform Discharges  
  4. EYEM - Eye Movement (artifact)
  5. ARTF - Other Artifacts
  6. BCKG - Background (normal)
- **Channels**: 23 channels @ 256 Hz
- **Window**: 5 seconds (vs 4 seconds for TUAB)

### Training Parameters (Paper)
- **Batch Size**: 500 (vs 100 for TUAB)
- **Learning Rate**: 5e-4
- **Kernel Size**: (1, 55) for temporal conv (vs (1, 15) for TUAB)
- **Metrics**: Balanced Accuracy, Weighted F1, Cohen's Kappa

### Paper Performance
- **Balanced Accuracy**: 0.6232 ± 0.0114
- **Weighted F1**: 0.8187 ± 0.0063  
- **Cohen's Kappa**: 0.6351 ± 0.0134

### Critical Split Strategy (from paper)
**QUOTE**: "For the data splitting of TUAB and TUEV, we strictly follow the same strategy as BIOT to compare all methods fairly."
- Use predefined train/eval split (290 train, 80 eval subjects)
- NOT LOSO (that's for other datasets)
- NOT random split
- Must be comparable to BIOT/EEGPT results

## 🚀 Implementation Phases

### Phase 1: Dataset Acquisition (Day 1)

#### 1.1 Download TUEV Dataset
```bash
# Create download script
cat > scripts/download_tuev.sh << 'EOF'
#!/bin/bash
# Temple University EEG Events Dataset Download

DATA_DIR="data/datasets/external/tuh_eeg_events"
mkdir -p $DATA_DIR

# TUEV v2.0.1 download (requires credentials)
echo "Downloading TUEV v2.0.1..."
echo "Username: nedc-tuh-eeg"
echo "You'll need the password from Temple University"

rsync -auxvL \
  nedc-tuh-eeg@www.isip.piconepress.com:data/tuh_eeg/tuh_eeg_events/v2.0.1/ \
  $DATA_DIR/v2.0.1/

echo "Download complete!"
echo "[Paper] Claims: 112,491 5-second segments"
echo "[Reality] Table 13 shows: 1000 samples (3.9 seconds)"
EOF

chmod +x scripts/download_tuev.sh
```

#### 1.2 Verify Dataset Structure
```python
# scripts/verify_tuev_dataset.py
import json
from pathlib import Path
import pandas as pd

def verify_tuev():
    base = Path("data/datasets/external/tuh_eeg_events/v1.0.1")
    
    # Check for annotation files
    train_csv = base / "train" / "annotations.csv"
    eval_csv = base / "eval" / "annotations.csv"
    
    if train_csv.exists():
        df = pd.read_csv(train_csv)
        print(f"Train samples: {len(df)}")
        print(f"Classes: {df['label'].value_counts()}")
    
    # Expected structure:
    # - 6 classes: SPSW, GPED, PLED, EYEM, ARTF, BCKG
    # - TCP montage (23 channels)
    # - 5-second windows
```

### Phase 2: Dataset Implementation (Day 2)

#### 2.1 Create TUEV Dataset Class
```python
# experiments/eegpt_linear_probe/tuev_dataset.py

import torch
import numpy as np
from pathlib import Path
import json
import mne

class TUEVDataset(torch.utils.data.Dataset):
    """TUEV 6-class event detection dataset."""
    
    CLASS_MAPPING = {
        'SPSW': 0,  # Spike and Sharp Wave
        'GPED': 1,  # Generalized Periodic Epileptiform Discharges
        'PLED': 2,  # Periodic Lateralized Epileptiform Discharges
        'EYEM': 3,  # Eye Movement
        'ARTF': 4,  # Artifacts
        'BCKG': 5   # Background
    }
    
    def __init__(
        self,
        data_dir: Path,
        split: str = 'train',
        window_size: float = 5.0,  # 5 seconds for TUEV
        sampling_rate: int = 256,
        cache_dir: Path = None
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        self.n_samples = int(window_size * sampling_rate)  # 1280 samples
        
        # Load annotations
        self.annotations = self._load_annotations()
        
        # Create cache if needed
        if cache_dir:
            self.cache_dir = Path(cache_dir) / "tuev_5s_cache"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._build_cache()
    
    def _load_annotations(self):
        """Load TUEV annotations CSV."""
        csv_path = self.data_dir / self.split / "annotations.csv"
        # Expected columns: file_path, start_time, end_time, label, channels
        return pd.read_csv(csv_path)
    
    def __getitem__(self, idx):
        """Get preprocessed 5-second window."""
        ann = self.annotations.iloc[idx]
        
        # Load EDF segment
        edf_path = self.data_dir / ann['file_path']
        raw = mne.io.read_raw_edf(edf_path, preload=False)
        
        # Extract 5-second window
        start = ann['start_time']
        raw.crop(tmin=start, tmax=start + self.window_size)
        data = raw.get_data()
        
        # Ensure 23 channels (TCP montage)
        if data.shape[0] != 23:
            data = self._remap_channels(data, raw.ch_names)
        
        # Normalize
        data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-6)
        
        # Convert label
        label = self.CLASS_MAPPING[ann['label']]
        
        return torch.tensor(data, dtype=torch.float32), label
    
    def __len__(self):
        return len(self.annotations)
```

#### 2.2 Create Memory-Mapped Version
```python
# experiments/eegpt_linear_probe/tuev_mmap_dataset.py

class TUEVMemoryMappedDataset(torch.utils.data.Dataset):
    """Memory-mapped TUEV for efficient loading."""
    
    def __init__(self, cache_dir: Path, split: str = 'train'):
        self.cache_dir = Path(cache_dir)
        self.split = split
        
        # Load index
        index_file = self.cache_dir / f"{split}_index.json"
        with open(index_file) as f:
            self.index = json.load(f)
        
        self.samples = self.index['samples']
        self.n_classes = 6
    
    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        
        # Load from disk (no RAM usage)
        cache_file = self.cache_dir / sample_info['cache_file']
        data = torch.load(cache_file, map_location='cpu', weights_only=True)
        
        return data['x'], data['y']
    
    def __len__(self):
        return len(self.samples)
```

### Phase 3: Training Pipeline Adaptation (Day 3)

#### 3.1 Create 6-Class Linear Probe
```python
# experiments/eegpt_linear_probe/train_tuev_aligned.py

import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score, cohen_kappa_score

class TUEVLinearProbe(nn.Module):
    """6-class probe for TUEV event classification."""
    
    def __init__(self, config):
        super().__init__()
        
        # Channel adapter (23 channels → model channels)
        self.channel_adapter = nn.Conv1d(
            23,  # TUEV has 23 channels
            config['probe']['channel_adapter_out'],
            kernel_size=1
        )
        
        # Temporal convolution (5s windows need larger kernel)
        self.temporal_conv = nn.Conv1d(
            config['probe']['channel_adapter_out'],
            config['probe']['channel_adapter_out'],
            kernel_size=55,  # Paper specifies (1, 55) for TUEV
            groups=config['probe']['channel_adapter_out'],  # Depthwise
            padding='same'
        )
        
        # 6-class classifier
        self.probe = nn.Sequential(
            nn.Linear(config['probe']['input_dim'], config['probe']['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(config['probe']['dropout']),
            nn.Linear(config['probe']['hidden_dim'], 6)  # 6 classes!
        )
    
    def forward(self, features):
        # features from EEGPT: (batch, n_summary_tokens, embed_dim)
        x = features.mean(dim=1)  # Average pool
        return self.probe(x)  # (batch, 6)
```

#### 3.2 Adapt Training Loop
```python
def train_epoch_tuev(model, probe, train_loader, optimizer, scheduler, device, config):
    """Training loop for 6-class TUEV."""
    model.eval()  # Backbone frozen
    probe.train()
    
    criterion = nn.CrossEntropyLoss(
        weight=compute_class_weights(train_loader)  # Handle imbalance
    )
    
    all_preds = []
    all_labels = []
    losses = []
    
    for batch_idx, (data, labels) in enumerate(tqdm(train_loader)):
        data = data.to(device)
        labels = labels.to(device)
        
        # Extract EEGPT features
        with torch.no_grad():
            features = model.extract_features(data)
        
        # Get predictions
        logits = probe(features)
        loss = criterion(logits, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()  # Per-batch stepping
        
        # Track metrics
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        losses.append(loss.item())
    
    # Compute metrics
    metrics = {
        'loss': np.mean(losses),
        'balanced_acc': balanced_accuracy_score(all_labels, all_preds),
        'weighted_f1': f1_score(all_labels, all_preds, average='weighted'),
        'cohen_kappa': cohen_kappa_score(all_labels, all_preds)
    }
    
    return metrics
```

### Phase 4: Configuration & Launch (Day 4)

#### 4.1 Create TUEV Config
```yaml
# experiments/eegpt_linear_probe/configs/tuev_5s_paper_aligned.yaml

data:
  dataset: "tuev"
  window_size: 5.0  # 5 seconds for TUEV
  sampling_rate: 256
  n_channels: 23
  n_classes: 6
  batch_size: 500  # Paper specifies 500 for TUEV
  num_workers: 4
  pin_memory: true

model:
  eegpt_checkpoint: "${BGB_DATA_ROOT}/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"
  freeze_backbone: true
  
probe:
  use_channel_adapter: true
  channel_adapter_in: 23  # TUEV channels
  channel_adapter_out: 58  # EEGPT expects 58
  temporal_kernel: 55  # Paper specifies for TUEV
  input_dim: 768  # EEGPT embedding dim
  hidden_dim: 256
  dropout: 0.2
  n_classes: 6

training:
  n_epochs: 100
  learning_rate: 5.0e-4  # Paper specifies 5e-4 for TUEV
  scheduler:
    max_lr: 5.0e-4
    final_lr: 5.0e-7
    cycle_momentum: false  # Critical for AdamW!
  
metrics:
  - balanced_accuracy
  - weighted_f1
  - cohen_kappa
  - confusion_matrix
```

#### 4.2 Create Launch Script
```bash
# experiments/eegpt_linear_probe/LAUNCH_TUEV.sh

#!/bin/bash
set -e

echo "=== TUEV 6-Class Event Detection Training ==="
echo "Target Performance (from paper):"
echo "  - Balanced Accuracy: 0.6232"
echo "  - Weighted F1: 0.8187"
echo "  - Cohen's Kappa: 0.6351"
echo ""

# Environment setup
export CUDA_VISIBLE_DEVICES=0
export BGB_DATA_ROOT=/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data

# Test scheduler first
echo "Testing scheduler behavior..."
python test_scheduler_dry_run.py --config configs/tuev_5s_paper_aligned.yaml

# Launch training
LOG_FILE="logs/tuev_training_$(date +%Y%m%d_%H%M%S).log"
echo "Starting training, logging to: $LOG_FILE"

python train_tuev_aligned.py \
    --config configs/tuev_5s_paper_aligned.yaml \
    --device cuda \
    2>&1 | tee $LOG_FILE

echo "Training complete!"
```

### Phase 5: Hierarchical Pipeline Integration (Day 5)

#### 5.1 Create Combined Pipeline
```python
# src/brain_go_brrr/services/hierarchical_eeg_analyzer.py

class HierarchicalEEGAnalyzer:
    """Two-stage: TUAB (abnormal?) → TUEV (what type?)"""
    
    def __init__(self, tuab_model_path, tuev_model_path, device='cuda'):
        # Load EEGPT backbone
        self.eegpt = EEGPTWrapper(device=device)
        
        # Load TUAB binary classifier
        self.tuab_probe = self._load_probe(tuab_model_path, n_classes=2)
        
        # Load TUEV event classifier
        self.tuev_probe = self._load_probe(tuev_model_path, n_classes=6)
        
        self.event_names = ['SPSW', 'GPED', 'PLED', 'EYEM', 'ARTF', 'BCKG']
    
    def analyze(self, eeg_data, threshold=0.5):
        """Full hierarchical analysis."""
        
        # Extract features once
        with torch.no_grad():
            features = self.eegpt.extract_features(eeg_data)
        
        # Stage 1: Abnormal detection
        abnormal_logits = self.tuab_probe(features)
        abnormal_prob = torch.softmax(abnormal_logits, dim=-1)[..., 1]
        
        results = {
            'is_abnormal': abnormal_prob > threshold,
            'abnormal_confidence': abnormal_prob.item()
        }
        
        # Stage 2: Event classification (only if abnormal)
        if results['is_abnormal']:
            event_logits = self.tuev_probe(features)
            event_probs = torch.softmax(event_logits, dim=-1)
            
            event_idx = event_probs.argmax().item()
            results.update({
                'event_type': self.event_names[event_idx],
                'event_confidence': event_probs[0, event_idx].item(),
                'all_event_probs': {
                    name: prob.item() 
                    for name, prob in zip(self.event_names, event_probs[0])
                }
            })
            
            # Generate clinical description
            results['clinical_description'] = self._generate_description(results)
        else:
            results.update({
                'event_type': 'BCKG',
                'clinical_description': 'Normal background activity'
            })
        
        return results
    
    def _generate_description(self, results):
        """Generate human-readable clinical description."""
        
        descriptions = {
            'SPSW': 'Epileptiform activity with spike and sharp wave discharges',
            'GPED': 'Generalized periodic epileptiform discharges detected',
            'PLED': 'Periodic lateralized epileptiform discharges detected',
            'EYEM': 'Eye movement artifacts present',
            'ARTF': 'Non-physiological artifacts detected',
            'BCKG': 'Background activity'
        }
        
        event = results['event_type']
        conf = results['event_confidence']
        
        if event in ['SPSW', 'GPED', 'PLED']:
            severity = 'High clinical significance'
        elif event in ['EYEM', 'ARTF']:
            severity = 'Technical issue - repeat recording recommended'
        else:
            severity = 'Normal variant'
        
        return f"{descriptions[event]} (confidence: {conf:.1%}). {severity}."
```

## 📋 Validation & Testing

### Test Data Preparation
```python
# scripts/prepare_tuev_test_set.py

def prepare_test_set():
    """Prepare held-out test set for final evaluation."""
    
    # Use official eval split
    eval_dir = Path("data/datasets/external/tuh_eeg_events/v1.0.1/eval")
    
    # Create balanced test set
    # Ensure each class has sufficient samples
    # Save as separate cache for fast loading
```

### Performance Metrics
```python
# scripts/evaluate_tuev.py

def evaluate_tuev_model(model_path, test_loader):
    """Comprehensive evaluation matching paper metrics."""
    
    metrics = {
        'balanced_accuracy': [],
        'weighted_f1': [],
        'cohen_kappa': [],
        'per_class_f1': {},
        'confusion_matrix': None
    }
    
    # Run 3 times with different seeds (paper protocol)
    for seed in [42, 123, 456]:
        torch.manual_seed(seed)
        # ... evaluation code ...
    
    # Report mean ± std as in paper
    print(f"Balanced Accuracy: {np.mean(metrics['balanced_accuracy']):.4f} ± {np.std(metrics['balanced_accuracy']):.4f}")
    print(f"Target from paper: 0.6232 ± 0.0114")
```

## 🚨 Critical Implementation Notes

### 1. **Dataset Differences from TUAB**
- **Window Size**: 5 seconds (not 4)
- **Channels**: 23 (not 20)
- **Kernel Size**: (1, 55) (not (1, 15))
- **Batch Size**: 500 (not 100)

### 2. **Class Imbalance**
TUEV likely has imbalanced classes. Need:
- Weighted loss function
- Stratified sampling
- Per-class metrics

### 3. **Memory Considerations**
With 112,491 samples × 5 seconds × 23 channels:
- Use memory-mapped dataset
- Cache preprocessed windows
- Batch size 500 might need adjustment

### 4. **Debugging Tips**
```python
# Quick sanity checks
assert window_size == 5.0, "TUEV uses 5-second windows"
assert n_channels == 23, "TUEV uses TCP montage (23 channels)"
assert n_classes == 6, "TUEV has 6 event types"
assert kernel_size[1] == 55, "Paper specifies (1, 55) kernel"
```

## 📊 Expected Timeline

- **Day 1**: Download dataset, verify structure
- **Day 2**: Implement dataset classes, build cache
- **Day 3**: Adapt training pipeline for 6-class
- **Day 4**: Configure and launch training
- **Day 5**: Integrate hierarchical pipeline
- **Day 6-7**: Debug, optimize, evaluate

## 🎯 Success Criteria

Match or exceed paper performance:
- ✅ Balanced Accuracy ≥ 0.62
- ✅ Weighted F1 ≥ 0.81
- ✅ Cohen's Kappa ≥ 0.63

Plus demonstrate:
- ✅ Hierarchical pipeline working (TUAB → TUEV)
- ✅ Clinical descriptions generated
- ✅ Real-time inference (<100ms)

## 📚 References

1. EEGPT Paper (primary reference)
2. Temple University EEG Corpus documentation
3. BIOT paper (for data split strategy)
4. Our TUAB implementation (for patterns to follow)

---

**Next Immediate Action**: Run `scripts/download_tuev.sh` to get the dataset!