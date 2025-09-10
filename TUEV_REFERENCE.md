# TUEV Reference Implementation (EEGPT Authors)

**Source**: `reference_repos/EEGPT/downstream_tueg/`  
**Paper Target**: 62.32% ± 1.14% balanced accuracy  
**Created**: September 10, 2025

## Table of Contents
1. [Data Pipeline](#data-pipeline)
2. [Model Architecture](#model-architecture)
3. [Training Configuration](#training-configuration)
4. [Critical Implementation Details](#critical-implementation-details)
5. [File Structure](#file-structure)

## Data Pipeline

### Raw Data Processing (`dataset_maker/make_TUEV.py`)

```python
# Signal processing pipeline
def readEDF(fileName):
    # 1. Load EDF with MNE
    Rawdata = mne.io.read_raw_edf(fileName, preload=True)
    
    # 2. Drop irrelevant channels (keep 23 standard)
    Rawdata.drop_channels([ch for ch in Rawdata.ch_names if ch in drop_channels])
    
    # 3. Filter and resample
    Rawdata.filter(0.1, 75.0)    # Bandpass 0.1-75 Hz
    Rawdata.notch_filter(50.0)   # Notch at 50 Hz
    Rawdata.resample(200)        # Resample to 200 Hz
    
    # 4. Get data in microvolts
    signals = Rawdata.get_data(units='uV')
    return signals
```

### Event Extraction
```python
def BuildEvents(signals, times, EventData):
    fs = 200.0
    features = np.zeros([numEvents, numChan, int(fs) * 5])  # 5 seconds at 200Hz
    
    for i in range(numEvents):
        start = np.where(times >= EventData[i, 1])[0][0]
        end = np.where(times >= EventData[i, 2])[0][0]
        
        # Extract a fixed 5s window (1000 samples) around each annotated event
        features[i, :] = signals[:, start - 2*int(fs) : end + 2*int(fs)]
        labels[i, :] = int(EventData[i, 3])  # Label from .rec file
    
    return features, labels
```

**Key Points**:
- Extract fixed 5-second segments (exactly 1000 samples) around annotated events
- 23 channels × 1000 samples (5s @ 200Hz)
- Signals saved in **microvolts (μV)** - MNE scales to μV before writing pickles via `get_data(units='uV')`
- **NO bipolar montage** - line 151: `#signals = convert_signals(signals, Rawdata)` is commented out

### Channel Configuration

**23 Input Channels (Standard Order)**:
```python
chOrder_standard = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 
                    'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 
                    'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 
                    'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 
                    'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 
                    'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
```

**20 Target Channels for Model**:
```python
use_channels_names = ['FP1','FPZ','FP2','F7','F3','FZ','F4','F8',
                      'T7','C3','CZ','C4','T8','P7','P3','PZ','P4','P8','O1','O2']
```

## Model Architecture

### Channel Mapping (`run_class_finetuning_EEGPT_change_tuev.py`)
```python
model = EEGPTClassifier(
    num_classes=6,                           # 6 event types
    in_channels=23,                          # Input channels
    img_size=[20, 1000],                     # Target size after mapping
    use_channels_names=use_channels_names,   # 20 channel names
    use_chan_conv=True,                      # Enable 23→20 Conv2d mapping
    use_mean_pooling=args.use_mean_pooling
)
```

### Data Loading (`utils.py`)
```python
class TUEVLoader(torch.utils.data.Dataset):
    def __getitem__(self, index):
        sample = pickle.load(open(self.files[index], "rb"))
        X = sample["signal"]                 # Shape: (23, 1000)
        Y = int(sample["label"][0] - 1)      # CRITICAL: Subtract 1 (1-6 → 0-5)
        X = torch.FloatTensor(X)
        return X, Y
```

## Training Configuration

### Hyperparameters (`finetune_TUEV_EEGPT.sh`)
```bash
--batch_size 400        # Distributed across GPUs
--lr 5e-4               # Learning rate
--weight_decay 0.05     # Weight decay
--warmup_epochs 5       # Warmup epochs
--epochs 30             # Total epochs
--layer_decay 0.65      # Layer-wise LR decay
--drop_path 0.2         # DropPath rate
--smoothing 0.1         # Label smoothing (default in args)
```

### Loss Function
```python
# From run_class_finetuning_EEGPT_change_tuev.py
criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)  # smoothing=0.1
```

### Optimizer Configuration
```python
# AdamW with layer-wise learning rate decay
opt = 'adamw'
opt_eps = 1e-8
opt_betas = None  # Use default
clip_grad = None  # No gradient clipping
```

## Critical Implementation Details

### 1. Label Mapping
**CRITICAL**: Labels are subtracted by 1 in the dataloader:
```python
Y = int(sample["label"][0] - 1)  # Original 1-6 → Model expects 0-5
```

### 2. Data Splits
```python
# From make_TUEV.py
seed = 4523
np.random.seed(seed)

# 80/20 subject-based train/val split
val_sub = np.random.choice(train_sub, size=int(len(train_sub) * 0.2), replace=False)
train_sub = list(set(train_sub) - set(val_sub))
```

### 3. NO Class Balancing
- No WeightedRandomSampler
- No class weights in loss function
- Standard cross-entropy with label smoothing

### 4. Distributed Training
```bash
GPUS_PER_NODE=2
--batch_size 400  # Total batch size across GPUs
# Some scripts use --enable_deepspeed; otherwise standard PyTorch DDP
```

### 5. Evaluation Metrics
```python
metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
```

### 6. Data Format
- Pickled dictionaries with keys: 'signal', 'offending_channel', 'label'
- Signal shape: (23, 1000) in microvolts (μV)
- Files named: `{subject_id}_{session}_{segment}-{idx}.pkl`

## File Structure

```
downstream_tueg/
├── dataset_maker/
│   └── make_TUEV.py           # Creates event segments
├── run_class_finetuning_EEGPT_change_tuev.py  # Main training script
├── utils.py                    # TUEVLoader class
├── finetune_TUEV_EEGPT.sh    # Launch script with hyperparameters
└── Modules/models/
    └── EEGPT_mcae_finetune_change_tuev.py  # Model with channel conv

Data structure expected:
../datasets/downstream/tuh_eeg_events/v2.0.1/edf/
├── processed_train/  # Pickled segments
├── processed_eval/   # Pickled segments
└── processed_test/   # Pickled segments
```

## Key Observations

### What They Do:
1. **Event-only extraction**: 5s segments around annotated events
2. **23→20 channel mapping**: Learned Conv2d layer
3. **Label smoothing**: 0.1 smoothing factor
4. **No class balancing**: Despite severe imbalance
5. **Microvolts**: Data kept in μV, not converted to Volts
6. **Subject-based splits**: 80/20 train/val by subject

### What They DON'T Do:
1. **No sliding windows**: Only event-centered segments
2. **No bipolar montage**: Commented out, uses referential
3. **No normalization mentioned**: Raw μV values
4. **No class weights**: Standard loss despite imbalance
5. **No data augmentation**: Direct loading of segments

## Performance Claims
- Paper reports: 62.32% ± 1.14% BAC
- 6-class classification
- Evaluated on held-out test set

---

**Note**: This document represents the EXACT implementation from the EEGPT reference repository, not our interpretation or modifications.