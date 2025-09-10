# TUEV Reference Implementation (EEGPT Authors)

**Source**: `reference_repos/EEGPT/downstream_tueg/`  
**Paper Target**: 62.32% ± 1.14% balanced accuracy  
**Created**: September 10, 2025
**Last Updated**: September 10, 2025 - Added critical missing components

## 🔴 CRITICAL COMPONENTS FOR ANOTHER AI TO VERIFY

**ATTENTION**: If you're analyzing the reference repo independently, focus on these critical aspects that caused our training to fail:

1. **LinearWithConstraint in classifier head** (file: `Modules/models/EEGPT_mcae_finetune_change_tuev.py`)
   - Look for `LinearWithConstraint(30720, num_classes, max_norm=1)`
   - Without this, training collapses to majority classes

2. **Conv2dWithConstraint in channel mapper** (same file)
   - Full pipeline: Conv2dWithConstraint → BatchNorm → GELU → DepthwiseConv → BatchNorm → Dropout
   - Not just a simple Conv2d!

3. **Per-iteration LR scheduling** (file: `utils.py`)
   - Function: `cosine_scheduler` - creates iteration-level schedule
   - Applied in training loop before each optimizer.step()

4. **Loss function source** (file: `run_class_finetuning_EEGPT_change_tuev.py`)
   - Uses `timm.loss.LabelSmoothingCrossEntropy`
   - Not a custom implementation

5. **Layer-wise LR decay** (file: `optim_factory.py`)
   - Function: `get_parameter_groups` with `LayerDecayValueAssigner`
   - Deeper layers get exponentially lower learning rates

## ⚠️ DATA PATH WARNING FOR IMPLEMENTATION

**CRITICAL**: When implementing, the data structure MUST be:
```
data/datasets/tuev/       # NO /raw subdirectory!
├── edf/
│   ├── train/           # Official split directory
│   └── eval/            # Official split directory
└── cache/
    └── tuev_event_segments/
        ├── train/*.pkl  # MUST have actual pickle files
        └── eval/*.pkl   # MUST have actual pickle files
```

Common implementation error: Using `data/datasets/tuev/raw` which doesn't exist!

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
Note: the reference code also applies an 'offset' when slicing per recording; omitted here for brevity. The window length is enforced to 1000 samples.

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

### Channel Mapping Architecture (CRITICAL DETAILS)
```python
# The channel mapper is NOT just a simple Conv2d!
self.chan_conv = nn.Sequential(
    Conv2dWithConstraint(in_channels=23, out_channels=20, kernel_size=1, max_norm=1),
    nn.BatchNorm2d(20),
    nn.GELU(),
    nn.Conv2d(20, 20, kernel_size=(1,55), groups=20, padding='same'),  # Depthwise temporal conv
    nn.BatchNorm2d(20),
    nn.Dropout(0.8),  # Heavy dropout IN the mapper itself!
)

# Usage in forward():
if self.use_chan_conv:
    x = x[:,:,None]  # Add spatial dimension: (B,23,1000) → (B,23,1,1000)
    x = self.chan_conv(x)[:,:,0]  # Apply mapping and remove spatial: → (B,20,1000)
```

Classifier head behavior (from authors' code):
```python
# CRITICAL: Uses LinearWithConstraint, not standard Linear!
class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super().__init__(*args, **kwargs)
    
    def forward(self, x):
        if self.doWeightNorm:
            # Renormalize weights EVERY forward pass
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super().forward(x)

# Actual classifier:
self.head = nn.Sequential(
    nn.Dropout(0.8),
    LinearWithConstraint(30720, num_classes, max_norm=1),  # NOT nn.Linear!
)

# forward():
x = target_encoder(...)
x = x.flatten(1)                  # Flatten ALL temporal summary tokens
x = self.head(x)                  # Dropout + LinearWithConstraint
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
opt_betas = None  # Use default (0.9, 0.999)
clip_grad = None  # No gradient clipping

# CRITICAL: Uses COSINE scheduler for BOTH LR and weight decay!
lr_schedule = cosine_scheduler(
    base_value=5e-4,
    final_value=1e-6,  # min_lr
    epochs=30,
    niter_per_ep=num_training_steps_per_epoch,
    warmup_epochs=5
)

wd_schedule = cosine_scheduler(
    base_value=0.05,
    final_value=0.05,  # Same as base (constant in practice)
    epochs=30,
    niter_per_ep=num_training_steps_per_epoch,
    warmup_epochs=0  # No warmup for WD
)

# Layer-wise LR decay implementation
layer_decay = 0.65
num_layers = model.get_num_layers()  # ~12 for EEGPT
lr_scales = [layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)]
# Deeper layers get smaller LR multipliers
```

## Critical Implementation Details (EXPANDED Sep 10, 2025)

### 1. LinearWithConstraint - THE MOST CRITICAL COMPONENT
```python
class LinearWithConstraint(nn.Linear):
    """CRITICAL: Without this, training collapses to majority classes!"""
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super().__init__(*args, **kwargs)
    
    def forward(self, x):
        if self.doWeightNorm:
            # Renormalize weights EVERY forward pass - prevents explosion
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super().forward(x)
```

### 2. Per-Iteration Scheduling (NOT Per-Epoch!)
```python
# From utils.py - cosine_scheduler function
def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0):
    """Creates per-iteration schedule, not per-epoch!"""
    warmup_iters = warmup_epochs * niter_per_ep
    warmup_schedule = np.linspace(0, base_value, warmup_iters) if warmup_epochs > 0 else []
    
    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = [final_value + 0.5 * (base_value - final_value) * 
                (1 + math.cos(math.pi * i / len(iters))) for i in iters]
    
    return np.concatenate((warmup_schedule, schedule))

# Applied in training loop BEFORE optimizer.step():
for i, batch in enumerate(dataloader):
    it = epoch * steps_per_epoch + i
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr_schedule[it]  # Update EVERY iteration
```

### 3. Loss Function - Use timm's Implementation
```python
from timm.loss import LabelSmoothingCrossEntropy
criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
# NOT a custom implementation - use timm for exact parity
```

### 4. Label Mapping
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

### 4. Classifier Tokens
- Consumes ALL temporal summary tokens (flattened). No mean pooling on TUEV.

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

## Compatibility Notes (Our Environment)

- When rebuilding our cache with MNE, you may see: `NOTE: pick_channels() is a legacy function. New code should use inst.pick(...)`.
- This warning pertains to our preprocessing code paths that still call `raw.pick_channels(...)`; it is not part of the authors’ reference code and does not affect alignment with their results.
