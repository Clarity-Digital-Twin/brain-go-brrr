# TUEV Reference Implementation - COMPLETE DOCUMENTATION
**Source Repository**: EEGPT Reference Implementation  
**Paper Target**: 62.32% ± 1.14% balanced accuracy (Table 3, page 7)  
**Created**: September 10, 2025  
**Last Updated**: September 10, 2025 - EXHAUSTIVE ANALYSIS

## 🔴 CRITICAL IMPLEMENTATION CHECKLIST

### MUST-HAVE Components (WILL FAIL WITHOUT THESE):

1. **Data Scaling by 100** 
   - Location: `downstream_tueg/engine_for_finetuning_EEGPT.py:65,174`
   - `samples = samples.float().to(device) / 100`
   
2. **LinearWithConstraint with max_norm=1** 
   - Location: `downstream_tueg/Modules/models/EEGPT_mcae_finetune_change_tuev.py:579-590`
   - Note: Constraint layers exist in two places; the model uses the in-file definitions WITH `@autocast(True)`. The duplicates in `downstream_tueg/Modules/Network/utils.py` do not use autocast.
   - Renormalizes weights EVERY forward pass
   
3. **Conv2dWithConstraint with max_norm=1**
   - Location: `downstream_tueg/Modules/models/EEGPT_mcae_finetune_change_tuev.py:593-608`
   - Note: As above, the model’s in-file definition uses `@autocast(True)`; the version in `Modules/Network/utils.py` does not.
   
4. **Per-Iteration LR Scheduling**
   - Location: `downstream_tueg/utils.py:cosine_scheduler`
   - Location: `downstream_tueg/engine_for_finetuning_EEGPT.py:58-63`
   - Updates EVERY iteration, not epoch
   
5. **Label Smoothing CrossEntropy**
   - Location: `downstream_tueg/run_class_finetuning_EEGPT_change_tuev.py:477-480`
   - From `timm.loss.LabelSmoothingCrossEntropy`
   - Smoothing factor: 0.1

## Data Pipeline - COMPLETE FLOW

### 1. Raw Data Download
- **Paper Reference**: Section C.2.6, page 19
- **Download URL**: https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml
- **Version/Paths in repo**:
  - Preprocessing script targets `v2.0.0` at `downstream_tueg/dataset_maker/make_TUEV.py:184`.
  - Training/loader expects processed data under `v2.0.1/edf/processed/` (see `prepare_TUEV_dataset` call in `run_class_finetuning_EEGPT_change_tuev.py`). Ensure your processed folders are available at the expected training path, or align both to the same version.
- **Structure**: 
  ```
  datasets/downstream/tuh_eeg_events/v2.0.1/edf/
  ├── train/    # Training data with .edf and .rec files
  └── eval/     # Evaluation data with .edf and .rec files
  ```

### 2. Data Preprocessing (`downstream_tueg/dataset_maker/make_TUEV.py`)

#### Signal Processing Pipeline (lines 116-138):
```python
def readEDF(fileName):
    # Step 1: Load with MNE
    Rawdata = mne.io.read_raw_edf(fileName, preload=True)
    
    # Step 2: Drop channels to keep only 23 standard
    # Drop list defined at line 11-12
    drop_channels = ['PHOTIC-REF', 'IBI', 'BURSTS', ...]
    Rawdata.drop_channels(useless_chs)
    
    # Step 3: Reorder to standard (line 125)
    chOrder_standard = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', ...]  # Line 14-15
    Rawdata.reorder_channels(chOrder_standard)
    
    # Step 4: Filter and resample
    Rawdata.filter(l_freq=0.1, h_freq=75.0)  # Line 129
    Rawdata.notch_filter(50.0)               # Line 130
    Rawdata.resample(200, n_jobs=5)          # Line 131
    
    # Step 5: Get data in MICROVOLTS
    signals = Rawdata.get_data(units='uV')   # Line 134 - CRITICAL!
```

#### Event Extraction (lines 18-40):
```python
def BuildEvents(signals, times, EventData):
    fs = 200.0  # Sampling rate after resampling
    features = np.zeros([numEvents, numChan, int(fs) * 5])  # 5 seconds
    
    # Extract fixed 5s window around each event
    for i in range(numEvents):
        start = np.where(times >= EventData[i, 1])[0][0]
        end = np.where(times >= EventData[i, 2])[0][0]
        # Window: 2s before start to 2s after end = 5s total
        features[i, :] = signals[:, offset + start - 2*int(fs) : offset + end + 2*int(fs)]
        labels[i, :] = int(EventData[i, 3])  # Event type label
```

#### Data Splits (lines 216-235):
```python
seed = 4523  # Fixed seed for reproducibility
np.random.seed(seed)

# 80/20 subject-based train/val split
val_sub = np.random.choice(train_sub, size=int(len(train_sub) * 0.2), replace=False)
train_sub = list(set(train_sub) - set(val_sub))
```

### 3. Data Loading (`downstream_tueg/utils.py`)

#### TUEVLoader Class (lines 720-741):
```python
class TUEVLoader(torch.utils.data.Dataset):
    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["signal"]              # Shape: (23, 1000) in μV
        Y = int(sample["label"][0] - 1)   # CRITICAL: 1-6 → 0-5
        X = torch.FloatTensor(X)
        return X, Y
```

#### prepare_TUEV_dataset Function (lines 740-763):
```python
def prepare_TUEV_dataset(root):
    seed = 4523  # Same seed as preprocessing
    np.random.seed(seed)
    
    train_files = os.listdir(os.path.join(root, "processed_train"))
    val_files = os.listdir(os.path.join(root, "processed_eval"))
    test_files = os.listdir(os.path.join(root, "processed_test"))
    
    train_dataset = TUEVLoader(os.path.join(root, "processed_train"), train_files)
    test_dataset = TUEVLoader(os.path.join(root, "processed_test"), test_files)
    val_dataset = TUEVLoader(os.path.join(root, "processed_eval"), val_files)
```

## Model Architecture - COMPLETE DETAILS

### Channel Configuration (`downstream_tueg/run_class_finetuning_EEGPT_change_tuev.py`)

#### Input Channels (23 channels, lines 208-211):
```python
ch_names = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 
            'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 
            'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 
            'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 
            'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 
            'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
```

#### Target Channels (20 channels, lines 201-206):
```python
use_channels_names = ['FP1','FPZ', 'FP2',
                      'F7', 'F3', 'FZ', 'F4', 'F8',
                      'T7', 'C3', 'CZ', 'C4', 'T8',
                      'P7', 'P3', 'PZ', 'P4', 'P8',
                      'O1', 'O2']
```

### Channel Mapping Layer (`downstream_tueg/Modules/models/EEGPT_mcae_finetune_change_tuev.py`)

#### Channel Convolution (lines 697-709):
```python
self.chan_conv = torch.nn.Sequential(
    Conv2dWithConstraint(in_channels, img_size[0], 1),  # 23→20, max_norm=1
    nn.BatchNorm2d(img_size[0]),
    nn.GELU(),
    nn.Conv2d(img_size[0], img_size[0], kernel_size=(1,55), 
              groups=img_size[0], padding='same'),  # Depthwise temporal
    nn.BatchNorm2d(img_size[0]),
    nn.Dropout(0.8),  # HEAVY dropout
)
```

### Classifier Head (lines 764-772):
```python
self.head = nn.Sequential(
    nn.Dropout(0.8),  # Another heavy dropout 
    LinearWithConstraint(30720, num_classes, max_norm=1),
)
# 30720 = 512 (embed_dim) × 4 (summary tokens) × 15 (temporal patches)
# Details: img_size[1]=1000, patch_size=stride=64 ⇒ N=15; embed_dim=512; embed_num=4.
```

### Weight Constraint Implementation (two definitions)

#### In-model definition with autocast (used)
`downstream_tueg/Modules/models/EEGPT_mcae_finetune_change_tuev.py:579-590` (shown above).

#### Duplicate utility definition (no autocast)
`downstream_tueg/Modules/Network/utils.py:1-48`
```python
class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):  # no autocast here
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)
```

## Training Pipeline - COMPLETE CONFIGURATION

### Training Script (`downstream_tueg/finetune_TUEV_EEGPT.sh`)

```bash
#!/usr/bin/env bash
MASTER_PORT=$((12000 + $RANDOM % 20000))  # Line 5

CUDA_VISIBLE_DEVICES=4,5 OMP_NUM_THREADS=1 python -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_port ${MASTER_PORT} \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="localhost" \
    run_class_finetuning_EEGPT_change_tuev.py \
    --output_dir ./checkpoints_TUEV/finetune_tuev_eegpt/ \
    --log_dir ./log/finetune_tuev_eegpt \
    --model EEGPT \
    --finetune ../checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt \
    --weight_decay 0.05 \
    --batch_size 400 \           # Total batch across GPUs
    --lr 5e-4 \
    --update_freq 1 \
    --warmup_epochs 5 \
    --epochs 30 \
    --layer_decay 0.65 \
    --drop_path 0.2 \
    --dist_eval \
    --save_ckpt_freq 5 \
    --disable_rel_pos_bias \
    --abs_pos_emb \
    --dataset TUEV \
    --enable_deepspeed \
    --seed 0
```

Note: Flags like `--drop_path`, `--abs_pos_emb`, and `--disable_rel_pos_bias` are part of the generic CLI but are not consumed by the custom `EEGPTClassifier` path in this repo (the transformer is instantiated directly with fixed drop rates in the model file).

### Critical Training Loop Details (`downstream_tueg/engine_for_finetuning_EEGPT.py`)

#### Data Processing (lines 65-70):
```python
# CRITICAL: Scale by 100!
samples = samples.float().to(device, non_blocking=True) / 100
# Optional reshape for downstream compatibility (model flattens it back):
samples = rearrange(samples, 'B N (A T) -> B N A T', T=200)  # B,23,5,200 → model flattens to B,23,1000

if is_binary:  # False for TUEV (6 classes)
    targets = targets.float().unsqueeze(-1)
```

#### Mixed Precision Training (lines 77-79):
```python
with torch.cuda.amp.autocast():
    loss, output = train_class_batch(model, samples, targets, criterion, input_chans)
```

#### Per-Iteration LR Update (lines 58-63):
```python
if lr_schedule_values is not None:
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
        if wd_schedule_values is not None and param_group["weight_decay"] > 0:
            param_group["weight_decay"] = wd_schedule_values[it]
```

### Optimizer Configuration (`downstream_tueg/optim_factory.py`)

#### Layer-wise Learning Rate Decay (lines 37-45):
```python
class LayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values  # [0.65^12, 0.65^11, ..., 0.65^0]
    
    def get_scale(self, layer_id):
        return self.values[layer_id]
    
    def get_layer_id(self, var_name):
        return get_num_layer_for_vit(var_name, len(self.values))
```

#### Parameter Groups (lines 48-95):
```python
def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), 
                         get_num_layer=None, get_layer_scale=None):
    # Groups parameters by layer and applies decay
    # Biases and LayerNorm params have no weight decay
    # Each layer gets its own learning rate scale
```

### Loss Function (`downstream_tueg/run_class_finetuning_EEGPT_change_tuev.py`)

#### Lines 476-480:
```python
if args.nb_classes == 1:  # Binary (TUAB)
    criterion = torch.nn.BCEWithLogitsLoss()
elif args.smoothing > 0.:  # Multi-class with smoothing (TUEV)
    criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)  # 0.1
else:
    criterion = torch.nn.CrossEntropyLoss()
```

### Learning Rate Schedule (`downstream_tueg/utils.py`)

#### Cosine Scheduler (exact implementation needed for reproduction):
```python
def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, 
                     warmup_epochs=0, warmup_steps=-1):
    # Creates per-iteration schedule
    # Warmup for 5 epochs
    # Cosine decay from 5e-4 to 1e-6
```

## Evaluation Pipeline

### Metrics Implementation (`downstream_tueg/utils.py`)

#### Lines 866-886:
```python
def get_metrics(output, target, metrics, is_binary, threshold=0.5):
    if is_binary:
        results = binary_metrics_fn(target, output, metrics=metrics, threshold=threshold)
    else:
        results = multiclass_metrics_fn(target, output, metrics=metrics)
    return results
```

#### Metrics Used (line 248):
```python
metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
```

### Evaluation Loop (`downstream_tueg/engine_for_finetuning_EEGPT.py`)

#### Lines 155-203:
```python
@torch.no_grad()
def evaluate(data_loader, model, device, header='Test:', 
             ch_names=None, metrics=['acc'], is_binary=True):
    model.eval()
    
    for step, batch in enumerate(data_loader):
        EEG = batch[0]
        target = batch[-1]
        EEG = EEG.float().to(device) / 100  # Same scaling!
        EEG = rearrange(EEG, 'B N (A T) -> B N A T', T=200)
        
        with torch.cuda.amp.autocast():
            output = model(EEG)
            loss = criterion(output, target)
```

## Model Checkpoint Loading (`downstream_tueg/run_class_finetuning_EEGPT_change_tuev.py`)

### Lines 365-399:
```python
if args.finetune:
    checkpoint = torch.load(args.finetune, map_location='cpu')
    print("Load ckpt from %s" % args.finetune)
    
    # CRITICAL: Load from 'state_dict' key
    checkpoint_model = checkpoint['state_dict']  # Line 398
    utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
```

## Weight Initialization (`downstream_tueg/Modules/models/EEGPT_mcae_finetune_change_tuev.py`)

### Truncated Normal Initialization (lines 39-72):
```python
def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # Used for all linear layers
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
```

### Model Initialization (lines 486-499):
```python
def _init_weights(self, m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=self.init_std)  # init_std=0.02
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        trunc_normal_(m.weight, std=self.init_std)
```

## Paper Cross-References

### TUEV Dataset Description
- **Paper Table 1** (page 5): 288 subjects, 6 classes
- **Paper Section C.2.6** (page 19): Event types and preprocessing
- **Paper Table 3** (page 7): Performance metrics

### TUEV Performance Claims
- **Balanced Accuracy**: 62.32% ± 1.14%
- **Weighted F1**: 81.87% ± 0.63%
- **Cohen's Kappa**: 63.51% ± 1.34%
- **9.5% improvement** over BIOT baseline

### Implementation Details from Paper
- **Section C.2.6** (page 19): "convolution kernel size for TUEV was (1, 55)"
- **Table 13** (page 20): Model architecture for TUEV
- Paper appendix notes a batch size of ~500 for TUEV; this repo’s launch script uses 400.

## Critical Missing Details NOT in Original Document

1. **@autocast(True) decorators** on constraint layers (present in the model file; not in `Modules/Network/utils.py`)
2. **init_std=0.02** for weight initialization
3. **Exact 30720 dimension** calculation
4. **Data divided by 100** during training
5. **Reshape to patches** with T=200 (applied in engine; flattened in model)
6. **Layer-wise LR decay** implementation
7. **PyHealth metrics** library usage
8. **Subject-based splits** with seed 4523
9. **No bipolar montage** (line 151 commented out)
10. **Mixed precision** training with amp

## File Structure Summary
```
downstream_tueg/
├── dataset_maker/
│   └── make_TUEV.py                    # Preprocessing pipeline
├── Modules/
│   ├── models/
│   │   └── EEGPT_mcae_finetune_change_tuev.py  # Model architecture
│   └── Network/
│       └── utils.py                    # Constraint layers
├── run_class_finetuning_EEGPT_change_tuev.py   # Main training script
├── engine_for_finetuning_EEGPT.py      # Training/eval loops
├── utils.py                             # Data loading, metrics, schedulers
├── optim_factory.py                     # Optimizer configuration
└── finetune_TUEV_EEGPT.sh             # Launch script

datasets/downstream/tuh_eeg_events/v2.0.1/edf/
├── processed/
│   ├── processed_train/                # Pickle files
│   ├── processed_eval/                 # Pickle files
│   └── processed_test/                 # Pickle files
```

## Reproduction Checklist

- [ ] Data in μV, divided by 100 during training
- [ ] LinearWithConstraint with max_norm=1 and @autocast
- [ ] Conv2dWithConstraint with max_norm=1 and @autocast
- [ ] Label smoothing = 0.1
- [ ] Layer decay = 0.65
- [ ] Dropout = 0.8 in both channel mapper and classifier
- [ ] Per-iteration LR scheduling
- [ ] Reshape data to patches with T=200
- [ ] Labels subtracted by 1 (1-6 → 0-5)
- [ ] Mixed precision training with torch.cuda.amp
- [ ] Load from checkpoint['state_dict']
- [ ] 23→20 channel mapping with learned conv
- [ ] Seed = 4523 for data splits, seed = 0 for training

---
**END OF EXHAUSTIVE DOCUMENTATION**
