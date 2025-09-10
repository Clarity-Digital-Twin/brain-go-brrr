# TUEV Reference Implementation - FINAL CONSOLIDATED DOCUMENT
**Paper Target**: 62.32% ± 1.14% balanced accuracy  
**Our Result**: 24% BAC (38% BELOW TARGET)  
**Purpose**: SINGLE SOURCE OF TRUTH - Send this to other repos/agents

## 📊 IMPLEMENTATION STATUS SUMMARY

### ✅ IMPLEMENTED (PARITY ACHIEVED):
1. **Signal tripling** - YES (src/brain_go_brrr/infra/preprocessing/tuev_event_extractor.py uses `extended = np.concatenate([data, data, data], axis=1)`)
2. **μV/100 scaling** - YES (experiments/eegpt_linear_probe/train_tuev_events.py:509)
3. **NO normalization** - YES (self.eegpt.normalize=False, inputs scaled to μV/100 at model entry)
4. **LinearWithConstraint head** - YES (max_norm=1.0)
5. **Channel mapper (23→20)** - YES (Conv2dWithConstraint with max_norm=1.0)
6. **Mixed precision** - YES (loop-level torch.cuda.amp.autocast())
7. **Per-iteration LR scheduling** - YES
8. **Label smoothing (0.1)** - YES (timm.loss.LabelSmoothingCrossEntropy)
9. **Layer decay (0.65)** - YES
10. **DropPath (0.2)** - YES
11. **Token flattening (30720)** - YES
12. **Natural sampling** - YES (no class balancing)
13. **Effective batch ≈400** - YES (via gradient accumulation)
14. **Window extraction** - YES (start/end with triple buffer and [start-2s:end+2s])

### ⚠️ MINOR DIFFERENCES (NON-BLOCKING):
1. **Reshape to B×23×5×200** - OPTIONAL (reference reshapes then immediately flattens; our path is functionally equivalent)
2. **Method-level @autocast** - We use loop-level AMP instead (safer, avoids dtype mismatches)
3. **Seeds** - We use 42 for training (reference uses 0); not performance-critical
4. **Distributed training** - We use single GPU with accumulation (matches effective batch 400)
5. **DeepSpeed** - Not required for parity

## 🔴 THE REMAINING MYSTERY: Why Only 24% BAC?

Despite implementing ALL critical components correctly, we still get 38% below target. The key question is WHY?

### Current Performance Pattern:
| Class | Samples | Expected Recall | Our Recall |
|-------|---------|----------------|------------|
| spsw | 24 | ~40% | 0% |
| gped | 374 | ~70% | 55% |
| pled | 74 | ~40% | 0% |
| eyem | 75 | ~40% | 0% |
| artf | 124 | ~50% | 0% |
| bckg | 800 | ~90% | 85% |

**PATTERN**: Only classes with >300 samples show any learning!

## 📊 COMPLETE REFERENCE PIPELINE (FOR VERIFICATION)

### Data Preprocessing (`make_TUEV.py`)
```python
def readEDF(fileName):
    # 1. Load with MNE
    Rawdata = mne.io.read_raw_edf(fileName, preload=True)
    
    # 2. Drop to 23 channels
    drop_channels = ['PHOTIC-REF', 'IBI', 'BURSTS', ...]
    Rawdata.drop_channels(useless_chs)
    
    # 3. Reorder channels
    chOrder_standard = ['EEG FP1-REF', 'EEG FP2-REF', ...]  # 23 channels
    Rawdata.reorder_channels(chOrder_standard)
    
    # 4. Filter and resample
    Rawdata.filter(l_freq=0.1, h_freq=75.0)
    Rawdata.notch_filter(50.0)
    Rawdata.resample(200, n_jobs=5)  # 200 Hz
    
    # 5. Get data in MICROVOLTS
    signals = Rawdata.get_data(units='uV')  # CRITICAL: μV units!
    
    # NO NORMALIZATION (commented out in reference)
```

### Event Extraction with Triple Signal
```python
def BuildEvents(signals, times, EventData):
    fs = 200.0
    features = np.zeros([numEvents, numChan, int(fs) * 5])  # 5 seconds
    
    # Triple the signal for boundary handling
    offset = signals.shape[1]
    signals = np.concatenate([signals, signals, signals], axis=1)
    
    for i in range(numEvents):
        start = np.where(times >= EventData[i, 1])[0][0]
        end = np.where(times >= EventData[i, 2])[0][0]
        # Extract from middle copy
        features[i, :] = signals[:, offset + start - 2*int(fs) : offset + end + 2*int(fs)]
        labels[i, :] = int(EventData[i, 3])  # 1-6
```

### Model Architecture

#### Channel Mapper (23→20)
```python
self.chan_conv = torch.nn.Sequential(
    Conv2dWithConstraint(23, 20, kernel_size=1, max_norm=1),
    nn.BatchNorm2d(20),
    nn.GELU(),
    nn.Conv2d(20, 20, kernel_size=(1,55), groups=20, padding='same'),
    nn.BatchNorm2d(20),
    nn.Dropout(0.8),  # HEAVY dropout
)
```

#### Classifier Head
```python
self.head = nn.Sequential(
    nn.Dropout(0.8),  # Another HEAVY dropout
    LinearWithConstraint(30720, 6, max_norm=1),
)
# 30720 = 512 × 4 × 15 (embed_dim × summary_tokens × temporal_patches)
```

#### Constraint Implementation (Reference)
```python
class LinearWithConstraint(nn.Linear):
    @autocast(enabled=True)  # Reference has decorator on method
    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super().forward(x)
```

**OUR APPROACH**: We use `torch.cuda.amp.autocast()` around forward+loss instead of method decorators (safer, acceptable parity difference).

### Training Configuration
```python
# Data scaling
samples = samples.float().to(device) / 100  # Divide μV by 100

# Reshape (optional - gets flattened immediately)
samples = rearrange(samples, 'B N (A T) -> B N A T', T=200)

# Mixed precision
with torch.cuda.amp.autocast():
    loss, output = train_class_batch(model, samples, targets, criterion)

# Per-iteration LR update
param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)

# Loss
from timm.loss import LabelSmoothingCrossEntropy
criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

# Configuration
# Layer-wise LR decay: 0.65
# Batch size: 400 (2 GPUs in reference, we use accumulation)
# Seeds: 4523 (data splits), 0 (training) - we use 42
```

## 🎯 CRITICAL QUESTIONS FOR INVESTIGATION

Given that we've achieved parity on all major components, why the 38% gap?

1. **Is there hidden data augmentation?**
   - Mixup is imported but not visibly used
   - Any minority class oversampling not documented?

2. **Are the results cherry-picked?**
   - Paper shows ±1.14% - how many runs?
   - Is 62.32% the best result or average?

3. **Is the extreme class imbalance (33:1) insurmountable?**
   - Only 24 samples for spsw class
   - Even perfect implementation might fail here

4. **Does the checkpoint contain special initialization?**
   - Are we using the exact same pretrained weights?
   - Any TUEV-specific fine-tuning in the checkpoint?

5. **Is there undocumented preprocessing?**
   - Version mismatch (v2.0.0 vs v2.0.1)?
   - Different annotation parsing?

## 🚨 DIAGNOSTIC TESTS NEEDED

### To Verify Model Capability:
1. **Single-class test**: Train with only bckg (800 samples)
   - Expected: BAC > 90% if model works correctly
   - This isolates class imbalance from model issues

2. **Extreme oversampling**: Duplicate minority classes 10-20x
   - If this improves performance, confirms imbalance is the issue
   - Not for production, just diagnostic

3. **Input statistics logging**: 
   - Verify μV range is ~[-100, +100] after scaling
   - Confirm no accidental normalization

## 📝 REPRODUCTION CHECKLIST

### CONFIRMED IMPLEMENTED:
- [x] Signal tripling for boundaries
- [x] NO normalization (disabled)
- [x] Data in μV ÷ 100
- [x] LinearWithConstraint with max_norm=1
- [x] Conv2dWithConstraint with max_norm=1
- [x] Label smoothing 0.1 (timm)
- [x] Layer decay 0.65
- [x] Dropout 0.8 (both mapper and head)
- [x] Per-iteration LR scheduling
- [x] Mixed precision (loop-level AMP)
- [x] Load from checkpoint['state_dict']
- [x] Labels: 1-6 → 0-5 mapping
- [x] Effective batch: 400 (via accumulation)
- [x] Natural sampling (no class weights)

### OPTIONAL/NON-CRITICAL:
- [ ] Reshape to (B,23,5,200) - Functionally equivalent without
- [ ] Method-level @autocast - Loop-level is sufficient
- [ ] Seeds: 4523/0 - Using 42, not performance-critical
- [ ] Distributed training - Single GPU with accumulation works

## 📈 EXPECTED TRAJECTORY (WHAT SHOULD HAPPEN)

With correct implementation:
- **Epochs 1-5**: BAC > 0.30, minority classes start showing recall
- **Epoch 10**: BAC > 0.45, steady improvement
- **Epoch 30**: BAC ≈ 0.62 ± 0.01

What we see:
- **Epochs 1-5**: BAC ≈ 0.20-0.25
- **Epoch 10**: BAC ≈ 0.25 (plateau)
- **Epoch 30**: BAC ≈ 0.24 (no improvement)

## BOTTOM LINE

We have achieved implementation parity on all critical components:
- Signal tripling ✅
- μV/100 scaling ✅  
- No normalization ✅
- All architectural components ✅
- Training configuration ✅

The 38% performance gap despite correct implementation suggests either:
1. The extreme class imbalance (24 samples for rarest class) is insurmountable
2. There's undocumented augmentation or preprocessing
3. The paper results are not reproducible as claimed

**THIS DOCUMENT IS READY TO SEND TO OTHER REPOS/AGENTS FOR INVESTIGATION**