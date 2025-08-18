# 🔴 TUEV CRITICAL ARCHITECTURE - FROM TABLE 13 OF PAPER

## ⚠️ THIS IS COMPLETELY DIFFERENT FROM WHAT WE THOUGHT!

### Exact Architecture from Paper (Table 13, page 20)

```
Input Size: 23 × 1000 (NOT 23 × 1280!)
↓
Conv1d: kernel=1, stride=1, groups=1, padding=0
  23 channels → 20 channels (REDUCTION, not expansion to 58!)
↓
BatchNorm + GELU
  20 × 1000
↓
Conv1d: kernel=55, stride=1, groups=20, padding=27 (depthwise)
  20 × 1000 → 20 × 1000
↓
BatchNorm + GELU
  20 × 1000
↓
Dropout(0.5)  ← CRITICAL: 0.5, not 0.25 like TUAB!
  20 × 1000
↓
EEGPT Encoder: kernel=64, stride=64
  20 × 1000 → 15 × 4 × 512
↓
Flatten + Linear
  15 × 4 × 512 → 6 classes
```

### The 20 Channels Used (EXACT ORDER MATTERS!)

From page 615 of the paper:
```python
channels = [
    'FP1', 'FPZ', 'FP2',  # Frontal
    'F7', 'F3', 'FZ', 'F4', 'F8',  # Frontal
    'T7', 'C3', 'CZ', 'C4', 'T8',  # Temporal/Central
    'P7', 'P3', 'PZ', 'P4', 'P8',  # Parietal
    'O1', 'O2'  # Occipital
]
```

### Critical Implementation Details

1. **Input is 1000 samples, not 1280!**
   - 1000 samples @ 256 Hz = 3.90625 seconds
   - Maybe they take 5s windows and subsample to 1000?
   - Or maybe the "5-second" claim is wrong?

2. **Channel Reduction (23→20), NOT Expansion!**
   - First conv reduces from 23 TUEV channels to 20 standard channels
   - Maps to specific 10-20 system channels listed above

3. **Depthwise Temporal Convolution**
   - Groups=20 means each channel processed independently
   - Kernel=55 with padding=27 maintains size

4. **Higher Dropout for TUEV**
   - 0.5 dropout vs 0.25 for TUAB
   - Likely due to smaller dataset / overfitting risk

5. **Output Shape Different**
   - TUAB: 31 × 4 × 512
   - TUEV: 15 × 4 × 512
   - Because 1000/64 = 15.625 → 15 patches

### Optimizer Configuration

From page 587:
```python
optimizer = AdamW(lr=5e-4)  # Constant, no schedule!
batch_size = 500
```

**NOT OneCycle like pretraining!**

### What This Means for Implementation

```python
class TUEVLinearProbe(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Channel selection (23 → 20)
        self.channel_reducer = nn.Conv1d(23, 20, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(20)
        
        # Temporal convolution (depthwise)
        self.temporal_conv = nn.Conv1d(
            20, 20, 
            kernel_size=55, 
            groups=20,  # Depthwise!
            padding=27
        )
        self.bn2 = nn.BatchNorm1d(20)
        
        # Dropout - HIGHER than TUAB!
        self.dropout = nn.Dropout(0.5)
        
        # Classifier head
        self.classifier = nn.Linear(15 * 4 * 512, 6)
        
    def forward(self, x):
        # x: (batch, 23, 1000)
        x = F.gelu(self.bn1(self.channel_reducer(x)))
        x = F.gelu(self.bn2(self.temporal_conv(x)))
        x = self.dropout(x)
        
        # Pass through frozen EEGPT
        features = self.eegpt_encoder(x)  # (batch, 15, 4, 512)
        
        # Flatten and classify
        features = features.view(features.size(0), -1)
        return self.classifier(features)
```

### ⚠️ CRITICAL QUESTIONS TO RESOLVE

1. **Why 1000 samples instead of 1280?**
   - Paper says "5-second samples" but architecture uses 1000
   - 1000 @ 256Hz = 3.90625s, not 5s
   - Subsampling? Cropping? Error in paper?

2. **How to handle our 250Hz data?**
   - Resample 250→256Hz first?
   - Or adjust window to 976 samples (3.90625s @ 250Hz)?

3. **Window extraction strategy?**
   - Sliding windows or non-overlapping?
   - Paper doesn't specify for TUEV

### VALIDATION

Before training, MUST verify:
```python
assert input_shape == (batch, 23, 1000), "Wrong input!"
assert dropout_rate == 0.5, "Wrong dropout!"
assert n_channels_after_conv1 == 20, "Wrong channel mapping!"
assert optimizer_lr == 5e-4, "Wrong learning rate!"
assert batch_size == 500, "Wrong batch size!"
```

---

**THIS CHANGES EVERYTHING ABOUT OUR TUEV IMPLEMENTATION!**