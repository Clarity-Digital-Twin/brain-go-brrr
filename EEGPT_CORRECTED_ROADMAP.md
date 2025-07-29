# 🎯 EEGPT CORRECTED ROADMAP - Based on Paper Analysis

_Created: July 29, 2025 @ 11:55 PM_

## 🚨 CRITICAL CORRECTIONS TO PREVIOUS DOCUMENTS

After line-by-line analysis of the EEGPT paper, here's what we got WRONG and what we need to fix:

## ❌ What We Got Wrong

### 1. Feature Extraction is COMPLETELY BROKEN

```python
# CURRENT (WRONG):
summary_features = features.mean(dim=(0, 1))  # Destroys all information!

# CORRECT (from paper):
# EEGPT encoder outputs shape: (B, N_patches, embed_num, embed_dim)
# where embed_num = S = 4 (summary tokens)
# We need to extract these summary tokens, NOT average everything!
```

### 2. We Misunderstood the Architecture

- EEGPT uses **S=4 learnable summary tokens** (like [CLS] tokens)
- These aggregate information across the entire sequence
- The linear probe uses ONLY these summary tokens (4×512=2048 dims)
- We've been averaging features instead of using summary tokens!

### 3. Linear Probing Still Needs Data

- Paper uses FULL datasets, not 10 examples
- Sleep-EDFx: 197 subjects with proper train/val/test splits
- We can't expect 80% accuracy with tiny datasets

## ✅ What We Have Right

1. **Model checkpoint loads correctly** (973MB)
2. **Vision Transformer architecture exists**
3. **Preprocessing pipeline is correct**
4. **API infrastructure is solid**

## 🎯 CORRECTED MVP ROADMAP

### Phase 1: Fix Feature Extraction (2 days)

#### Day 1: Study Reference Implementation

```python
# Look at reference_repos/EEGPT/eegpt_mcae.py
# Find how they extract summary tokens

class EEGPT_mcae(nn.Module):
    def __init__(self):
        self.encoder = EEGTransformer()
        # Note: embed_num = 4 (summary tokens)

    def forward(self, x):
        # Returns shape: (B, N, embed_num, embed_dim)
        # embed_num dimension contains the summary tokens!
```

#### Day 2: Fix Our Implementation

```python
def extract_features(self, data, channel_names):
    # Current: averaging everything (WRONG)
    # Correct: extract summary tokens

    # Run through encoder
    x = self.encoder(data_tensor, chan_ids)
    # x shape: (1, N_patches, 4, 512)

    # Extract summary tokens (embed_num dimension)
    summary_tokens = x[0, :, :, :].mean(dim=0)  # Average across patches
    # Result: (4, 512) - these are our features!

    return summary_tokens  # Shape: (4, 512)
```

### Phase 2: Implement Proper Linear Probing (2 days)

#### Day 3: Create Linear Probe Module

```python
class LinearProbeClassifier(nn.Module):
    def __init__(self, num_channels_in=19, num_classes=2):
        super().__init__()
        # 1. Adaptive spatial filter (maps input channels to EEGPT's 58)
        self.spatial_filter = nn.Conv1d(num_channels_in, 58, 1)

        # 2. Linear classifier (takes 4 summary tokens × 512 dims)
        self.classifier = nn.Linear(4 * 512, num_classes)

    def forward(self, eeg_features):
        # eeg_features shape: (4, 512) from EEGPT
        features_flat = eeg_features.flatten()  # (2048,)
        logits = self.classifier(features_flat)
        return logits
```

#### Day 4: Training Pipeline

```python
def train_linear_probe(eegpt_encoder, train_loader, val_loader):
    # Freeze EEGPT encoder
    eegpt_encoder.eval()
    for param in eegpt_encoder.parameters():
        param.requires_grad = False

    # Only train the probe
    probe = LinearProbeClassifier()
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)

    for epoch in range(100):  # Paper uses 100-200 epochs
        for batch in train_loader:
            # Extract features with frozen EEGPT
            with torch.no_grad():
                features = eegpt_encoder.extract_features(batch['eeg'])

            # Train probe
            logits = probe(features)
            loss = F.cross_entropy(logits, batch['labels'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### Phase 3: Sleep Staging MVP (3 days)

#### Day 5: Prepare Sleep-EDF Data

```python
# We already have Sleep-EDF downloaded!
# Paper achieves 82.24% accuracy with linear probing

def prepare_sleep_edf():
    # Load 30s epochs as specified in paper
    # EEGPT processes 4s windows, so need sliding window

    for subject in sleep_edf_subjects:
        raw = mne.io.read_raw_edf(subject['psg'])
        annotations = mne.read_annotations(subject['hypnogram'])

        # Extract 30s epochs with labels
        # Create 4s windows with 2s overlap
        # This matches paper's approach
```

#### Day 6-7: Train & Validate

```python
# Expected results from paper:
# - Sleep-EDFx: 82.24% balanced accuracy
# - Using only linear probing
# - 10-fold cross-validation
```

## 📊 Realistic Performance Expectations

From EEGPT paper Table 2 (with linear probing):

| Task          | Paper Result | Our Target | Data Needed  |
| ------------- | ------------ | ---------- | ------------ |
| Sleep-EDFx    | 82.24%       | 75-80%     | 197 subjects |
| Motor Imagery | 58-80%       | 55-70%     | 10+ subjects |
| Abnormality   | 67.55%       | 60-65%     | 500+ files   |
| P300          | 65.02%       | 60%+       | 50+ subjects |

## 🛠️ Concrete Next Steps

### Tomorrow (Day 1):

1. **Read reference implementation carefully**

   ```bash
   cd reference_repos/EEGPT
   grep -n "summary" *.py
   grep -n "embed_num" *.py
   ```

2. **Find where summary tokens are extracted**

   ```python
   # Look for code that handles embed_num dimension
   # This is the key to fixing our features
   ```

3. **Test feature discrimination**
   ```python
   # Generate very different EEG patterns
   # Verify features are actually different
   # Current: cosine similarity = 1.0 (BAD)
   # Target: cosine similarity < 0.7 (GOOD)
   ```

### Success Criteria:

- [ ] Features discriminate between different EEG patterns
- [ ] Summary tokens extracted correctly (shape: 4×512)
- [ ] Linear probe achieves >60% on any task
- [ ] Can reproduce at least one result from paper

## 💡 Key Insights from Paper

1. **"Linear probing" = Frozen encoder + trainable linear layer**
   - Not just slapping an MLP on averaged features
   - Requires proper summary token extraction

2. **Summary tokens are the secret**
   - 4 learnable tokens that aggregate information
   - These are what get passed to downstream tasks
   - We've been ignoring them completely!

3. **Data requirements are real**
   - Can't get 80% accuracy with 10 examples
   - Need proper train/val/test splits
   - Paper uses full datasets

## 🚀 The REAL Path Forward

**We're NOT fucked!** We have:

- ✅ Valid checkpoint that loads
- ✅ Correct architecture (just using it wrong)
- ✅ Good infrastructure

We need to:

1. Fix feature extraction (use summary tokens)
2. Implement proper linear probing
3. Train on adequate data
4. Follow the paper EXACTLY

**Time to working MVP: 1 week** (not 6 months)

---

_Previous documents claimed we had "no EEGPT" - this is WRONG. We have EEGPT, we're just using it incorrectly. The path forward is clear: fix feature extraction, implement proper linear probing, train on real data._
