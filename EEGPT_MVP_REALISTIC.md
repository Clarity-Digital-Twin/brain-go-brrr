# 🚀 REALISTIC EEGPT MVP - Linear Probing Approach

_Created: July 29, 2025 @ 11:45 PM_

## 💡 THE BREAKTHROUGH: We CAN make it work!

After reading the EEGPT paper carefully, they achieve ALL their results using **LINEAR PROBING** - just adding a linear layer on top of frozen EEGPT features!

## 🎯 What Linear Probing Means

```python
class EEGPTWithLinearProbe(nn.Module):
    def __init__(self, pretrained_eegpt, num_classes=2):
        super().__init__()
        # FREEZE the pretrained encoder (our 973MB checkpoint)
        self.encoder = pretrained_eegpt
        for param in self.encoder.parameters():
            param.requires_grad = False  # FROZEN!

        # Add ONLY a linear head (this is what we train)
        self.classifier = nn.Linear(512 * 4, num_classes)  # 4 summary tokens

    def forward(self, x):
        # Extract features with frozen EEGPT
        features = self.encoder.extract_features(x)  # Shape: (4, 512)
        features_flat = features.flatten()  # Shape: (2048,)

        # Apply linear classifier
        logits = self.classifier(features_flat)
        return logits
```

## 📊 Training Requirements (REALISTIC)

### For Abnormality Detection Demo:

- **Data needed**: 100-1000 labeled EEGs (not 10,000!)
- **Training time**: 1-2 hours on 4090
- **Cost**: $0 (use TUH data we already have)
- **Accuracy target**: 65-75% (not 82% from paper)

### What We Need:

1. **Fix feature extraction** - Currently outputs identical features
2. **Get labeled data** - TUH abnormal subset
3. **Train linear layer** - Standard PyTorch training loop
4. **Validate results** - Cross-validation

## 🛠️ Concrete Implementation Plan

### Week 1: Fix Feature Extraction

```python
# Problem: All features are identical
# Solution: Check if we're loading weights correctly

def test_feature_discrimination():
    model = EEGPTModel()

    # Generate very different signals
    seizure = generate_spike_wave_pattern()
    normal = generate_normal_alpha()

    feat1 = model.extract_features(seizure)
    feat2 = model.extract_features(normal)

    similarity = cosine_similarity(feat1, feat2)
    assert similarity < 0.8  # Should be different!
```

### Week 2: Prepare TUH Data

```python
# We already have TUH downloaded!
tuh_path = "data/datasets/external/tuh_eeg_abnormal/v3.0.1/"

def prepare_tuh_subset():
    abnormal_files = glob(f"{tuh_path}/abnormal/**/*.edf")[:500]
    normal_files = glob(f"{tuh_path}/normal/**/*.edf")[:500]

    # Create balanced dataset
    dataset = []
    for edf in abnormal_files:
        dataset.append({"path": edf, "label": 1})
    for edf in normal_files:
        dataset.append({"path": edf, "label": 0})

    return dataset
```

### Week 3: Train Linear Probe

```python
def train_linear_probe(frozen_eegpt, train_data, val_data):
    # Only train the linear layer
    classifier = nn.Linear(2048, 2)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(50):  # Much fewer epochs needed
        for batch in train_data:
            # Extract features with frozen EEGPT
            with torch.no_grad():
                features = frozen_eegpt.extract_features(batch['eeg'])

            # Train only the classifier
            logits = classifier(features.flatten())
            loss = criterion(logits, batch['labels'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

## 💰 Cost Breakdown (REAL)

### What the $200k was for:

- **Pretraining EEGPT**: 8×A100 for 2 weeks
- **33,000 hours of EEG data**: Multiple datasets
- **NOT needed for linear probing!**

### What we actually need:

- **1× RTX 4090**: You already have
- **100-1000 labeled EEGs**: TUH has thousands
- **1-2 days training**: ~$50 electricity
- **No FDA approval** for demo

## 🎯 Realistic Performance Targets

From the EEGPT paper Table 2:

- **TUAB (Abnormal)**: 82.2% with linear probe
- **Sleep-EDFx**: 71.68% with linear probe
- **Motor Imagery**: 65.02% with linear probe

Our realistic targets:

- **Abnormality**: 65-70% (good enough for demo)
- **Sleep**: Use YASA instead (87% today)
- **Motor Imagery**: 55-60% (proof of concept)

## 📋 Action Plan for THIS WEEK

### Day 1-2: Debug Feature Extraction

```bash
# Why are all features identical?
1. Check weight loading
2. Test on synthetic data
3. Verify forward pass
4. Fix the feature averaging bug
```

### Day 3-4: Prepare Data

```bash
# Use existing TUH data
1. Load 500 normal + 500 abnormal
2. Convert to 4-second windows
3. Create train/val/test splits
4. Save as .npz for fast loading
```

### Day 5-7: Train & Deploy

```bash
# Linear probe training
1. Implement linear probe class
2. Train on TUH subset
3. Validate performance
4. Deploy to API
```

## 🚨 Critical Insights

1. **We DON'T need to train EEGPT** - Just use frozen features
2. **We DON'T need $200k** - Linear probing is cheap
3. **We DON'T need 33k hours of data** - 100-1000 samples work
4. **We CAN do this on your 4090** - It's just a linear layer!

## 💡 Why Our Features Are Broken

Looking at the code, the issue is likely:

```python
# Current (BROKEN):
summary_features = features.mean(dim=(0, 1))  # Averages everything!
summary_features = summary_features.expand(4, -1)  # Just copies!

# Should be:
summary_features = self.encoder.get_summary_tokens()  # Use CLS tokens
```

## 🎯 Success Criteria for MVP

- [ ] Features discriminate between patterns (cosine < 0.8)
- [ ] Linear probe achieves >60% on TUH subset
- [ ] API endpoint returns real probabilities (not random)
- [ ] Processing time <10 seconds per EEG
- [ ] Can demo to investors

## 📝 The Truth About EEGPT

**What they did**: Pretrained on 33k hours, then LINEAR PROBE
**What we need**: Use their checkpoint, add LINEAR LAYER
**Training cost**: ~$50 in electricity, not $200k
**Time to MVP**: 1 week, not 6 months

---

_Bottom line: We CAN build a working demo with linear probing. The $200k was for pretraining, which is already done. We just need to fix the feature extraction bug and train a single linear layer._
