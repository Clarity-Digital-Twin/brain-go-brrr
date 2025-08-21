# 🚀 EEGPT MVP - Test-Driven Development Plan

_Created: July 29, 2025 @ Midnight_

## 🎯 THE TRUTH: We Have EEGPT, We're Just Using It Wrong!

After deep analysis, we have a CLEAR path to a working MVP. The foundation is GOOD, implementation is WRONG.

## 🔴 Current State: Why Nothing Works

```python
# THE SMOKING GUN (lines 223-227 in eegpt_model.py):
summary_features = features.mean(dim=(0, 1))  # Averages EVERYTHING!
summary_features = summary_features.expand(4, -1)  # Just copies 4 times!

# This is why:
# - All features are identical
# - Cosine similarity = 1.0
# - "Abnormality detection" is random
```

## 🟢 What Should Happen (Per Paper)

EEGPT uses **4 summary tokens** that aggregate information. These are learned during pretraining and are the ONLY features used for downstream tasks.

```python
# Correct approach:
# 1. EEGPT encoder outputs summary tokens directly
# 2. We extract these tokens (not average everything)
# 3. Linear probe uses these 4×512 features
```

## 📋 TDD Roadmap for Working MVP

### Day 1: Fix Feature Extraction (TDD)

#### Test 1: Features Should Discriminate

```python
def test_features_discriminate_between_patterns():
    """Different EEG patterns should produce different features."""
    model = EEGPTModel()

    # Generate very different patterns
    alpha_waves = generate_sine_wave(10)  # 10 Hz alpha
    beta_waves = generate_sine_wave(25)   # 25 Hz beta
    seizure = generate_spike_wave(3)      # 3 Hz spike-wave

    # Extract features
    feat_alpha = model.extract_features(alpha_waves, ch_names)
    feat_beta = model.extract_features(beta_waves, ch_names)
    feat_seizure = model.extract_features(seizure, ch_names)

    # Features should be different!
    sim_alpha_beta = cosine_similarity(feat_alpha, feat_beta)
    sim_alpha_seizure = cosine_similarity(feat_alpha, feat_seizure)

    assert sim_alpha_beta < 0.9, f"Alpha/Beta too similar: {sim_alpha_beta}"
    assert sim_alpha_seizure < 0.8, f"Alpha/Seizure too similar: {sim_alpha_seizure}"
```

#### Test 2: Summary Token Shape

```python
def test_summary_token_extraction():
    """Should extract 4 summary tokens of 512 dims each."""
    model = EEGPTModel()

    # Any EEG data
    data = np.random.randn(19, 1024)
    features = model.extract_features(data, ch_names)

    # Should be (4, 512) not averaged garbage
    assert features.shape == (4, 512), f"Wrong shape: {features.shape}"

    # Tokens should be different from each other
    token_sims = []
    for i in range(4):
        for j in range(i+1, 4):
            sim = cosine_similarity(features[i], features[j])
            token_sims.append(sim)

    assert max(token_sims) < 0.95, "Summary tokens too similar!"
```

### Day 2: Implement Linear Probe (TDD)

#### Test 3: Linear Probe Architecture

```python
def test_linear_probe_architecture():
    """Linear probe should match paper specification."""
    probe = LinearProbeHead(num_classes=2)

    # Input: 4 summary tokens × 512 dims
    features = torch.randn(1, 4, 512)
    features_flat = features.view(1, -1)  # (1, 2048)

    # Output: class logits
    logits = probe(features_flat)
    assert logits.shape == (1, 2)

    # Should be trainable
    assert len(list(probe.parameters())) > 0
```

#### Test 4: Frozen Encoder + Trainable Probe

```python
def test_encoder_frozen_probe_trainable():
    """Encoder stays frozen, only probe trains."""
    model = EEGPTModel()
    probe = LinearProbeHead()

    # Freeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = False

    # Probe remains trainable
    trainable_params = sum(p.numel() for p in probe.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.encoder.parameters() if not p.requires_grad)

    assert trainable_params > 0, "Probe must be trainable"
    assert frozen_params > 0, "Encoder must be frozen"
```

### Day 3: Sleep Staging MVP (TDD)

#### Test 5: Sleep Data Processing

```python
def test_sleep_edf_processing():
    """Process Sleep-EDF into 4s windows for EEGPT."""
    # We have Sleep-EDF data!
    edf_path = "data/datasets/external/sleep-edf/SC4001E0-PSG.edf"

    raw = mne.io.read_raw_edf(edf_path, preload=True)

    # Should create 4s windows with labels
    windows, labels = prepare_sleep_windows(raw)

    assert windows.shape[2] == 1024  # 4s × 256Hz
    assert len(labels) > 0
    assert set(labels).issubset({0, 1, 2, 3, 4})  # 5 sleep stages
```

#### Test 6: End-to-End Sleep Classification

```python
def test_sleep_classification_pipeline():
    """Full pipeline: EEG → EEGPT → Linear Probe → Sleep Stage."""
    model = EEGPTModel()
    probe = LinearProbeHead(num_classes=5)  # 5 sleep stages

    # Load one window
    window = load_sleep_window()  # Shape: (19, 1024)

    # Extract features
    with torch.no_grad():
        features = model.extract_features(window, ch_names)

    # Classify
    features_flat = features.flatten().unsqueeze(0)
    logits = probe(features_flat)

    assert logits.shape == (1, 5)
    probs = F.softmax(logits, dim=-1)
    assert probs.sum() == pytest.approx(1.0)
```

### Day 4: Training Pipeline (TDD)

#### Test 7: Training Loop

```python
def test_linear_probe_training():
    """Can train linear probe on real data."""
    model = EEGPTModel()
    probe = LinearProbeHead(num_classes=5)

    # Mini dataset
    train_data = [(window1, 0), (window2, 1), (window3, 2)]

    optimizer = torch.optim.Adam(probe.parameters())
    initial_loss = None

    for epoch in range(10):
        epoch_loss = 0
        for window, label in train_data:
            # Forward pass
            with torch.no_grad():
                features = model.extract_features(window, ch_names)

            logits = probe(features.flatten().unsqueeze(0))
            loss = F.cross_entropy(logits, torch.tensor([label]))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if initial_loss is None:
            initial_loss = epoch_loss

    # Loss should decrease
    assert epoch_loss < initial_loss * 0.8
```

### Day 5: API Integration (TDD)

#### Test 8: API Endpoint

```python
def test_api_sleep_classification():
    """API endpoint for sleep classification."""
    client = TestClient(app)

    # Upload EDF
    with open("test_sleep.edf", "rb") as f:
        response = client.post(
            "/api/v1/eeg/sleep/classify",
            files={"edf_file": f}
        )

    assert response.status_code == 200
    data = response.json()

    # Should return hypnogram
    assert "hypnogram" in data
    assert "confidence" in data
    assert data["confidence"] > 0.5  # Not random!
```

## 🎯 Success Metrics

### Week 1 Goals:

- [ ] Fix feature extraction (pass discrimination tests)
- [ ] Implement linear probe (match paper architecture)
- [ ] Train on Sleep-EDF subset (>60% accuracy)
- [ ] Deploy one working endpoint

### Expected Performance (Realistic):

| Task     | Paper | Our Target | Timeline |
| -------- | ----- | ---------- | -------- |
| Sleep    | 82.2% | 60-70%     | Week 1   |
| Abnormal | 67.6% | 55-60%     | Week 2   |
| Motor    | 58.5% | 50-55%     | Week 3   |

## 📝 Critical Implementation Notes

### 1. Fix Feature Extraction:

```python
# WRONG (current):
summary_features = features.mean(dim=(0, 1))

# RIGHT (todo):
# Check reference implementation for how to get summary tokens
# Likely involves special tokens or pooling mechanism
```

### 2. Data Requirements:

- Sleep-EDF: We have 197 subjects ✅
- Need proper train/val/test splits
- Can start with 10% for quick iteration

### 3. Training Details:

- Linear probe only (encoder frozen)
- 100-200 epochs typical
- Learning rate: 1e-3 to 1e-4
- Batch size: 32-64

## 🚀 The Path is Clear!

**We are NOT fucked!** We have:

1. Valid EEGPT checkpoint ✅
2. Working architecture ✅
3. Clear bugs to fix ✅
4. TDD tests to guide us ✅

**One week to working MVP** following this TDD approach.

---

_Key insight: We've been trying to average our way to glory when EEGPT already provides summary tokens. Time to use the model as designed!_
