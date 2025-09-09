# 🧠 TUSZ TEMPORAL DETECTION - CRITICAL BRAINSTORMING

**Created**: September 9, 2025  
**Status**: 🔴 ACTIVE RESEARCH  
**Purpose**: Determine optimal architecture for TUSZ temporal seizure detection

---

## 🚨 FUNDAMENTAL QUESTION

**Should we use EEGPT for TUSZ, or do we need a completely different pipeline?**

---

## 📊 THE CORE DIFFERENCE: Classification vs Detection

### TUAB/TUEV (What EEGPT was designed for)
- **Task**: Binary/multi-class classification
- **Input**: 4-second windows (independent)
- **Output**: Single label per window
- **Temporal**: NO - windows can be shuffled
- **Pipeline**: EEG → EEGPT → Linear probe → Class
- **Success**: EEGPT works great! (TUAB: 86.9% AUROC)

### TUSZ (Fundamentally different)
- **Task**: Temporal event detection
- **Input**: SEQUENCES of windows (order matters!)
- **Output**: Start/stop times of seizures
- **Temporal**: YES - must track state across time
- **Pipeline**: EEG → ??? → Temporal model → Post-process → Events
- **Challenge**: EEGPT alone can't handle temporal dependencies

---

## 🔬 RESEARCH FINDINGS (2025)

### What the Literature Says

#### EEGPT Paper Analysis (From Our Literature)
- **Architecture**: Vision Transformer with masked autoencoding (10M parameters)
- **Input**: 4-second windows @ 256Hz (1024 samples)
- **Patches**: 250ms temporal patches (64 samples each)
- **Output**: 2048-dim embeddings (4×512 summary tokens)
- **Training**: Mask-based reconstruction + spatio-temporal alignment
- **Key limitation**: Processes each window INDEPENDENTLY
- **No temporal modeling**: Each 4s window treated as isolated sample
- **Success domains**: TUAB (86.9% AUROC), sleep staging, MI classification

#### TUSZ SOTA (2024-2025 Web Research)
- **ResBiLSTM**: 95.03% accuracy on TUSZ
- **GAT-BiLSTM**: 98.52% accuracy (spatial-temporal)
- **SE-TCN-BiGRU**: Strong performance
- **TCN advantages**: Better than RNNs for long sequences
- **Trend**: Moving away from feature engineering (tsfresh) to end-to-end

#### TUSZ Corpus Details (Shah 2018)
- **196 train, 50 eval patients**
- **~8% seizure prevalence** (massive class imbalance)
- **Annotations**: SPSW, PLED, GPED events
- **Challenge**: Temporal localization critical

#### Picone 2021 Evaluation Framework
- **TAES**: Time-Aligned Event Scoring (Jaccard ≥ 0.5)
- **ATWV**: Borrowed from speech (β = 999.9)
- **FA/24h**: Clinical gold standard
- **Key insight**: OVLP (any overlap) too permissive
- **Best systems**: CNN+LSTM architectures

#### Critical Insights
1. **EEGPT lacks temporal continuity** - treats windows as bags, not sequences
2. **Seizures evolve temporally** - require state tracking across windows
3. **Post-processing critical** - merge gaps, minimum duration
4. **Evaluation complexity** - temporal alignment matters

---

## 💡 ARCHITECTURAL OPTIONS (UPDATED WITH SeizureTransformer)

### Option 1: EEGPT + Temporal Head (Original Idea)
```
EEG Windows → EEGPT (features) → BiLSTM → Detection
```
**Verdict**: ❌ WRONG APPROACH
- EEGPT processes windows independently
- No temporal continuity
- Overkill features (2048-dim)

### Option 2: CNN + BiLSTM (Picone 2021 Best)
```
EEG → LFCC/CNN features → BiLSTM → 3-Stage Post-Process
```
**Verdict**: ✅ PROVEN TO WORK
- Best in Picone's evaluation
- Requires heavy post-processing
- Still only 0.45 ATWV

### Option 3: SeizureTransformer (Wu 2025 - NEW!)
```
EEG → CNN Encoder → Transformer (8 layers) → U-Net Decoder → Time-step Probabilities
```
**Verdict**: 🔥 MOST PROMISING
- Direct time-step predictions
- Global temporal attention
- No post-processing needed
- Claims SOTA performance

### Option 4: Parallel Pipelines (Final Architecture)
```
TUAB/TUEV: EEG → EEGPT → Linear probe
TUSZ:      EEG → SeizureTransformer variant
```
**Verdict**: ✅ RECOMMENDED
- Best tool for each task
- EEGPT for classification (proven)
- SeizureTransformer for temporal detection

---

## 🤔 KEY QUESTIONS TO ANSWER

1. **Can EEGPT embeddings capture seizure patterns?**
   - EEGPT trained on reconstruction, not seizure detection
   - Embeddings might miss subtle temporal transitions
   - Need empirical testing

2. **Is temporal modeling more important than features?**
   - Literature suggests YES
   - BiLSTM consistently beats static classifiers
   - Post-processing is critical

3. **Should we treat TUSZ as a separate problem?**
   - Different evaluation metrics (FA/24h vs accuracy)
   - Different data requirements (sequences vs windows)
   - Different clinical use case

---

## 🧪 PROPOSED EXPERIMENTS

### Experiment 1: EEGPT Baseline
```python
# Test if EEGPT can even detect seizures (window-level)
for window in seizure_windows:
    embedding = eegpt.encode(window)
    # Can we separate seizure from non-seizure embeddings?
```

### Experiment 2: EEGPT + LSTM
```python
# Add minimal temporal modeling to EEGPT
class EEGPTTemporal(nn.Module):
    def __init__(self):
        self.eegpt = load_pretrained_eegpt()
        self.lstm = nn.LSTM(2048, 256, bidirectional=True)
        self.fc = nn.Linear(512, 1)
```

### Experiment 3: Lightweight CNN-LSTM
```python
# Skip EEGPT entirely
class LightweightTUSZ(nn.Module):
    def __init__(self):
        self.cnn = nn.Conv1d(22, 64, kernel_size=15)  # Extract features
        self.lstm = nn.LSTM(64, 128, bidirectional=True)
        self.fc = nn.Linear(256, 1)
```

---

## 📈 PERFORMANCE TARGETS

### Minimum Viable
- FA/24h < 20 @ 90% sensitivity
- ATWV > 0.3
- Process 1 hour in < 5 minutes

### Production Ready
- FA/24h < 10 @ 95% sensitivity  
- ATWV > 0.5
- Real-time processing capability

### SOTA Competitive
- Match ResBiLSTM: 95%+ accuracy
- FA/24h < 5
- ATWV > 0.6

---

## 🎯 RECOMMENDATION (AFTER DEEP RESEARCH)

Based on comprehensive analysis:

1. **CONFIRMED: Use Option 3 - Parallel Pipelines**
   - Keep EEGPT for TUAB/TUEV (proven 86.9% AUROC)
   - Build specialized temporal model for TUSZ
   - Different problems require different architectures

2. **For TUSZ specifically**:
   ```python
   # Recommended Architecture
   EEG → CNN (lightweight features) → BiLSTM (temporal) → Post-process → Events
   
   # NOT Recommended
   EEG → EEGPT (2048-dim) → BiLSTM → Events  # Overkill & wrong abstraction
   ```

3. **Why EEGPT is WRONG for TUSZ**:
   - **Fundamental mismatch**: EEGPT does window classification, TUSZ needs sequence modeling
   - **No temporal memory**: Each 4s window processed independently
   - **Computational waste**: 10M params for feature extraction when 100K CNN works
   - **Wrong training objective**: Masked reconstruction vs temporal evolution
   - **Literature evidence**: All SOTA TUSZ systems use lightweight CNN + RNN

4. **What We Actually Need**:
   - **Temporal state tracking** across windows
   - **Sequential processing** not parallel windows
   - **Lightweight features** (64-128 dims, not 2048)
   - **Heavy temporal modeling** (BiLSTM/TCN)
   - **Sophisticated post-processing**

---

## 🔥 GAME-CHANGER: SeizureTransformer (Wu 2025)

### Revolutionary Architecture (Just Found!)
- **Paper**: "Scaling U-Net with Transformer for Simultaneous Time-Step Level Seizure Detection"
- **Key Innovation**: TIME-STEP LEVEL predictions (not window-level!)
- **Architecture**: 
  - 1D CNN encoder (32→512 filters)
  - Residual CNN stack
  - **Transformer encoder with GLOBAL ATTENTION** (8 layers, 4 heads)
  - U-Net style decoder with skip connections
  - Direct sigmoid output per time step

### Why This Changes EVERYTHING
1. **No window classification** - Direct time-step probabilities
2. **No complex post-processing** - Built into architecture
3. **Transformer attention** - Captures long-range temporal dependencies
4. **U-Net design** - Multi-scale temporal features
5. **Claims to outperform ALL existing approaches**

### Critical Design Choices
- **Input**: 19 channels × 15360 samples (60 seconds @ 256Hz)
- **Positional encoding**: Added to transformer
- **Skip connections**: From encoder to decoder (U-Net style)
- **Output**: Probability at EVERY time step

### This Solves Our Dilemma!
- Uses Transformer (like EEGPT) but PROPERLY for temporal
- Processes long sequences with global attention
- No need for LSTM hidden state management
- Direct seizure probability per sample

## 🚀 UPDATED NEXT STEPS

1. [ ] Study SeizureTransformer architecture in detail
2. [ ] Compare with our EEGPT + BiLSTM approach
3. [ ] Implement SeizureTransformer variant for TUSZ
4. [ ] Test against CNN-BiLSTM baseline
5. [ ] Make final architecture decision

---

## 📝 CRITICAL INSIGHTS FROM DR. PICONE'S LITERATURE

### From Picone 2021 - Objective Evaluation Metrics

#### Scoring Metrics Reality Check
- **OVLP (Any Overlap)**: Too permissive, gives artificially high sensitivity
- **EPOCH**: Samples at fixed intervals (0.25s typical), weighs long events heavily
- **TAES**: Our target - considers % overlap, balances short/long events
- **ATWV**: β = 999.9 for seizure detection, <0.5 indicates poor performance
- **Key**: All 5 TUSZ systems scored ATWV < 0.5 (even CNN/LSTM at best)

#### System Performance on TUSZ v1.1.1
| System | Architecture | Key Result |
|--------|-------------|------------|
| HMM/SdA | HMM + Stacked Denoising Autoencoder | Baseline, high FA |
| HMM/LSTM | HMM + LSTM postprocessor | Better than SdA |
| IPCA/LSTM | PCA + LSTM | Detects longer events |
| CNN/MLP | Pure deep learning | Simple but effective |
| **CNN/LSTM** | CNN + LSTM | **Best: Lowest FA rate** |

#### Critical Implementation Details
- **Features**: LFCC (Linear Frequency Cepstral Coefficients)
  - 0.1 sec frame, 0.2 sec window
  - 7 cepstral coefficients + derivatives = 26 dims/channel
  - Energy terms added
- **Context Window**: 7-41 frames typical
- **Post-processing**: 3-stage with language model smoothing

#### FA/24h Reality
- **Clinical requirement**: < 10 FA/24h
- **Best system (CNN/LSTM)**: Still far from clinical acceptance
- **Key insight**: Low FA region (0-0.2 FPR) is what matters clinically

### From Shah 2018 - TUSZ Corpus Details

#### Dataset Statistics
- **v1.1.1**: 196 train, 50 eval patients
- **Seizure prevalence**: ~8% (massive imbalance)
- **Annotations**: 6 classes (3 signal, 3 noise)
  - Signal: SPSW, PLED, GPED
  - Noise: ARTF, EYEM, BCKG
- **Variability**: Wide range of seizure durations (seconds to minutes)

### From Lopez 2017 - TUAB Thesis

#### Human vs Machine Performance
- **Human error rate**: ~1% for normal/abnormal classification
- **Best ML (CNN-MLP)**: 21.2% error rate
- **HMM baseline**: 26.1% error rate
- **Key**: Even simple normal/abnormal is hard for ML

### CRITICAL OBSERVATIONS

1. **No One Has Solved TUSZ Well**
   - Best ATWV < 0.5 (poor by speech standards)
   - FA rates still too high for clinical use
   - Gap between research and deployment

2. **Temporal Alignment is Everything**
   - OVLP gives false confidence (100% sensitivity misleading)
   - TAES/EPOCH reveal true performance
   - Start/stop times matter clinically

3. **Post-Processing > Model Architecture**
   - 3-stage post-processing critical
   - Language model smoothing helps
   - Merge gaps, minimum duration essential

4. **Feature Engineering Still Relevant**
   - LFCC features work well (26 dims)
   - Raw signal → Deep learning didn't help much
   - Context window size matters (7-41 frames)

5. **Class Imbalance Kills Performance**
   - 8% seizure prevalence
   - Need weighted loss functions
   - FA rate dominates clinical acceptance

---

## 🔗 REFERENCES

- Picone 2021: TUSZ evaluation framework
- EEGPT paper: Window-level tasks only
- 2024 Research: BiLSTM dominates TUSZ
- Clinical requirement: FA/24h < 10

---

**THIS DOCUMENT IS ACTIVELY EVOLVING - DECISIONS NOT FINAL**