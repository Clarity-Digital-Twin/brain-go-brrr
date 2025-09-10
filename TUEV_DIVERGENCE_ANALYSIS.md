# TUEV Implementation Divergence Analysis: How We Got It Wrong

## Executive Summary

We fundamentally misunderstood the TUEV task. We implemented **continuous event detection** (sliding windows over full recordings) while EEGPT implemented **event segment classification** (pre-extracted segments around known events). This is a completely different problem.

Status check: the core conclusion above is correct. Our current pipeline optimizes a harder temporal detection problem; the EEGPT reference optimizes classification of event-centered segments.

## The Critical Misunderstanding

### What EEGPT Actually Does (from their code):
```python
# From make_TUEV.py
def BuildEvents(signals, times, EventData):
    # Extract 5-second segments around ANNOTATED events only
    features = np.zeros([numEvents, numChan, int(fs) * 5])  # 5 seconds at 200Hz
    for i in range(numEvents):
        # Get 2 seconds before and after the event (5s total)
        features[i, :] = signals[:, offset + start - 2*int(fs) : offset + end + 2*int(fs)]
        labels[i, :] = int(EventData[i, 3])  # Single label per segment
```

Notes from the reference repo:
- Filtering: 0.1–75 Hz, notch at 50 Hz, resample to 200 Hz.
- Channels: uses TUH “-REF” channel names; a later model stage maps from 23 REF inputs to 20 classifier channels via a small channel conv (not bipolar).

### What We Implemented:
```python
# From tuev_dataset.py
# Create 4-second windows from annotation
window_duration = 4.0  # seconds
current_start = start_sec
while current_start < end_sec:
    # SLIDING WINDOWS over ENTIRE recording
    window_end = min(current_start + window_duration, end_sec)
    annotations.append({'start': current_start, 'end': window_end, 'label': label})
    current_start += window_duration  # Slide to next window
```

## How This Divergence Happened: The Chain of Mistakes

### Mistake #1: Copied TUAB's Approach Without Thinking
- **TUAB** is binary abnormal/normal detection on full recordings → sliding windows make sense
- **TUEV** is multi-class event classification → needs event-centered segments
- We blindly applied TUAB's pipeline to TUEV without considering the fundamental difference

### Mistake #2: Misinterpreted "288" in Table 1
```
Table 1: Datasets for pretraining and downstream tasks
| TUEV | Event | 288 | 6 |
```
- We thought: "288 total samples? That's weird but okay..."
- Reality: 288 likely means subjects/sessions, with many segments per subject
- EEGPT extracts ~100-200 segments per subject around annotated events

### Mistake #3: Ignored the Input Dimensions
```
Table 13: Model design for TUEV dataset
| 23 × 1000 | conv1d | ... |
```
- 1000 samples = 5 seconds at 200Hz (not 4 seconds at 256Hz)
- They resample to 200Hz and use 5-second windows
- We kept 256Hz and used 4-second windows

### Mistake #4: Didn't Question the Class Imbalance
- Our approach: 99.5% background, 0.5% events → extreme imbalance
- Their approach: Only event segments → naturally balanced
- We thought: "We'll just use class weights!" (Wrong problem entirely)

## The Paper's Ambiguity That Led Us Astray

### What the Paper Says:
> "Each dataset underwent similar and distinct preprocessing steps, including cropping (4s), re-referencing (average), channels selecting, scaling (mV) and resampling (256Hz)."

### What It Actually Means:
- "cropping (4s)" refers to OTHER datasets (MI, Sleep, etc.)
- TUEV uses 5-second windows at 200Hz (see Table 13)
- The preprocessing description is generic, not TUEV-specific

### Missing Critical Details:
1. Paper never mentions they only extract event segments
2. Paper never mentions bipolar montage conversion (and the reference code does not apply it)
3. Paper never explains "288" (subjects? segments? files?)
4. Paper shows high F1 (81.87%) with moderate BAC (62.32%) but doesn't explain why

## Why Our Training Failed Catastrophically

### The Numbers Tell the Story:
- **Our dataset (measured)**: 180,205 windows in train cache, with 179,444 labeled as background (99.58%). File: `data/cache/tuev_23ch_paper_parity/train/index_train_mne-ar-v4.json`.
- **Their dataset**: ~1,000 segments, balanced across 6 classes
- **Our result**: Model learns to predict only background (16.67% BAC)
- **Their result**: Model learns all classes (62.32% BAC)

### The Loss Function Disaster:
```python
# Both use the same loss
criterion = CrossEntropyLoss(label_smoothing=0.1)

# But on completely different data distributions:
# Ours: [9950 BCKG, 20 SPSW, 10 GPED, 10 PLED, 5 EYEM, 5 ARTF]
# Theirs: [~167 each class, balanced]
```

## The Cascading Effects

### 1. We Added Complexity to "Fix" the Wrong Problem:
- Added MNE+Autoreject preprocessing (not in EEGPT)
- Added channel mapper (23→20 conversion)
- Tried class weights (not needed for balanced data)
- Added paper parity mode (chasing ghosts)

### 2. We Debugged the Wrong Things:
- Spent time on normalization (not the issue)
- Debugged channel ordering (not the issue)
- Fixed cache versions multiple times (not the issue)
- The real issue: **We're solving a different problem**

### 3. We Created Technical Debt:
- Multiple cache versions (v1, v2, v3, v4)
- Two dataset implementations (20ch vs 23ch)
- Complex preprocessing pipeline
- All for the wrong task definition

## How TUAB Success Masked the TUEV Problem

### TUAB Worked Because:
- Binary classification is simpler
- Full-recording analysis matches our sliding window approach
- 83% AUROC achieved → seemed like we understood EEGPT

### This Created False Confidence:
- "We got TUAB working, TUEV should be similar"
- "Just change the number of classes from 2 to 6"
- "The same pipeline should work"

## The Root Cause: Insufficient Investigation

### We Didn't:
1. **Read the reference implementation carefully** - make_TUEV.py clearly shows event extraction
2. **Question the extreme imbalance** - 99.5% one class should have been a red flag
3. **Verify our assumptions** - "288" needed investigation
4. **Check input dimensions** - 23×1000 vs our 20×1024 mismatch

### We Assumed:
1. TUEV is like TUAB (it's not)
2. Sliding windows are always correct (they're not)
3. The paper describes everything (it doesn't)
4. High F1 with moderate BAC is normal (it indicates balanced data)

## Lessons Learned

### 1. Always Check Reference Code First
- Papers omit critical details
- Code reveals the actual implementation
- Our 2-day debugging could have been 2-hour fix

### 2. Question Extreme Imbalances
- 99.5% one class is almost never the intended distribution
- If you see this, you're probably doing something wrong

### 3. Different Tasks Need Different Pipelines
- TUAB: Abnormality detection on full recordings
- TUEV: Event classification on pre-extracted segments
- TUSZ: Temporal seizure detection (yet another paradigm)

### 4. Input Dimensions Are Sacred
- If the paper says 23×1000, use exactly that
- Don't assume you can change dimensions without consequences

## The Fix: Two Options

### Option 1: Match EEGPT Exactly (Recommended)
1. Pre-extract 5-second segments around annotated events at 200 Hz (tmin=-2s, tmax=+3s around event).
2. Use the TUH referential “-REF” channel set with the reference ordering; do not convert to bipolar (reference conversion is present but commented out).
3. Keep unweighted CrossEntropy with label_smoothing=0.1; no explicit class balancing required when using event-only segments and subject-level splits.
4. Store segments in a safe cache format (prefer `.pt` tensors with torch.save) alongside a META/index; exact pickles are not required for parity.
5. Train with warmup and layer decay as in the reference (e.g., warmup_epochs≈5, layer_decay≈0.9, batch_size 64–100), targeting 20×1000 model input (with a 23→20 learned channel conv if needed).

### Option 2: Redefine the Problem
1. Acknowledge we're doing temporal event detection (harder)
2. Use appropriate metrics (not classification metrics)
3. Expect much lower performance
4. Consider this research, not replication

## Conclusion

We built a **temporal event detection system** when we needed an **event segment classifier**. This fundamental misunderstanding cascaded through our entire implementation, creating a system that could never achieve the paper's results. The lesson: **Always verify your problem definition before implementing the solution.**

---

**Status**: Root cause identified
**Impact**: Current implementation is solving the wrong problem
**Fix Required**: Complete reimplementation of data pipeline
**Time Wasted**: ~2 weeks
**Time to Fix**: ~2-3 days with correct approach

---

Corrections and Paper‑Parity Checklist (Actionable)

What was inaccurate or needed refinement above (corrected here):
- Bipolar montage: not required for paper parity. The reference `make_TUEV.py` includes a `convert_signals` (bipolar) function but it is commented out; training is performed on referential “-REF” channels reordered to a standard list.
- Class balancing: the reference does not perform explicit per-class balancing. They construct event-only datasets and split by subject; loss is unweighted CrossEntropy with smoothing=0.1.
- Input shape: the classifier head operates on 20×1000 (20 channels × 1000 samples at 200 Hz). The pipeline ingests 23 REF channels and uses a learned channel conv to reach 20 (see `run_class_finetuning_EEGPT_change_tuev.py`, `use_channels_names` = 20, and `use_chan_conv=True`).
- File format: pickles are not essential. Our repo standards encourage safe `.pt` with metadata. Paper fidelity is about content/shape, not serialization format.
- Our imbalance measurement: precisely 179,444/180,205 (99.58%) windows labeled background in train cache.

Paper‑Parity Implementation Plan (within our architecture):
- Preprocessing (src)
  - Add an event‑segment builder that:
    - Reads EDF + `.rec`/`.lab`, filters (0.1–75 Hz), notches 50 Hz, resamples to 200 Hz.
    - Reorders to the TUH “-REF” channels expected by the reference (23 inputs).
    - Extracts 5 s segments centered on events (−2 s to +3 s).
    - Saves tensors as `(C=23, T=1000)` float32 in Volts (SI units) with META/index.
- Dataset (src)
  - `TUEVEventSegmentDataset` returning `(x: (23, 1000), y: Long)`; optional mode to apply a 23→20 learned channel conv consistent with reference.
- Training (experiments, thin)
  - `experiments/eegpt_linear_probe/train_tuev_segments.py` that:
    - Uses unweighted CrossEntropy with label_smoothing=0.1.
    - Adds warmup (≈5 epochs) and layer decay (≈0.9) as in reference.
    - Uses batch_size 64–100; eval metrics: BAC (primary), weighted F1, Cohen’s kappa.
    - Targets model input 20×1000 via learned channel conv from 23 inputs.
- Splits
  - Create subject‑level splits analogous to `processed_{train,eval,test}` in the reference (train subjects → split 80/20 into train/eval; test from eval folder).

Acceptance criteria for paper parity:
- Segment cache validated: `(23, 1000)`, sr=200, filters 0.1–75 Hz, notch 50 Hz, REF channels ordered as in reference.
- Training reaches BAC ≈ 0.62 ± 0.02 on eval with smoothing=0.1, unweighted loss, warmup + layer_decay.

If instead we keep temporal detection (current pipeline):
- Switch to detection‑appropriate training: WeightedRandomSampler and/or focal loss; consider hard‑negative mining and event‑centric sampling; evaluate with detection metrics and event matching, not only per‑window BAC.
