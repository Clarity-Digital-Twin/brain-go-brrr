# TUEV Implementation Divergence Analysis: How We Got It Wrong

## Executive Summary

We fundamentally misunderstood the TUEV task. We implemented **continuous event detection** (sliding windows over full recordings) while EEGPT implemented **event segment classification** (pre-extracted segments around known events). This is a completely different problem.

Status check (current): the core conclusion above is correct and the divergence has been remediated in code. We now implement event‑only extraction and a parity‑ready training path in `src/`, with a single thin trainer in `experiments/`.

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

Notes verified in the reference repo (paths below):
- Filtering: 0.1–75 Hz, notch 50 Hz, resample 200 Hz (readEDF).
- Channels: TUH “-REF” names; training uses referential channels; optional bipolar conversion is present but commented out in the maker.
- Model input: a learned channel conv maps 23 referential inputs to 20 classifier channels. The classifier operates on 20×1000 (channels×samples).

### What We Implemented (current repo):
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

### Missing Critical Details (now verified in reference code):
1. Paper never mentions they only extract event segments
2. Paper never requires bipolar montage conversion (reference code ships a conversion function but does not use it in dataset maker)
3. Paper never explains "288" (subjects? segments? files?)
4. Paper shows high F1 (81.87%) with moderate BAC (62.32%); reference training confirms event-only classification, not temporal detection

## Why Our Training Failed Catastrophically

### The Numbers Tell the Story:
- Our dataset (measured): 180,205 windows in train cache, with 179,444 labeled as background (99.58%). Source: `data/cache/tuev_23ch_paper_parity/train/index_train_mne-ar-v4.json` → `class_counts`.
- Reference dataset: event-only pickled segments, each `(23, 1000)` at 200 Hz, stored under `processed_{train,eval,test}`; they do not generate sliding windows.
- Our result: Model learns to predict only background (16.67% BAC) due to extreme imbalance.
- Reference result: Model learns all classes (≈62.32% BAC) on event-only data.

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

## The Fix: Paper Parity (Implemented)

### DEFINITIVE ANSWERS TO ALL QUESTIONS:

#### Q1: .rec/.lab Parser - EXISTS AND WORKS ✅
**Answer**: We ALREADY have a working `.lab` parser in `tuev_dataset.py:_load_annotations()`
```python
# Lines 316-323 in tuev_dataset.py - WORKS PERFECTLY
start_us = float(parts[0])  # microseconds from .lab
end_us = float(parts[1])
label = parts[2].lower()
start_sec = start_us / 1e6  # Convert to seconds
end_sec = end_us / 1e6
```
**Action**: Modify to extract 5s segments instead of sliding windows. Parser itself is fine.

#### Q2: Subject-Level Splits - 80/20 RANDOM ✅
**Answer**: Reference uses simple 80/20 random split at subject level:
```python
# Lines 224-226 in make_TUEV.py
val_sub = np.random.choice(train_sub, size=int(len(train_sub) * 0.2), replace=False)
train_sub = list(set(train_sub) - set(val_sub))
```
**Action**: Extract subject ID from filename, do 80/20 split with fixed seed.

#### Q3: Single GPU Training - ADJUST BATCH SIZE ✅
**Answer**: Reference uses 2 GPUs with batch_size=400 total (200 per GPU):
```bash
# Line 11: GPUS_PER_NODE=${GPUS_PER_NODE:-2}
# Line 24: --batch_size 400
```
**Action**: For single GPU, use batch_size=64-100 to fit memory. Accumulate gradients if needed.

#### Q4: Fallback Path - NOT NEEDED ❌
**Answer**: Default trainer supports both strict parity and a compatibility fallback:
- Strict parity (native 1000) via `--use_parity` (EEGPT configured with `time_steps=1000`, `patch_stride=64`).
- Compatibility fallback pads 1000→1024 for standard EEGPT checkpoints (kept for continuity while parity is validated).

### Parity Path: Match EEGPT Exactly (Implemented)
1. Extract 5s event‑centered segments at 200Hz (−2..+3s) → (23,1000) float32 V.
2. Use "-REF" channels; no bipolar conversion.
3. Learned 23→20 channel mapper (spatial conv + depthwise temporal conv).
4. Unweighted CrossEntropy with label_smoothing=0.1.
5. Warmup=5 epochs + cosine anneal; layer_decay=0.65; lr=5e-4; wd=0.05; epochs≈30; effective batch≈400 via grad accumulation.
6. Subject‑level 80/20 split (seeded) fallback when pre‑split dirs not present.
7. EEGPT encoder supports native 1000 via patch stride (or 1000→1024 padding for compatibility).

Key locations:
- Extractor: `src/brain_go_brrr/infra/preprocessing/tuev_event_extractor.py`
- Dataset (+annotation parser for `_ch000.lab`): `src/brain_go_brrr/infra/data/tuev_event_dataset.py`
- Mapper (23→20): `src/brain_go_brrr/infra/ml_models/channel_mapper.py`
- EEGPT parity support (time_steps/patch_stride): `src/brain_go_brrr/infra/ml_models/eegpt_architecture.py`
- Trainer (single): `experiments/eegpt_linear_probe/train_tuev_events.py` (`--use_parity` flag)

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
- Bipolar montage: not required. The reference `make_TUEV.py` ships a `convert_signals` (bipolar) helper but it is commented out; training uses referential “-REF” channels.
- 23→20 mapping: learned conv present in `EEGPT_mcae_finetune_change_tuev.py` and enabled in `run_class_finetuning_EEGPT_change_tuev.py` with `use_chan_conv=True` and `img_size=[20, 1000]`.
- Segments and rate: event-only `(23, 1000)` segments at 200 Hz (5 s) built by `make_TUEV.py` (`BuildEvents`, `readEDF`).
- Loss: unweighted, `LabelSmoothingCrossEntropy(smoothing=0.1)`.
- Warmup/layer decay: `warmup_epochs=5`, `layer_decay=0.65` per `finetune_TUEV_EEGPT.sh` (defaults in args are 0.9 layer_decay but training script sets 0.65).
- Batch/Epochs: `batch_size=400` (distributed), `epochs=30` per `finetune_TUEV_EEGPT.sh`.
- File format: reference uses pickle; we will store `.pt` tensors with META/index, maintaining shapes/semantics.
- Our imbalance: precisely 179,444/180,205 (99.58%) background windows in train cache.

Paper‑Parity Implementation Plan (within our architecture):
- Preprocessing (src)
  - Add an event‑segment builder that:
    - Reads EDF + `.rec`/`.lab`, filters (0.1–75 Hz), notch 50 Hz, resample 200 Hz.
    - Reorders to reference “-REF” channels (23 inputs).
    - Extracts 5 s segments centered on events (−2 s to +3 s).
    - Saves `(23, 1000)` float32 Volts with META/index.
- Dataset (src)
  - `TUEVEventSegmentDataset` returning `(x: (23, 1000), y: Long)`; no sliding windows.
- Training (experiments, thin)
  - `experiments/eegpt_linear_probe/train_tuev_events.py` wrapping src components; loss `LabelSmoothingCrossEntropy(0.1)`; warmup 5, layer_decay 0.65; batch≈400 effective; metrics: BAC (primary), weighted F1, kappa; input `20×1000` via 23→20 mapper.
- Splits
  - Subject‑level: mirror `processed_{train,eval,test}` in the reference.

Acceptance criteria for paper parity (unchanged):
- Segment cache validated: `(23,1000)`, sr=200, filters 0.1–75 Hz, notch 50 Hz, REF channels in reference order.
- Training reaches BAC ≈ 0.62 ± 0.02 on eval with smoothing=0.1, unweighted loss, warmup + layer_decay.

Verification commands:
- Inspect parser for TUEV annotations:
  - `sed -n '77,86p' src/brain_go_brrr/infra/data/tuev_event_dataset.py`
- Check parity stride support:
  - `rg -n "patch_stride|time_steps" src/brain_go_brrr/infra/ml_models/eegpt_architecture.py`
- Train (compatibility fallback):
  - `python experiments/eegpt_linear_probe/train_tuev_events.py --data_dir data/datasets/tuev --eegpt_checkpoint <ckpt> --epochs 1 --batch_size 32`
- Train (strict parity, native 1000):
  - `python experiments/eegpt_linear_probe/train_tuev_events.py --data_dir data/datasets/tuev --eegpt_checkpoint <ckpt> --use_parity --epochs 1 --batch_size 32`

If instead we keep temporal detection (current pipeline):
- Switch to detection‑appropriate training: WeightedRandomSampler and/or focal loss; consider hard‑negative mining and event‑centric sampling; evaluate with detection metrics and event matching, not only per‑window BAC.
