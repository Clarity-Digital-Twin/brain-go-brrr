# TUEV Unified Specifications - Single Source of Truth

## Labels Used
- **[Paper]**: Direct quote from EEGPT.md with line number
- **[Local]**: Our actual downloaded data at v2.0.1
- **[Decision]**: What we will implement

---

## 1. Dataset Specifications

### Window Size - THE CORE ISSUE
- **[Paper Text]** Line 585: "112,491 5-second samples"
- **[Paper Table 13]** Line 606: Input is "23 × 1000"
- **[Math]**: 1000 ÷ 256 Hz = 3.90625 seconds (NOT 5!)
- **[Decision]**: USE TABLE 13 (1000 samples) - it's the implementation

### Subject Count
- **[Paper]** Line 183, Table 1: "288" subjects
- **[Local]** Our v2.0.1: 370 subjects (290 train, 80 eval)
- **[Decision]**: Use our 370 subjects (more data is better)

### Classes
- **[Paper]** Line 575: 6 classes (SPSW, GPED, PLED, EYEM, ARTF, BCKG)
- **[Local]** Confirmed in .lab files
- **[Decision]**: 6 classes ✓

### Channels
- **[Paper]** Line 585: "23 channels at 256 Hz"
- **[Local]** Our files: 26-27 channels at 250 Hz
- **[Decision]**: Resample 250→256 Hz, select 23 channels

---

## 2. Model Architecture - FROM TABLE 13

**[Paper]** Lines 604-613, Table 13 shows EXACTLY:

```
Line 606: | 23 × 1000    | conv1d         | 1      | 1      | 1      | 0       |
Line 607: | 20 × 1000    | batchnorm,gelu | -      | -      | -      | -       |
Line 608: | 20 × 1000    | conv1d         | 55     | 1      | 20     | 27      |
Line 609: | 20 × 1000    | batchnorm,gelu | -      | -      | -      | -       |
Line 610: | 20 × 1000    | dropout(0.5)   | -      | -      | -      | -       |
Line 611: | 20 × 1000    | eegpt-encoder  | 64     | 64     | -      | -       |
Line 612: | 15 × 4 × 512 | flatten,linear | -      | -      | -      | -       |
          |              | [Paper] This denotes 15 temporal patches × S=4 summary tokens × 512 dims = 30,720 features (EEGPT Table 13; §2.4)
          |              | [Paper Line 164: "encoder passes output tokens corresponding to summary tokens"]
          |              | [Reference Code Line 769: LinearWithConstraint(30720, num_classes) - HARDCODED PROOF!]
          |              | [Our bug: only extracting LAST 4 tokens = 2,048 features]
Line 613: | 6            | output         | -      | -      | -      | -       |
```

### Key Parameters
- **[Paper]** Input: 23 × 1000
- **[Paper]** Channel reduction: 23 → 20
- **[Paper]** Temporal kernel: 55, padding: 27, groups: 20
- **[Paper]** Dropout: 0.5
- **[Paper]** Output shape: 15 × 4 × 512 (Table 13)
- **[Paper]** Line 164: "encoder passes output tokens corresponding to summary tokens" (§2.4)
- **[Reference]** Line 769 in EEGPT_mcae_finetune_change_tuev.py: LinearWithConstraint(30720, num_classes)
- **[Correct]**: 15 temporal patches × 4 summary tokens × 512 = 30,720 features
- **[Current Bug]**: Only extracting last 4 tokens = 2,048 features
- **[Decision]**: Process each temporal patch separately, get 4 summary tokens per patch

### Channel Names
**[Paper]** Line 615: The 20 channels are:
"FP1, FPZ, FP2, F7, F3, FZ, F4, F8, T7, C3, CZ, C4, T8, P7, P3, PZ, P4, P8, O1, O2"

---

## 3. Training Configuration

### Batch Size & Learning Rate
- **[Paper]** Line 587: "batch size for TUAB was 100, and for TUEV, it was 500"
- **[Paper]** Line 587: "learning rate of 5e-4"
- **[Paper]** Line 587: "Both experiments used the same optimizer" (name not specified)
- **[Decision]**: Batch 500, LR 5e-4, AdamW (inferred from pretraining), constant schedule (not stated in paper)

### Data Split
- **[Paper]** Line 197: "For the data splitting of TUAB and TUEV, we strictly follow the same strategy as BIOT"
- **[Local]** We have: 290 train, 80 eval subjects
- **[Decision]**: Use existing train/eval split (BIOT strategy)

### Experiment Protocol
- **[Paper]** Line 197: "we repeated each experiment three times and calculated the standard deviation"
- **[Decision]**: Run 3 times with different seeds

---

## 4. Performance Targets

**[Paper]** Line 230, Table 3:
- Balanced Accuracy: 0.6232 ± 0.0114
- Weighted F1: 0.8187 ± 0.0063
- Cohen's Kappa: 0.6351 ± 0.0134

---

## 5. Local Data Reality

### What We Have
- **[Local]** Path: `/data/datasets/external/tuh_eeg/TUEV/v2.0.1/`
- **[Local]** Files: 518 EDFs, 11,396 .lab files
- **[Local]** Subjects: 370 (290 train, 80 eval)
- **[Local]** Sampling: 250 Hz (needs resampling)
- **[Local]** Channels: 26-27 (need selection)

### Required Preprocessing
1. Resample 250 Hz → 256 Hz
2. Select 23 channels (TCP montage)
3. Extract windows from .lab annotations
4. Crop/pad to exactly 1000 samples
5. Map to 20 standard channels

---

## 6. Implementation Checklist

```python
# These values are CONFIRMED from paper
assert input_shape == (batch, 23, 1000), "Table 13, line 606"
assert channels_after_conv1 == 20, "Table 13, line 607"
assert temporal_kernel == 55, "Table 13, line 608"
assert temporal_padding == 27, "Table 13, line 608"
assert temporal_groups == 20, "Table 13, line 608"
assert dropout_rate == 0.5, "Table 13, line 610"
assert output_shape == (batch, 15, 4, 512), "15 temporal × 4 summary tokens × 512"
# Paper Table 13 shows this shape, paper text confirms summary tokens are used
assert n_classes == 6, "Table 13, line 613"
assert batch_size == 500, "Paper line 587"
assert learning_rate == 5e-4, "Paper line 587"
```

---

## 7. Contradictions Resolved

### Paper's 5s vs 1000 samples
- **Problem**: Paper text says 5s, Table shows 1000 samples
- **Analysis**: Both TUAB and TUEV use ~78% of claimed window
- **Resolution**: Table 13 is implementation truth - use 1000

### Version Number
- **Problem**: Scripts say v1.0.1, status says v2.0.1
- **Resolution**: We have v2.0.1 - update all references

---

**THIS DOCUMENT IS THE SINGLE SOURCE OF TRUTH FOR TUEV IMPLEMENTATION**