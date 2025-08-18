# TUEV - EXACT QUOTES FROM EEGPT PAPER WITH LINE NUMBERS

## SOURCE OF TRUTH: EEGPT.md lines verified on 2024-08-18

### 1. INPUT SIZE - PAPER HAS INTERNAL CONTRADICTION

**Line 585 says:**
> "The EEG signals contain 23 channels at 256 Hz and are segmented into 112,491 5-second samples."

**BUT Table 13 (Line 606) says:**
> `| 23 × 1000    | conv1d         | 1      | 1      | 1      | 0       |`

**MATH CHECK:**
- 1000 samples ÷ 256 Hz = 3.90625 seconds
- NOT 5 seconds!
- **THE PAPER CONTRADICTS ITSELF**

### 2. ARCHITECTURE - TABLE 13 (Lines 604-613)

```
Line 604: | Input Size   | Operator       | kernel | stride | groups | padding |
Line 605: |--------------|----------------|--------|--------|--------|---------|
Line 606: | 23 × 1000    | conv1d         | 1      | 1      | 1      | 0       |
Line 607: | 20 × 1000    | batchnorm,gelu | -      | -      | -      | -       |
Line 608: | 20 × 1000    | conv1d         | 55     | 1      | 20     | 27      |
Line 609: | 20 × 1000    | batchnorm,gelu | -      | -      | -      | -       |
Line 610: | 20 × 1000    | dropout(0.5)   | -      | -      | -      | -       |
Line 611: | 20 × 1000    | eegpt-encoder  | 64     | 64     | -      | -       |
Line 612: | 15 × 4 × 512 | flatten,linear | -      | -      | -      | -       |
Line 613: | 6            | output         | -      | -      | -      | -       |
```

**FACTS FROM TABLE:**
- Input: 23 × 1000 (NOT 1280)
- First conv: 23 → 20 channels (REDUCTION)
- Dropout: 0.5 (NOT 0.25)
- Output shape: 15 × 4 × 512
- Final: 6 classes

### 3. CHANNEL NAMES (Line 615)

**Exact quote:**
> "The 23-channel input is first to reduce the number of channels to 20 by the convolution. Then, the eegpt-encoder uses the following 20 channel embeddings as the inputs' channel embeddings: [FP1, FPZ, FP2, F7, F3, FZ, F4, F8, T7, C3, CZ, C4, T8, P7, P3, PZ, P4, P8, O1, O2]."

### 4. OPTIMIZER AND BATCH SIZE (Line 587)

**Exact quote:**
> "The convolution kernel size for TUAB was (1, 15), and for TUEV, it was (1, 55). Both experiments used the same optimizer and a learning rate of 5e-4. Due to GPU memory limitations, the batch size for TUAB was 100, and for TUEV, it was 500."

**FACTS:**
- Kernel: (1, 55) for TUEV
- Learning rate: 5e-4
- Batch size: 500
- "Same optimizer" = AdamW (mentioned elsewhere for TUAB)

### 5. DATA SPLIT STRATEGY (Line 197)

**Exact quote:**
> "For the data splitting of TUAB and TUEV, we strictly follow the same strategy as BIOT"

### 6. EXPERIMENT REPETITION (Line 197)

**Exact quote:**
> "To ensure the reliability of the experiments, we repeated each experiment three times and calculated the standard deviation."

### 7. SUBJECT COUNT (Line 183)

**From Table 1:**
> `| TUEV       | Event     | 288      | 6       |`

### 8. PERFORMANCE METRICS (Lines 230-231)

**From Table 3:**
```
Line 230: | Ours          | 25M        | 0.6232±0.0114     | 0.8187±0.0063 | 0.6351±0.0134 |
```

---

## WHAT THE PAPER GOT WRONG/UNCLEAR:

1. **CONTRADICTION**: Says "5-second samples" but uses 1000 samples (3.9s)
2. **NO MENTION** of how to handle this discrepancy
3. **NO MENTION** of whether windows are sliding or non-overlapping
4. **NO MENTION** of OneCycle schedule for TUEV (unlike other tasks)

## WHAT WE KNOW FOR CERTAIN:

1. ✅ Input is 23 × 1000 (from Table 13)
2. ✅ Reduces to 20 channels (from Table 13)
3. ✅ Dropout is 0.5 (from Table 13)
4. ✅ Kernel is 55 with padding 27 (from Table 13)
5. ✅ Batch size is 500 (from text)
6. ✅ Learning rate is 5e-4 (from text)
7. ✅ Use BIOT split strategy (from text)
8. ✅ Repeat 3 times (from text)

---

**THIS IS THE TRUTH FROM THE PAPER. THE PAPER ITSELF HAS ERRORS/CONTRADICTIONS.**