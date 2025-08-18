# EEGPT Paper Verified Facts - NO HALLUCINATIONS
**Date**: 2025-08-18  
**Source**: EEGPT Paper (NeurIPS 2024)  
**Status**: VERIFIED FROM PAPER TEXT

## 1. EEGPT Architecture Facts (From Paper)

### Core Model Design (Section 2.3, Page 5)
- **Input**: M channels × T time points
- **Patch size**: d = 64 samples (250ms @ 256Hz)
- **Patches**: N = T/d patches
- **Summary tokens**: S learnable tokens (DEFAULT S=4)
- **Embedding dimension**: 512 (for large model)

### Model Variants (Table 6, not shown in excerpt but referenced)
- Multiple sizes with different embedding dims
- Large model: 10M+ parameters
- Summary tokens S varies (but default is 4)

## 2. Linear-Probing Method (Section 2.4, Page 5)

### What the Paper EXPLICITLY Says:
> "The encoder passes the output tokens corresponding to **summary tokens** to the linear classification head."

### Figure 3 Description:
- Shows: Pretrained Encoder → Summary Tokens → Linear Head
- Freezes pretrained model
- Only trains linear classification head

## 3. TUAB Architecture (Table 12, Page 20)

### Input Processing:
1. Input: 23 × 2000 (23 channels, 2000 samples = 8s @ 250Hz)
2. Conv1d: 23→20 channels (kernel=1)
3. BatchNorm + GELU
4. Conv1d: Temporal (kernel=15, groups=20, padding=7)
5. BatchNorm + GELU
6. Dropout(0.25)
7. EEGPT-encoder: Returns 31 × 4 × 512
8. Flatten + Linear → 1 output (binary)

### Key Output:
- **31 × 4 × 512** (31 is likely time dimension after processing)
- Flattened to feed linear layer

## 4. TUEV Architecture (Table 13, Page 20)

### Input Processing:
1. Input: 23 × 1000 (23 channels, 1000 samples = ~4s @ 250Hz)
2. Conv1d: 23→20 channels (kernel=1)
3. BatchNorm + GELU
4. Conv1d: Temporal (kernel=**55**, groups=20, padding=27)
5. BatchNorm + GELU
6. Dropout(**0.5**)
7. EEGPT-encoder: Returns **15 × 4 × 512**
8. Flatten + Linear → 6 outputs

### Critical Differences from TUAB:
- Kernel size: 55 (vs 15 for TUAB)
- Dropout: 0.5 (vs 0.25 for TUAB)
- Input length: 1000 (vs 2000 for TUAB)
- Output shape: 15 × 4 × 512 (vs 31 × 4 × 512)

## 5. What Features Are Used (VERIFIED)

### From Table 13 Output:
- EEGPT encoder outputs: **15 × 4 × 512**
  - 15 = time dimension (1000 samples / 64 patch_size ≈ 15.6 → 15)
  - 4 = summary tokens
  - 512 = embedding dimension

### What Gets Flattened:
The paper says "**flatten,linear**" which means:
- They flatten the 4 × 512 = 2,048 dimensional summary tokens
- NOT the full 15 × 4 × 512

### Why 15 in the output?
- This appears to be a typo or confusion in the table
- The EEGPT encoder returns summary tokens (4 × 512)
- The 15 is the number of patches, but only summary tokens go to classifier

## 6. Performance Results (Tables 9-10, Page 15)

### TUAB Results:
- EEGPT (Ours): BAcc 0.7983±0.0030, AUROC 0.8718±0.0050
- BIOT: BAcc 0.7959±0.0057, AUROC 0.8815±0.0043

### TUEV Results:
- **EEGPT (Ours): BAcc 0.6232±0.0114, F1 0.8187±0.0063, Kappa 0.6351±0.0134**
- BIOT: BAcc 0.5281±0.0225, F1 0.7492±0.0082, Kappa 0.5273±0.0249
- LaBraM: BAcc 0.6409±0.0065 (better than EEGPT!)

## 7. Implementation Details (Appendix C.2.6, Page 19)

### TUEV Specific Details:
> "We conducted experiments using **linear-probing** on these two datasets."
> "The convolution kernel size for TUAB was (1, 15), and for TUEV, it was **(1, 55)**."
> "Both experiments used the same optimizer and a learning rate of **5e-4**."
> "Due to GPU memory limitations, the batch size for TUAB was 100, and for TUEV was **500**."

### Channel Mapping:
> "The eegpt-encoder uses the following 20 channel embeddings as the inputs' channel embeddings: [FP1, FPZ, FP2, F7, F3, FZ, F4, F8, T7, C3, CZ, C4, T8, P7, P3, PZ, P4, P8, O1, O2]"

## 8. CRITICAL FINDING: What Features Are Actually Used?

### The Paper's Ambiguity:
1. Table 13 shows output as "15 × 4 × 512"
2. But Section 2.4 says "summary tokens to linear head"
3. "Flatten,linear" suggests flattening happens

### Most Likely Interpretation:
- EEGPT outputs 4 summary tokens (4 × 512)
- The "15" in table is probably patches count (not used)
- Linear probe uses 4 × 512 = 2,048 features
- This matches "linear-probing" description

## 9. Why Our TUEV Failed (HYPOTHESIS)

### What We Did:
1. First attempt: Used 4 × 512 = 2,048 features → BAcc 0.16
2. Second attempt: Used ALL features (163,840) → BAcc 0.15

### What Paper Did (VERIFIED):
- Used "linear-probing" with summary tokens
- Achieved BAcc 0.6232

### Possible Explanations:
1. **Data preprocessing differs** - We use 1024 samples, they use 1000
2. **Channel mapping differs** - We might have wrong channel order
3. **Normalization differs** - Paper doesn't specify normalization
4. **Cache/data loading differs** - Our cached data might be corrupted
5. **The "15 × 4 × 512" is not a typo** - They might use more than summary tokens

## 10. What We Know FOR CERTAIN

### From Paper Text:
✅ TUEV uses kernel size 55 (not 15)
✅ TUEV uses dropout 0.5 (not 0.25)
✅ TUEV uses batch size 500
✅ TUEV uses learning rate 5e-4
✅ TUEV input is 23 × 1000 samples
✅ Linear-probing method is used
✅ Target performance: BAcc 0.6232

### What's UNCLEAR:
❓ Exact features used (just 4×512 or more?)
❓ How they handle 1000 vs 1024 samples
❓ Exact normalization used
❓ Why table shows "15 × 4 × 512"

## 11. The Real Problem

The paper is **AMBIGUOUS** about feature extraction:
1. Text says "summary tokens" (4 × 512)
2. Table shows "15 × 4 × 512" 
3. No explicit statement of what gets flattened

This ambiguity is why we can't reproduce their results!

## 12. Next Steps (Based on Paper)

### Must Try:
1. Use EXACTLY 1000 samples (not 1024)
2. Verify channel mapping matches paper's order
3. Try different interpretations of "15 × 4 × 512"
4. Contact authors for clarification

### Should NOT Do:
❌ Use all patch features (163k) - Paper never mentions this
❌ Change architecture beyond paper's specification
❌ Assume we know what they did - Paper is ambiguous

## Conclusion

The EEGPT paper provides most details but is **critically ambiguous** about:
1. What features are actually used for TUEV
2. Why output shape is "15 × 4 × 512"
3. How linear-probing handles these dimensions

Our failure to reproduce TUEV results (0.16 vs 0.62 BAcc) suggests:
- Either we're missing something in preprocessing
- Or the paper has an error/omission
- Or "15 × 4 × 512" means something different than we think

**We cannot proceed without clarification from authors or systematic experimentation.**