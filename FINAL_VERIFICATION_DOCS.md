# Final Verification Documents - EEGPT TUEV Implementation

## Summary of Our Discovery

**THE KEY FINDING**: The reference EEGPT implementation HARDCODES `LinearWithConstraint(30720, num_classes)` at line 769 of `EEGPT_mcae_finetune_change_tuev.py`. This proves definitively that TUEV uses **30,720 features** (15×4×512), NOT 153,600!

## Understanding Confirmed

1. **EEGPT processes each temporal patch separately**:
   - Input: 20 × 1000 (after channel adapter)
   - Creates 15 temporal patches (1000/64 = 15.625 → 15)
   - Each patch gets 4 summary tokens
   - Total: 15 × 4 × 512 = 30,720 features

2. **Reference implementation flow** (lines 532-558):
   ```python
   x = x.flatten(0, 1)  # (B*15, 20, 512) - process each temporal position
   summary_token = self.summary_token.repeat((x.shape[0], 1, 1))  # Add to EACH
   x = torch.cat([x, summary_token], dim=1)  # (B*15, 24, 512)
   # After transformer:
   x = x[:, -summary_token.shape[1]:, :]  # Extract 4 from EACH
   x = x.reshape((B, N, self.embed_num, -1))  # (B, 15, 4, 512)
   ```

3. **Classifier** (line 843):
   ```python
   x = x.flatten(1)  # (B, 30720)
   x = self.head(x)  # LinearWithConstraint(30720, 6)
   ```

## Documents Ready for External Verification

### 1. Core Architecture Documents
- **`EEGPT_TUEV_FIX.md`** - Explains the bug and solution (✅ Updated to 30,720)
- **`IMPLEMENTATION_PLAN.md`** - Step-by-step fix plan (✅ Already correct)
- **`experiments/eegpt_linear_probe/CRITICAL_DISCOVERY.md`** - Detailed architecture analysis

### 2. Specification Documents
- **`TUEV_UNIFIED_SPECS.md`** - Single source of truth (✅ Updated to 30,720)
- **`TUEV_QUICK_START.md`** - Quick reference guide (✅ Already correct)
- **`PROJECT_STATUS.md`** - Overall project status (✅ Already correct)

### 3. Paper Reference
- **`literature/markdown/EEGPT/EEGPT.md`** - Original paper in markdown
  - Line 164: "encoder passes output tokens corresponding to summary tokens"
  - Table 13: Shows "15 × 4 × 512" output shape
  - Section 2.4: Confirms summary tokens are used

### 4. Reference Implementation Evidence
- **`reference_repos/EEGPT/downstream_tueg/Modules/models/EEGPT_mcae_finetune_change_tuev.py`**
  - Line 769: `LinearWithConstraint(30720, num_classes)` - HARDCODED PROOF
  - Lines 532-558: Shows temporal processing mechanism
  - Line 843: Shows flattening to 30,720 features

## Key Specifications Confirmed

| Specification | Value | Evidence |
|--------------|-------|----------|
| Input shape | 23 × 1000 | Table 13, line 606 |
| After channel adapter | 20 × 1000 | Table 13, line 607 |
| Temporal patches | 15 | 1000/64 = 15.625 → 15 |
| Summary tokens per patch | 4 | S=4 in paper |
| Output shape | 15 × 4 × 512 | Table 13, line 612 |
| **Total features** | **30,720** | Reference line 769 |
| Temporal kernel | 55 | Table 13, line 608 |
| Dropout | 0.5 | Table 13, line 610 |
| Batch size | 500 | Appendix C.2.6 |
| Learning rate | 5e-4 | Appendix C.2.6 |
| Classes | 6 | SPSW, GPED, PLED, EYEM, ARTF, BCKG |

## Verification Checklist

✅ **Paper Table 13 shows "15 × 4 × 512"**
✅ **Paper text says "encoder passes output tokens corresponding to summary tokens"**
✅ **Reference code hardcodes 30,720 in classifier**
✅ **Reference processes each temporal position separately**
✅ **All docs now show 30,720 features (15×4×512)**
✅ **NO docs claim 153,600 features anymore**

## What External Agents Should Verify

1. **Feature count**: Confirm 30,720 features (15×4×512) NOT 153,600
2. **Processing**: Each temporal patch gets its own 4 summary tokens
3. **Architecture**: Temporal structure preserved, not collapsed
4. **Reference alignment**: Our understanding matches the reference implementation

## Confidence Level

**100% CERTAIN** - The reference implementation is unambiguous:
- Hardcoded `LinearWithConstraint(30720, num_classes)`
- Clear reshape operations showing (B, 15, 4, 512)
- Explicit temporal processing in forward pass

This is NOT interpretation - it's reading the actual code that works!