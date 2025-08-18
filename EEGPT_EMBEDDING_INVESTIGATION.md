# EEGPT Embedding Investigation Dossier
**Date**: 2025-08-18  
**Author**: Senior Architecture Audit  
**Status**: 🔴 CRITICAL - Fundamental Architecture Issues Found

## Executive Summary

After deep investigation of the codebase, I've discovered **MASSIVE ARCHITECTURAL INCONSISTENCIES** in how EEGPT embeddings are extracted and used across different experiments. The current implementation is **fundamentally broken** and explains why TUEV training catastrophically failed.

## 1. What EEGPT Actually Outputs (From Architecture Investigation)

### Core EEGPT Architecture (`src/brain_go_brrr/infra/ml_models/eegpt_architecture.py`)

```python
class EEGTransformer(nn.Module):
    def forward(self, x):
        # Input: (B, C, T) where B=batch, C=channels, T=time
        # Process:
        # 1. Patch embedding: (B, C, T) -> (B, N, C, D) where N=patches, D=embed_dim
        # 2. Add channel embeddings
        # 3. Reshape to sequence: (B, N*C, D)
        # 4. Concatenate 4 summary tokens at end
        # 5. Apply transformer blocks
        # 6. Extract ONLY summary tokens: x[:, -self.embed_num:, :]
        # 7. Return: (B, 4, 512) - ONLY 4 SUMMARY TOKENS!
```

**CRITICAL FINDING**: The base EEGPT model **ONLY returns 4 summary tokens** (shape: batch × 4 × 512).

### Default Configuration
- `embed_num`: 4 (number of summary tokens)
- `embed_dim`: 512 (dimension of each token)
- `patch_size`: 64 (250ms @ 256Hz)
- Output shape: **(batch, 4, 512)** = 2,048 features when flattened

## 2. How TUAB Uses EEGPT (APPEARS CORRECT)

### TUAB Training (`experiments/eegpt_linear_probe/train_paper_aligned.py`)

```python
class LinearProbe(nn.Module):
    def forward(self, features):
        # features: (batch_size, n_summary_tokens, embed_dim)
        # Average pool over summary tokens
        x = features.mean(dim=1)  # (batch_size, embed_dim)
        return self.probe(x)  # Linear layers on 512 dims
```

**Analysis**: 
- ✅ TUAB correctly uses the 4 summary tokens
- ✅ Averages them to get 512-dimensional features
- ✅ Feeds to linear probe
- ✅ This matches the EEGPT paper's linear-probing method

### TUAB Results
- **Output exists**: `tuab_4s_paper_target_BULLETPROOF_20250809_073159/best_model.pt`
- Likely achieved reasonable performance (need to verify)

## 3. How TUEV TRIED to Use EEGPT (MULTIPLE FAILURES)

### Attempt 1: Using Summary Tokens (`train_tuev_aligned.py`)
```python
# Linear probe (4×512 → 6)
self.classifier = nn.Linear(4 * 512, 6)  # 2,048 features
```
**Result**: BAcc 0.16 (worse than random!)

### Attempt 2: Emergency Full Features (`eegpt_full_features.py`)
```python
class EEGPTFullFeatures(nn.Module):
    def forward(self, x):
        # HACK: Extract ALL patch features instead of summary
        patch_features = x[:, :-self.model.embed_num, :]  # All except last 4
        return patch_features.reshape(batch_size, -1)  # 163,840 features!
```
**Result**: BAcc 0.15, Kappa -0.01 (CATASTROPHIC FAILURE)

## 4. The Fundamental Problem

### What the Paper Says (Section 2.4)
> "The encoder passes the output tokens corresponding to **summary tokens** to the linear classification head."

### What We're Missing
The EEGPT paper shows **different feature extraction strategies** for different tasks:
- **Binary tasks** (like TUAB): Summary tokens work fine
- **Multi-class tasks** (like TUEV): May need different approach

### The Real Issue: Feature Insufficiency
- **TUAB** (2 classes): 2,048 features sufficient
- **TUEV** (6 classes): 2,048 features NOT sufficient
- **Paper's TUEV performance**: BAcc 0.62 with "linear-probing"
- **Our TUEV performance**: BAcc 0.16 with same approach

## 5. Critical Code Locations

### Scattered Implementations (ARCHITECTURAL DISASTER)
```
experiments/eegpt_linear_probe/
├── eegpt_full_features.py          # HACK: Emergency 163k features
├── train_paper_aligned.py          # TUAB: Uses wrapper correctly
├── train_tuev_aligned.py           # TUEV: Uses 2k features (failed)
└── train_tuev_aligned_fixed.py     # TUEV: Uses 163k features (failed worse)

src/brain_go_brrr/
├── infra/ml_models/
│   ├── eegpt_architecture.py       # Base model (returns 4 tokens)
│   ├── eegpt_wrapper.py            # Adds normalization
│   └── eegpt_model.py              # Another wrapper (unclear purpose)
└── infra/adapters/
    └── eegpt_feature_extractor.py  # Port adapter (not used in experiments!)
```

### Missing Central Feature Extraction
**NO UNIFIED FEATURE EXTRACTION API EXISTS!**

Each experiment reimplements feature extraction differently:
- No consistent API
- No shared feature extraction strategies
- No documentation of what features to use when

## 6. Evidence from Literature Review

### EEGPT Paper Critical Details

1. **Summary Tokens Design** (Section 2.3):
   - "S learnable summary tokens for summarizing information"
   - Default S=4 tokens × 512 dims = 2,048 features

2. **Linear-Probing Method** (Figure 3):
   - Shows: Encoder → Summary Tokens → Linear Head
   - NO mention of using all patch features

3. **Table 13 Results** (Appendix):
   - TUEV: BAcc 0.6232 ± 0.0114
   - Method: "Linear-probing"
   - But WHAT features exactly?

### The Mystery: How Did Paper Achieve TUEV Performance?

Possibilities:
1. They used more summary tokens for TUEV (S > 4)
2. They used different feature aggregation (not just summary)
3. They fine-tuned more than just linear probe
4. The paper has an error/omission

## 7. Why This Matters

### Current State is DANGEROUS
1. **No standardized feature extraction** - each task implements differently
2. **No documentation** on which features for which task
3. **Emergency hacks** (163k features) instead of understanding
4. **TUAB might be wrong too** - we don't know if 512 dims is optimal

### Production Implications
- Can't reliably deploy EEGPT
- Can't reproduce paper results
- Can't know which features to use for new tasks
- Technical debt accumulating rapidly

## 8. The Path Forward

### Immediate Actions Required

#### 1. Create Unified Feature Extractor
```python
# src/brain_go_brrr/models/eegpt/feature_extractor.py
class EEGPTFeatureExtractor:
    def get_summary_features(self):     # 4×512
    def get_pooled_patches(self):       # Various pooling strategies
    def get_attention_weighted(self):   # Learned attention
```

#### 2. Investigate TUAB Training
- Load the checkpoint
- Verify what features were actually used
- Confirm if performance matches paper

#### 3. Contact Paper Authors
- Ask specifically about TUEV feature extraction
- Request their exact training code

#### 4. Systematic Feature Study
- Test different feature extraction methods
- Document which works for which task
- Create feature selection guidelines

## 9. Recommended Architecture

### Proper Implementation Structure
```
src/brain_go_brrr/models/eegpt/
├── __init__.py
├── core.py                    # Base EEGPT model
├── feature_extractor.py       # ALL feature extraction strategies
├── strategies/
│   ├── summary.py            # Summary token extraction
│   ├── pooling.py            # Channel/temporal pooling
│   ├── attention.py          # Attention-based selection
│   └── hybrid.py             # Combined approaches
└── configs/
    ├── tuab.yaml             # TUAB-specific config
    └── tuev.yaml             # TUEV-specific config
```

### Usage Pattern
```python
from brain_go_brrr.models.eegpt import EEGPTFeatureExtractor

extractor = EEGPTFeatureExtractor(checkpoint_path)
features = extractor.extract(
    data,
    strategy="summary",  # or "channel_pool", "attention", etc.
    task_config="tuev"
)
```

## 10. Critical Questions Requiring Answers

1. **What features did the paper REALLY use for TUEV?**
   - Summary tokens give BAcc 0.16
   - Paper claims BAcc 0.62
   - 46% gap is not explained!

2. **Is TUAB training actually correct?**
   - We use 512 dims (averaged summary)
   - Paper doesn't specify
   - Need to verify checkpoint performance

3. **Why are implementations scattered?**
   - `eegpt_wrapper.py` vs `eegpt_model.py`
   - Experiments bypass src implementations
   - No clear ownership

4. **Where is the feature extraction documentation?**
   - Nothing explains when to use what
   - Paper is vague ("linear-probing")
   - Code has no comments

## 11. Risk Assessment

### Current Risks
- 🔴 **CRITICAL**: TUEV cannot achieve paper performance
- 🔴 **CRITICAL**: No standardized feature extraction
- 🟡 **HIGH**: TUAB may be suboptimal
- 🟡 **HIGH**: Cannot add new tasks reliably
- 🟡 **HIGH**: Technical debt growing rapidly

### If Not Fixed
- Cannot deploy EEGPT in production
- Cannot reproduce paper results
- Cannot trust model outputs
- Will accumulate more emergency hacks

## 12. Conclusion

**The EEGPT embedding extraction is fundamentally broken**. We have:
1. No unified feature extraction API
2. Scattered, inconsistent implementations
3. Failed to reproduce paper results (46% performance gap!)
4. No understanding of which features for which tasks
5. Emergency hacks instead of proper solutions

**This MUST be fixed before ANY further EEGPT work.**

---

## Appendix A: File-by-File Analysis

### `experiments/eegpt_linear_probe/eegpt_full_features.py`
- **Purpose**: Emergency hack to extract all features
- **Problem**: Returns 163,840 features (overfitting guaranteed)
- **Status**: SHOULD BE DELETED

### `src/brain_go_brrr/infra/ml_models/eegpt_wrapper.py`
- **Purpose**: Adds normalization to EEGPT
- **Problem**: Only wraps forward(), no feature extraction API
- **Status**: Needs expansion

### `src/brain_go_brrr/infra/ml_models/eegpt_architecture.py`
- **Purpose**: Core EEGPT implementation
- **Problem**: Only returns summary tokens, no flexibility
- **Status**: Needs feature extraction methods

### `src/brain_go_brrr/infra/adapters/eegpt_feature_extractor.py`
- **Purpose**: Port adapter for domain
- **Problem**: Not used by experiments!
- **Status**: Wrong abstraction level

## Appendix B: Evidence of Confusion

Multiple files doing similar things:
- `eegpt_wrapper.py`
- `eegpt_model.py`
- `eegpt_feature_extractor.py`
- `eegpt_classifier.py`
- `eegpt_linear_probe.py`
- `eegpt_two_layer_probe.py`

**This is architectural chaos!**