# EEGPT Feature Extraction Architecture Report
**Author**: Senior Architecture Review  
**Date**: 2025-08-18  
**Priority**: 🔴 CRITICAL - Blocking Production  

## Executive Summary

You're absolutely right to be concerned. The current EEGPT feature extraction is a **MESS** scattered across experiments with no clear architectural strategy. This is NOT how production ML systems should be built.

**CORE ISSUE**: We're confusing model capabilities with task requirements, leading to ad-hoc feature extraction in random files.

## The Fundamental Architectural Problem

### Current State (BROKEN)
```
experiments/eegpt_linear_probe/
  ├── eegpt_full_features.py      # Emergency hack extracting ALL features
  ├── train_tuab.py                # Uses summary tokens (4×512)
  └── train_tuev_aligned_fixed.py  # Uses full features (163,840 dims)

src/brain_go_brrr/
  └── infra/ml_models/
      └── eegpt_architecture.py   # Base model, returns ONLY summary tokens
```

This is **architecturally WRONG** because:
1. Feature extraction strategy is coupled to training scripts
2. No reusable API for different extraction modes
3. Emergency fixes living in experiments instead of core
4. Violates Single Responsibility Principle

### What It SHOULD Be

```
src/brain_go_brrr/models/eegpt/
  ├── __init__.py
  ├── base_model.py              # Core EEGPT architecture
  ├── feature_extractor.py       # ALL extraction strategies
  └── task_heads.py              # Task-specific processing
```

## The Senior Architect's Answer: YES, Extract Everything First

### The Correct Architecture Pattern

```python
# src/brain_go_brrr/models/eegpt/feature_extractor.py

class EEGPTFeatureExtractor:
    """Central feature extraction API for all EEGPT operations."""
    
    def __init__(self, checkpoint_path: str):
        self.model = load_eegpt(checkpoint_path)
        self.model.eval()
    
    def extract_all_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract ALL available features in one forward pass.
        
        This is the MASTER method - runs ONCE, returns EVERYTHING.
        Downstream tasks select what they need.
        """
        # Single forward pass through transformer
        patch_embeddings, summary_tokens = self._forward_pass(x)
        
        return {
            # Raw outputs
            'patch_embeddings': patch_embeddings,      # (B, 320, 512) 
            'summary_tokens': summary_tokens,          # (B, 4, 512)
            
            # Computed aggregations (cheap operations)
            'global_features': summary_tokens.flatten(1),              # (B, 2048)
            'channel_features': self._pool_by_channel(patch_embeddings), # (B, 20, 512)
            'temporal_features': self._pool_by_time(patch_embeddings),   # (B, 16, 512)
            'full_features': patch_embeddings.flatten(1),              # (B, 163840)
        }
    
    # Convenience methods that internally call extract_all_features
    def get_summary_features(self, x):
        """For TUAB and simple classification."""
        return self.extract_all_features(x)['global_features']
    
    def get_channel_pooled_features(self, x):
        """For TUEV - channel-wise patterns."""
        return self.extract_all_features(x)['channel_features']
    
    def get_temporal_attention_features(self, x):
        """For seizure detection - temporal dynamics."""
        features = self.extract_all_features(x)
        return self._apply_temporal_attention(features['patch_embeddings'])
```

### Why This Is The Right Pattern

1. **Single Forward Pass**: EEGPT inference is expensive. Do it ONCE, extract everything.

2. **Separation of Concerns**: 
   - Model knows how to encode
   - Extractor knows how to extract
   - Tasks know what features they need

3. **Flexibility Without Chaos**: Tasks can experiment with different features without modifying core.

4. **Caching Friendly**: Can cache the full extraction and serve different views.

## The Critical Insight You're Missing

EEGPT is a **Vision Transformer** that produces:

```
Input (20 channels × 1024 samples)
    ↓
16 patches/channel × 20 channels = 320 patch tokens
    ↓
Transformer (self-attention across ALL patches)
    ↓
320 patch embeddings + 4 summary tokens
```

**THE KEY**: Those 4 summary tokens are LEARNED to summarize the 320 patches through attention. They're not just random projections!

### What Each Feature Type Captures

| Feature Type | Dimensions | What It Captures | Best For |
|-------------|------------|------------------|----------|
| Summary Tokens | 4×512 (2,048) | Global patterns via learned attention | Simple classification (TUAB) |
| Channel-Pooled | 20×512 (10,240) | Per-channel patterns | Channel-specific events (TUEV) |
| Temporal-Pooled | 16×512 (8,192) | Temporal dynamics | Time-locked events |
| Full Patches | 320×512 (163,840) | Everything (overkill) | Research only |

## Why TUEV Is Failing

**You're using 163,840 features for 83,932 training samples!**

This violates the fundamental ML rule: features << samples for linear models.

The paper likely uses **channel-pooled features** (20×512 = 10,240 dims) for TUEV because:
- SPSW, GPED, PLED are channel-specific patterns
- 10k features is reasonable for 83k samples
- Preserves channel locality unlike summary tokens

## The Immediate Fix

### Step 1: Create Proper Feature Extractor (30 min)
```bash
# Create the proper structure
mkdir -p src/brain_go_brrr/models/eegpt
touch src/brain_go_brrr/models/eegpt/__init__.py
```

### Step 2: Move Feature Extraction to Core
```python
# src/brain_go_brrr/models/eegpt/feature_extractor.py
# (Full implementation as shown above)
```

### Step 3: Update Training Scripts
```python
# experiments/eegpt_linear_probe/train_tuev.py
from brain_go_brrr.models.eegpt import EEGPTFeatureExtractor

extractor = EEGPTFeatureExtractor(checkpoint_path)

# Try channel-pooled first (10k features)
features = extractor.get_channel_pooled_features(x)  # (B, 20, 512)
features_flat = features.flatten(1)  # (B, 10240)

# If that fails, try temporal attention
features = extractor.get_temporal_attention_features(x)
```

## Architectural Principles Moving Forward

### 1. Models Are Services, Not Scripts
- EEGPT should be a service that provides features
- Training scripts are clients that consume features
- Never put model logic in experiments/

### 2. Extract Once, Use Many
- One forward pass, multiple feature views
- Let tasks choose their view
- Cache when possible

### 3. Explicit Over Implicit
```python
# BAD
features = model(x)  # What features? Who knows!

# GOOD  
all_features = extractor.extract_all_features(x)
task_features = all_features['channel_features']
```

### 4. Progressive Complexity
Start simple, add complexity only when needed:
1. Try summary tokens (2k features)
2. Try channel-pooled (10k features)  
3. Try temporal attention (custom)
4. Only use full features for research

## Action Items

### IMMEDIATE (Kill the Fire)
1. **ABORT current training** - it's using wrong features
2. Create `src/brain_go_brrr/models/eegpt/feature_extractor.py`
3. Implement channel-pooled extraction (10k features)
4. Restart TUEV with channel-pooled features

### THIS WEEK (Fix the Architecture)
1. Move ALL feature extraction to src/
2. Delete `experiments/eegpt_full_features.py`
3. Create comprehensive tests for feature extractor
4. Document which tasks use which features

### LONG TERM (Build It Right)
1. Feature extraction should be configurable via YAML
2. Add feature caching layer for repeated inference
3. Benchmark all extraction strategies systematically
4. Create feature selection optimizer

## The Bottom Line

**You're 100% correct** - feature extraction belongs in `src/` as a core capability, not scattered in experiments. The model should extract everything efficiently in one pass, and downstream tasks should select what they need.

Current approach (163k features) is architecturally wrong AND mathematically doomed. Switch to channel-pooled features (10k) immediately.

Remember: **Architecture is about making the right thing easy and the wrong thing hard.**

---

**Recommendation**: Stop current training NOW. Implement proper architecture. Then retry with 10k channel-pooled features.