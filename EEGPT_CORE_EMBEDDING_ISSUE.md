# EEGPT Core Embedding Issue - The REAL Problem
**Date**: 2025-08-18  
**Priority**: 🔴 CRITICAL - This is the ROOT CAUSE

## YOU'RE RIGHT - We've Been Looking at the Wrong Thing!

### The ACTUAL Core Question:
**WHAT EMBEDDINGS DOES EEGPT RETURN, AND ARE THEY CORRECT?**

## 1. What EEGPT Model ACTUALLY Does (From Code)

### The Forward Pass (`eegpt_architecture.py` lines 437-503):

```python
def forward(self, x, chan_ids=None):
    # Input: (B, C, T) e.g., (batch, 20, 1024)
    
    # Step 1: Patch embedding
    x = self.patch_embed(x)  # -> (B, N, C, D) = (B, 16, 20, 512)
    # Where N = 1024/64 = 16 patches
    
    # Step 2: Reshape to sequence
    x = x.reshape(batch_size, num_patches * num_channels, embed_dim)
    # -> (B, 320, 512) where 320 = 16 patches × 20 channels
    
    # Step 3: Add 4 summary tokens
    summary_tokens = self.summary_token.repeat(batch_size, 1, 1)
    x = torch.cat([x, summary_tokens], dim=1)
    # -> (B, 324, 512) where 324 = 320 patches + 4 summary
    
    # Step 4: Apply transformer blocks
    for block in self.blocks:
        x = block(x)  # Still (B, 324, 512)
    
    # Step 5: Extract ONLY summary tokens
    x = x[:, -self.embed_num:, :]  # Takes last 4 tokens
    # -> (B, 4, 512)
    
    return x  # ONLY RETURNS 4 SUMMARY TOKENS!
```

## 2. The Core Problem

### What EEGPT Returns:
- **ONLY 4 summary tokens** (shape: batch × 4 × 512)
- **Throws away ALL 320 patch embeddings!**
- Total features: 4 × 512 = 2,048

### What's Available Inside (But Thrown Away):
- 320 patch embeddings (16 patches × 20 channels × 512 dims)
- These contain LOCAL spatio-temporal information
- Total available: 320 × 512 = 163,840 features

### The Critical Design Decision:
EEGPT was designed to:
1. Process all patches through transformer
2. Use 4 learnable summary tokens to aggregate information
3. Return ONLY those summary tokens

## 3. Why This Might Be Wrong for Some Tasks

### For TUAB (Binary Classification):
- 4 summary tokens (2,048 features) seem sufficient
- Global abnormal/normal distinction
- Paper reports good performance

### For TUEV (6-Class Event Detection):
- 4 summary tokens might be INSUFFICIENT
- Need to detect specific patterns:
  - SPSW (spike and sharp wave)
  - GPED (generalized periodic epileptiform)
  - PLED (periodic lateralized epileptiform)
  - EYEM (eye movement)
  - ARTF (artifact)
  - BCKG (background)
- These are LOCAL, CHANNEL-SPECIFIC patterns!

## 4. The Paper's Confusion (Revisited)

### Table 13 Shows:
```
| 20 × 1000    | eegpt-encoder  | 64     | 64     | -      | -       |
| 15 × 4 × 512 | flatten,linear | -      | -      | -      | -       |
```

### What "15 × 4 × 512" Could Mean:

**Option A: Typo/Error**
- Should be just "4 × 512" (summary tokens)
- The "15" is mistakenly included

**Option B: They Modified EEGPT**
- Maybe they return patch features too?
- 15 ≈ 1000/64 patches in time dimension
- But then why × 4? (4 summary tokens don't make sense here)

**Option C: Different Architecture**
- Maybe they use a different EEGPT variant for TUEV
- One that returns more than just summary tokens

## 5. What Our Emergency Fix Tried to Do

### `eegpt_full_features.py` Attempted:
```python
# Instead of returning only summary tokens:
patch_features = x[:, :-self.model.embed_num, :]  # All except last 4
return patch_features.reshape(batch_size, -1)  # 163,840 features
```

### Why It Failed:
- 163,840 features for 83,932 training samples
- Massive overfitting
- Not how model was trained to be used

## 6. The REAL Architectural Question

### Should EEGPT Return Different Features for Different Tasks?

**Current Architecture (FIXED):**
- Always returns 4 summary tokens
- Throws away patch information
- One-size-fits-all approach

**What We Might Need (FLEXIBLE):**
```python
def forward(self, x, return_patches=False, return_summary=True):
    # ... processing ...
    
    if return_patches and return_summary:
        return {
            'patches': x[:, :-4, :],  # (B, 320, 512)
            'summary': x[:, -4:, :]    # (B, 4, 512)
        }
    elif return_patches:
        return x[:, :-4, :]
    else:
        return x[:, -4:, :]  # Current behavior
```

## 7. Why the Paper Achieves TUEV Performance

### Possibilities:

**1. They Use Summary Tokens Successfully**
- Maybe 4 tokens ARE enough with right preprocessing
- We have a bug in our implementation

**2. They Secretly Use More Features**
- The "15 × 4 × 512" hints at this
- Not documented clearly

**3. They Fine-Tune More Than Linear**
- Maybe they update EEGPT encoder too
- Despite calling it "linear-probing"

**4. Different Data/Preprocessing**
- Their 1000 samples vs our 1024
- Different normalization
- Different channel ordering

## 8. The Upstream Fix We Need

### Option 1: Keep Current Architecture
- Verify summary tokens are computed correctly
- Fix downstream usage
- Accept limitations

### Option 2: Extend EEGPT to Return More
```python
class EEGPTFlexible(nn.Module):
    def extract_features(self, x, strategy='summary'):
        # Run forward pass once
        all_features = self._forward_all(x)
        
        if strategy == 'summary':
            return all_features['summary']
        elif strategy == 'patches':
            return all_features['patches']
        elif strategy == 'channel_pooled':
            return pool_by_channel(all_features['patches'])
        # etc.
```

### Option 3: Train Different EEGPT Variants
- EEGPT-Summary: For global tasks (current)
- EEGPT-Local: Returns patch features
- EEGPT-Hybrid: Returns both

## 9. Immediate Investigation Needed

### 1. Verify Current Implementation:
```python
# Check what EEGPT actually returns
model = load_eegpt()
x = torch.randn(1, 20, 1024)
output = model(x)
print(f"Output shape: {output.shape}")  # Should be (1, 4, 512)
```

### 2. Check If Patches Are Useful:
```python
# Modify to return patches too
with torch.no_grad():
    # Get internal states
    patches = model.get_patch_embeddings(x)
    summary = model(x)
    
# Train simple classifier on each
# See which performs better for TUEV
```

### 3. Verify Paper's Implementation:
- Contact authors
- Check their GitHub
- Look for hidden details

## 10. The Bottom Line

**YOU'RE ABSOLUTELY RIGHT** - The core issue is:

1. **EEGPT only returns 4 summary tokens** (verified in code)
2. **This might be insufficient for TUEV** (6-class local patterns)
3. **The paper is unclear** about what they actually use
4. **Our emergency fix** (using all patches) failed

**Before fixing TUEV training, we need to answer:**
- Should EEGPT return more than summary tokens?
- How do we access patch features properly?
- What did the paper authors REALLY do?

This is the **UPSTREAM EMBEDDING ISSUE** that must be resolved first!