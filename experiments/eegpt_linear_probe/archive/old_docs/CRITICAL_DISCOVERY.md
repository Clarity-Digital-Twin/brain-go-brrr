# CRITICAL DISCOVERY: How EEGPT Actually Works for TUEV

## The Architecture We Misunderstood

Our implementation processes ALL patches together:
```
Input (B, 20, 1000) → Patches (B, 15*20, 512) → Transformer → 4 summary tokens total
```

## How Reference EEGPT Actually Works

The reference processes EACH temporal position SEPARATELY:

```python
# Line 514: Create patches
x = self.patch_embed(x)  # (B, N=15, C=20, D=512)

# Line 532: Flatten to process each temporal position
x = x.flatten(0, 1)  # (B*N, C, D) = (B*15, 20, 512)

# Line 535-536: Add summary tokens TO EACH temporal position
summary_token = self.summary_token.repeat((x.shape[0], 1, 1))  # (B*15, 4, 512)
x = torch.cat([x, summary_token], dim=1)  # (B*15, 24, 512)

# After transformer blocks (line 543): Extract summary tokens
x = x[:, -summary_token.shape[1]:, :]  # (B*15, 4, 512)

# Reshape back (lines 549-550, 558):
x = x.reshape((B, N, self.embed_num, -1))  # (B, 15, 4, 512)

# In classifier (line 843):
x = x.flatten(1)  # (B, 30720)
```

## The Key Insight

**EACH of the 15 temporal patches gets its OWN 4 summary tokens!**

- Not 4 summary tokens total
- But 15 × 4 = 60 summary tokens
- This preserves temporal structure
- Final features: 15 × 4 × 512 = 30,720

## Why This Makes Sense

1. Each 64-sample patch (250ms) gets analyzed independently
2. Each patch produces 4 summary features
3. The linear probe sees ALL temporal positions
4. This allows learning temporal patterns across the 1-second window

## The Fix

We need to restructure our encoder to:
1. Process patches per temporal position
2. Get summary tokens for EACH position
3. Return shape (B, num_temporal, 4, 512) not (B, 4, 512)

This explains:
- Why we get BAcc 0.15 (only 4 tokens, no temporal info)
- Why paper needs 30,720 features (15 positions × 4 tokens × 512)
- Why the reference hardcodes `LinearWithConstraint(30720, num_classes)`
