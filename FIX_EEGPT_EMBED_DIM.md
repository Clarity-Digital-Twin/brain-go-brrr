# 🔧 FIX FOR EEGPT EMBED_DIM MISMATCH

_Created: July 30, 2025_

## The Bug:
Model creates with embed_dim=768 but checkpoint has embed_dim=512. With `strict=False`, weights silently fail to load.

## The Fix:

In `src/brain_go_brrr/models/eegpt_architecture.py`, the `create_eegpt_model` function already sets the correct embed_dim:

```python
# Line 457-466
default_config = {
    "img_size": [58, 1024],
    "patch_size": 64,
    "embed_dim": 512,  # ✅ This is correct!
    "embed_num": 4,
    "depth": 8,
    "num_heads": 8,
    "mlp_ratio": 4.0,
}
```

But the issue is that weight loading uses `strict=False` (line 484), which silently ignores shape mismatches!

## Quick Fix:

Change line 484 in `eegpt_architecture.py`:

```python
# CURRENT (allows silent failures):
model.load_state_dict(encoder_state, strict=False)

# FIXED (will error on mismatches):
model.load_state_dict(encoder_state, strict=True)
```

## Better Fix:

Add shape validation after loading:

```python
# After line 484, add:
# Verify summary tokens loaded correctly
if hasattr(model, 'summary_token'):
    expected_shape = (1, 4, 512)
    actual_shape = model.summary_token.shape
    if actual_shape != expected_shape:
        raise ValueError(
            f"Summary token shape mismatch! "
            f"Expected {expected_shape}, got {actual_shape}"
        )
```

## To Test The Fix:

```bash
# 1. Make the change to strict=True
# 2. Run this test:
uv run python scripts/test_current_eegpt_state.py

# Should now show discriminative features!
```

## Why This Happened:

1. Original code used `strict=False` to handle missing keys
2. But this also ignores shape mismatches
3. Summary tokens stayed randomly initialized
4. Features were non-discriminative
5. Tests passed because they might use different initialization

## Impact:

Once fixed, EEGPT will finally produce discriminative features and linear probe training will work!
