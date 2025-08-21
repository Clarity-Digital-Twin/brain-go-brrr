# ⚠️ WARNING: OUTDATED DOCUMENTATION

_Created: July 30, 2025_

## These temp_docs contain CONFLICTING and OUTDATED information!

### The Confusion:
Multiple documents were created assuming the summary token fix was complete, but the weights weren't actually loading due to an embed_dim mismatch (768 vs 512).

### Documents with Issues:

1. **EEGPT_SUMMARY_TOKEN_FIX.md** - Claims fix is complete, but weights don't load
2. **EEGPT_INVESTIGATION_FINDINGS.md** - Correctly identifies broken features but doesn't know why
3. **CHECKPOINT_2025-07-29_FINAL.md** - Prematurely declares victory
4. **EEGPT_MVP_REALISTIC.md** - Assumes features work when they don't

### The Truth:
- Architecture was fixed ✅
- But weights don't load due to shape mismatch ❌
- Features remain non-discriminative ❌

### For Current Status:
See `EEGPT_TRUE_STATUS.md` in the root directory.

### For the Real Fix:
The model initialization uses embed_dim=768 but checkpoint has embed_dim=512.
With strict=False, PyTorch silently ignores this and summary tokens stay random.
