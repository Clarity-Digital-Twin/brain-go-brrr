# ⚠️ ALWAYS RUN THIS FIRST - NO EXCEPTIONS!

## Before ANY Training Run:

```bash
# 1. Run pre-flight check
cd experiments/eegpt_linear_probe
uv run python preflight_check.py

# 2. Only if ALL GREEN, then start training
tmux new-session -d -s eegpt_safe "uv run python train_overnight_optimized.py"
```

## Why?
- We lost 8 hours to a missing `import numpy`
- Pre-flight check catches ALL dependency issues
- Tests the EXACT validation code that crashed
- Verifies GPU memory and data paths

## Monitor Training:
```bash
tmux attach -t eegpt_fixed_v2
tail -f logs/overnight_fixed.log | grep -E "Epoch|AUROC|loss"
```

## Current Status:
- Training running since: Fri Aug 1 06:06:11 2025
- Expected completion: ~2-3 PM
- Target AUROC: ≥0.87