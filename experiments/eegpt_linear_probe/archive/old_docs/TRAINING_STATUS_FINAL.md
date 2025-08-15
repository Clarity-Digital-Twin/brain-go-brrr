# EEGPT Linear Probe Training Status - FINAL WORKING VERSION

**Last Updated**: 2025-08-06 08:10 UTC

## 🚀 CACHE BUILD IN PROGRESS

```bash
# Monitor cache build progress
tail -f cache_build.log | grep "Processing files"

# Current: 435/2993 files (14.5%) @ ~1.1 files/sec
# ETA: ~38 minutes remaining (25 min elapsed)
```

## ✅ WHAT'S FIXED

1. **Cache builder optimized**:
   - ✅ NO filtering (EEGPT trained on unfiltered data)
   - ✅ Channel order preserved (ordered=True)
   - ✅ Resample only when needed (saves 20% time)
   - ✅ File name collisions prevented
   - ✅ Tensor storage (not dicts) for fast loading

2. **4-second windows** (CRITICAL):
   - EEGPT was pretrained on 4s windows
   - Paper achieved 0.869 AUROC with 4s
   - 8s windows only get ~0.81 AUROC

3. **Cleaned up old broken shit**:
   - Archived to `archive/broken_attempts_aug6/`
   - Deleted old broken caches

## 📊 CACHE SPECIFICATIONS

| Parameter | Value |
|-----------|-------|
| Window size | 4.0 seconds |
| Stride | 2.0 seconds (50% overlap) |
| Sampling rate | 256 Hz |
| Window samples | 1024 |
| Channels | 20 (10-20 system) |
| Cache format | Tensors {'x': (N,C,T), 'y': labels} |
| Cache directory | `/data/cache/tuab_4s_final/` |
| Index file | `/data/cache/tuab_4s_final/index.json` |

## 🎯 LAUNCH TRAINING (After Cache Completes)

```bash
# Check if cache is ready
./experiments/eegpt_linear_probe/LAUNCH_FINAL_TRAINING.sh

# Monitor training
tmux attach -t eegpt_final

# Check speed (MUST be >400 it/s)
tail -f logs/final_4s_*.log | grep it/s
```

## ⚡ EXPECTED PERFORMANCE

| Metric | Expected | Red Flag |
|--------|----------|----------|
| Cache build time | ~45 minutes | >2 hours |
| Windows cached | >15M | <1M |
| Training speed | 400-600 it/s | <100 it/s |
| Epoch time | ~15 minutes | >1 hour |
| Target AUROC | ≥0.869 | <0.85 |
| Memory usage | ~8GB GPU | OOM errors |

## 🔴 IF TRAINING IS SLOW

If iteration speed <100 it/s, cache is NOT being used:
1. Check cache index exists and has entries
2. Verify config points to correct index
3. Check dataloader is using cached dataset

## 📁 FILE STRUCTURE

```
experiments/eegpt_linear_probe/
├── build_4s_cache_FINAL.py       # FIXED cache builder (running now)
├── LAUNCH_FINAL_TRAINING.sh      # Training launcher
├── train_paper_aligned.py        # Pure PyTorch trainer (works!)
├── configs/
│   └── tuab_4s_final.yaml       # Working config
└── archive/
    └── broken_attempts_aug6/      # Old broken shit
```

## 🎉 SUCCESS CRITERIA

- [ ] Cache build completes (~2993 files)
- [ ] Index shows >10M windows
- [ ] Training launches without errors
- [ ] Speed >400 it/s confirmed
- [ ] First epoch completes in <20 min
- [ ] AUROC reaches 0.869 ± 0.005

---

**DO NOT INTERRUPT THE CACHE BUILD!** Let it finish, then launch training.
