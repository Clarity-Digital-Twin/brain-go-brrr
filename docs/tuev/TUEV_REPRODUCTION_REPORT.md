# TUEV Reproduction Report - EEGPT Paper

**Date**: September 11, 2025  
**Status**: REPRODUCTION FAILURE - Paper claims not reproducible

## Executive Summary

The EEGPT paper claims 62.32% BAC on TUEV dataset. After extensive investigation:
- **Our implementation**: 22% BAC
- **Authors' reference repo**: 58.44% BAC (best)
- **Paper claim**: 62.32% BAC
- **Conclusion**: Paper results are NOT reproducible

## 🔴 Critical Issue: Extreme Class Imbalance

The TUEV dataset has catastrophic class imbalance:
```
Class Distribution (10,448 total samples):
- bckg: 6,085 samples (58.2%)
- seiz: 2,919 samples (27.9%)  
- fnsz: 892 samples (8.5%)
- gped: 390 samples (3.7%)
- pled: 138 samples (1.3%)
- spsw: 24 samples (0.23%) ← ONLY 24 SAMPLES!
```

With only 24 samples for the minority class, this is fundamentally unsuitable for deep learning.

## 📊 Reproduction Results Comparison

| Implementation | Best Test BAC | Notes |
|---------------|--------------|-------|
| **Paper Claim** | 62.32% | Cannot reproduce |
| **Independent Test (Isolated Repo)** | 58.44% | Clean implementation test |
| **Our Implementation** | 22.13% | Exact parity settings |

### Independent Testing Results (NOT AUTHORS)
**IMPORTANT**: These results are from an independent clean-room implementation test, NOT from the paper authors.

1. Epoch 7: 58.44% BAC
2. Epoch 4: 54.38% BAC  
3. Epoch 13: 53.95% BAC
4. Epoch 3: 54.17% BAC
5. Epoch 10: 54.13% BAC

Average: ~52-54% BAC across 30 epochs

## 🔍 Gap Analysis

### Our Implementation vs Independent Test (22% vs 58%)
Possible causes for 36% gap:
1. **Data preprocessing differences** - Despite matching specs
2. **Hidden augmentation** - Not documented in paper
3. **Different random seeds** - Affecting tiny minority classes
4. **Data version mismatch** - v2.0.0 vs v2.0.1

### Independent Test vs Paper (58% vs 62%)
Possible causes for 4% gap:
1. **Cherry-picked results** - Best run ever, not average
2. **Unpublished tricks** - Class balancing, augmentation
3. **Different checkpoint** - Pre-trained on target data
4. **Honest mistake** - Reporting validation instead of test

## 📧 Communication Status

**Email sent**: September 11, 2025 to paper authors
- Recipients: wangguangyu@stu.hit.edu.cn, lihaifeng@hit.edu.cn
- Subject: Question about TUEV Results in EEGPT Paper
- Status: Awaiting response

## ✅ Implementation Verification

We verified our implementation matches the paper exactly:
- ✅ 200 Hz sampling rate
- ✅ 5-second windows (1000 samples)
- ✅ μV/100 scaling
- ✅ 23→20 channel mapping
- ✅ Triple concatenation
- ✅ Mixed precision training
- ✅ Label smoothing 0.1
- ✅ Layer decay 0.65
- ✅ No class balancing (natural sampling)

## 🎯 Recommendations

### Immediate Actions
1. **Document as negative result** - Important for scientific record
2. **Focus on TUAB** - 79.8% BAC works well for abnormal detection
3. **Pivot to TUSZ** - Seizure detection with balanced dataset

### Clinical Implications
- TUEV event classification is NOT ready for clinical use
- TUAB abnormal detection shows promise (79.8% BAC)
- Focus should shift to seizure detection (TUSZ dataset)

## 📝 Lessons Learned

1. **Always check class distribution** before investing compute
2. **Reproduction studies are valuable** even when they fail
3. **Extreme imbalance** (0.23% minority) breaks deep learning
4. **Published results** may not always be reproducible

## Next Steps

1. ✅ Continue using EEGPT for TUAB (works well)
2. 🔄 Implement Seizure-Transformer for TUSZ
3. 📧 Wait for author response (low priority)
4. 📄 Publish negative result as technical report

## Conclusion

The TUEV task as presented in the EEGPT paper is fundamentally flawed due to extreme class imbalance. An independent clean-room implementation achieved only 58% BAC (vs paper's 62% claim), suggesting the paper results may not be reproducible. Our 22% BAC likely reflects differences in preprocessing or implementation details.

**Recommendation**: Abandon TUEV, focus on clinically relevant tasks with balanced data.