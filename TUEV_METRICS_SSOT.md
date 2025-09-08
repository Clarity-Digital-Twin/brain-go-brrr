# 📊 TUEV METRICS - SINGLE SOURCE OF TRUTH

**Created**: September 8, 2025
**Source**: EEGPT Paper Table 3 (Page 7)
**Purpose**: Definitive reference for TUEV performance targets

---

## 🎯 OFFICIAL PERFORMANCE TARGETS

### From EEGPT Paper (Our Model):
| Metric | Value | Standard Deviation |
|--------|-------|-------------------|
| **Balanced Accuracy (BAC)** | **62.32%** | ± 1.14% |
| **Weighted F1** | **81.87%** | ± 0.63% |
| **Cohen's Kappa** | **0.635** | ± 0.013 |

### Baseline Comparisons:
| Model | BAC | Weighted F1 | Kappa | Notes |
|-------|-----|-------------|-------|-------|
| **EEGPT (Ours)** | **62.32%** | **81.87%** | **0.635** | Target to match |
| LaBraM | 64.09% | 83.12% | 0.664 | Uses larger pretraining data |
| BIOT | 52.81% | 74.92% | 0.527 | Previous baseline |
| ST-T | 39.84% | 68.23% | 0.377 | Older baseline |

---

## 🚨 CRITICAL CORRECTIONS NEEDED

### 1. **REMAINING_DEBT.md Line 283**
- **WRONG**: "If TUEV val accuracy < 80% ⇒ implement Sprint 4"
- **CORRECT**: "If TUEV BAC < 60% ⇒ implement Sprint 4"
- **Rationale**: 80% is unrealistic; SOTA is ~64% (LaBraM)

### 2. **README.md**
- **Current**: "Event Detection (TUEV) | Target: 62% BAC"
- **Status**: ✅ CORRECT (matches EEGPT paper)

### 3. **docs/TRAINING.md**
- **Current**: "Balanced Accuracy: Main metric for TUEV (target: 0.62)"
- **Status**: ✅ CORRECT (matches EEGPT paper)

---

## 📈 REALISTIC EXPECTATIONS

### Performance Tiers:
- **POOR**: < 50% BAC (below BIOT baseline)
- **ACCEPTABLE**: 50-55% BAC (matches BIOT)
- **GOOD**: 55-60% BAC (approaching EEGPT)
- **TARGET**: 60-63% BAC (matches EEGPT paper)
- **EXCELLENT**: 63-65% BAC (approaches LaBraM)
- **UNREALISTIC**: > 70% BAC (never reported in literature)

### Decision Thresholds:
- **< 55% BAC**: Definitely implement Sprint 4 (channel mapper)
- **55-60% BAC**: Consider implementing Sprint 4
- **60-65% BAC**: TARGET ACHIEVED - Skip Sprint 4
- **> 65% BAC**: EXCELLENT - Exceeds expectations

---

## 🔍 WHY THE CONFUSION?

### The 80% Weighted F1 vs 62% BAC Issue:
- **Weighted F1 ~82%** looks high but is inflated by class imbalance
- **BAC ~62%** is the TRUE performance metric (balanced across classes)
- Someone likely confused Weighted F1 (82%) with BAC (62%) when writing "80% accuracy"

### Class Imbalance Impact:
- TUEV is 99.5% class 5 (background)
- Model can get 80%+ F1 by mostly predicting class 5
- BAC corrects for this imbalance
- **ALWAYS use BAC for TUEV**, not raw accuracy or F1

---

## ✅ ACTION ITEMS

1. **Fix REMAINING_DEBT.md**: Change "80%" to "60%" threshold
2. **Keep README.md**: Already correct (62% BAC target)
3. **Keep TRAINING.md**: Already correct (0.62 target)
4. **Monitor training**: Look for BAC, not weighted F1
5. **Decision point**: 60% BAC = success, not 80%

---

## 📝 MONITORING COMMANDS

```bash
# Watch for BAC specifically
tail -f experiments/eegpt_linear_probe/logs/tuev_mne_*.log | grep -i "balanced"

# Look for these lines:
# "Balanced Accuracy: 0.XX"
# "val_bac: 0.XX"
# "test_bac: 0.XX"
```

---

## 🎓 REFERENCES

1. **EEGPT Paper**: Table 3, Page 7 - TUEV results
2. **ALFEE Paper**: Figure 1 - Multi-model comparison (~65% BAC)
3. **LaBraM Paper**: Table showing 64.09% BAC on TUEV
4. **BIOT Paper**: Original 52.81% BAC baseline

---

## ⚠️ IMPORTANT NOTES

1. **Primary Metric**: ALWAYS use Balanced Accuracy (BAC) for TUEV
2. **Secondary Metrics**: Weighted F1 and Kappa for additional context
3. **Class Weights**: Essential due to 99.5% class imbalance
4. **Realistic Target**: 60-63% BAC, NOT 80%
5. **Sprint 4 Decision**: Based on 60% BAC threshold, not 80%

---

**THIS DOCUMENT IS THE SINGLE SOURCE OF TRUTH FOR TUEV METRICS**
