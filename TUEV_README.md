# TUEV Training Documentation Guide

## 📁 Document Hierarchy (USE IN THIS ORDER)

### 1. **TUEV_UNIFIED_SPECS.md** - PRIMARY REFERENCE ⭐
- Single source of truth
- All parameters with exact paper line numbers
- Clear [Paper] vs [Local] vs [Decision] labels
- Use this for implementation

### 2. **TUEV_CRITICAL_ARCHITECTURE.md** - ARCHITECTURE DETAILS
- Exact Table 13 reproduction
- Layer-by-layer architecture
- Implementation code examples

### 3. **TUEV_CRITICAL_SPECS.md** - SPECIFICATIONS
- Condensed specifications
- Validation checklist with assertions
- Performance targets

### 4. **TUEV_IMPLEMENTATION_PLAN.md** - DETAILED PLAN
- Phase-by-phase implementation
- Dataset loader examples
- Training pipeline code

### 5. **TUEV_QUICK_START.md** - QUICK REFERENCE
- At-a-glance parameters
- Common pitfalls
- Command summary

---

## ⚠️ CRITICAL FACTS TO REMEMBER

### The Paper Has a Contradiction
- **Text says**: "112,491 5-second samples" (line 585)
- **Table 13 shows**: 23 × 1000 input (3.9 seconds)
- **WE USE**: Table 13 (1000 samples) - it's the implementation

### Key Parameters (FROM TABLE 13)
```python
INPUT_SIZE = (23, 1000)  # NOT (23, 1280)!
WINDOW_SECONDS = 3.90625  # NOT 5!
DROPOUT = 0.5  # NOT 0.25!
KERNEL = 55  # NOT 15!
BATCH_SIZE = 500  # NOT 100!
```

### Our Local Data
- Path: `/data/datasets/external/tuh_eeg/TUEV/v2.0.1/`
- 518 EDF files, 11,396 .lab files
- 370 subjects (290 train, 80 eval)
- Files are 250 Hz → need resampling to 256 Hz
- Files have 26-27 channels → need to select 23

---

## ✅ Implementation Checklist

1. ✅ Dataset downloaded (v2.0.1)
2. ✅ Documentation aligned and consistent
3. ⬜ Create TUEV dataset loader
4. ⬜ Implement preprocessing (resample, channel selection)
5. ⬜ Build 6-class linear probe
6. ⬜ Train with exact Table 13 architecture
7. ⬜ Run 3 times (paper protocol)
8. ⬜ Compare to targets (BAC ≥ 0.62, F1 ≥ 0.81, κ ≥ 0.63)

---

## 🚫 Deleted Documents (No Longer Needed)
- ~~TUEV_INCONSISTENCY_ANALYSIS.md~~ - Investigation complete
- ~~TUEV_PAPER_PROOF.md~~ - Merged into UNIFIED_SPECS

---

**START WITH TUEV_UNIFIED_SPECS.md FOR IMPLEMENTATION**