# AutoReject Ablation Study Plan

## Key Question: Will AutoReject Help or Hurt AUROC?

### The Trade-off

**Potential Benefits:**
- Removes high-amplitude muscle/movement artifacts that inflate variance
- Keeps good sensors via interpolation instead of dropping whole windows
- Should improve signal-to-noise ratio for abnormality detection

**Potential Risks:**
- May mark epileptiform spikes as "artifacts" and interpolate/drop them
- Reduced effective sample count from dropped segments
- Could mask clinically important events if too aggressive

### Why EEGPT Authors Skipped It

1. **Focus on transfer learning** - Wanted clean attribution of gains to the model
2. **Dataset consistency** - TUAB already had basic QC; AutoReject would complicate reproducibility
3. **Compute budget** - Cross-validated threshold search for 1.4M windows is expensive

## Recommended Experiment Plan

### 1. Dry Run on Validation Set
- Log each segment: kept/interpolated/dropped
- Target: ≤10% windows discarded (else too aggressive)

### 2. Ablation Study

| Run | use_autoreject | Expected AUROC | Windows Kept | % Kept |
|-----|----------------|----------------|--------------|--------|
| A | false | 0.789 (baseline) | 930,495 | 100% |
| B | true (default) | ≥0.80? | ~837,000 | ~90% |
| C | true (conservative: consensus=0.05) | ? | ~884,000 | ~95% |

### 3. Clinical Safety Check
- Visualize 50 random windows that AutoReject repairs/drops
- Verify no epileptiform events are being removed
- If clinical events removed → reduce thresholds or use repair-only mode

### 4. Implementation Safeguards
- Save binary mask with every cleaned file for reconstruction
- Track "fraction bad channels" metric per batch
- Alert if rejection rate jumps above baseline

## Implementation Status

Current baseline (no AutoReject): **0.789 AUROC**

Next steps after baseline completes:
1. Enable `data.use_autoreject: true`
2. Monitor keep/repair/drop rates
3. Compare AUROC vs baseline
4. Visualize rejected segments for clinical review

## Key Metrics to Track

```python
# Per batch:
- n_windows_total
- n_windows_kept
- n_windows_interpolated  
- n_windows_dropped
- mean_bad_channels_per_window
- max_bad_channels_per_window

# Per epoch:
- auroc_with_ar vs auroc_without_ar
- class_balance_shift (normal vs abnormal retention rates)
```

## Decision Criteria

**Use AutoReject if:**
- AUROC improves by ≥0.01 (0.789 → 0.799+)
- Window drop rate <15%
- No systematic removal of epileptiform events
- Class balance remains within 2% of original

**Skip AutoReject if:**
- AUROC drops or no improvement
- >15% windows dropped
- Clinical events being removed
- Significant class imbalance introduced

## Notes

- AutoReject is just another hyperparameter to tune
- Conservative thresholds (consensus=0.05) may be safer for clinical data
- Always maintain ability to reconstruct original signal
- Consider "repair-only" mode if drops are problematic