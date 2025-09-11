# TUSZ Wrapper Integration Plan: SeizureTransformer + NEDC Eval

**Created**: September 9, 2025  
**Status**: 🔥 READY TO IMPLEMENT  
**Revolutionary Insight**: First to evaluate April 2025 SOTA with proper clinical metrics

---

## The Game-Changing Stack

```
SeizureTransformer (April 2025 weights)
            ↓
    Your Post-Processing 
            ↓
    NEDC Eval v6.0.0 ← THE MISSING PIECE
            ↓
Clinical Metrics (FA/24h, TAES, ATWV)
```

---

## Phase 1: Build the Wrapper Infrastructure (Days 1-2)

### Step 1: Create Evaluation Module
```python
# src/brain_go_brrr/evaluation/tusz_wrapper.py

import sys
import numpy as np
from pathlib import Path

# Add NEDC eval to path
sys.path.append('reference_repos/nedc_eeg_eval_v6.0.0/lib')
import nedc_eeg_eval_taes as taes
import nedc_eeg_eval_ovlp as ovlp
import nedc_eeg_eval_common as nec

class TUSZClinicalEvaluator:
    """
    Wrapper that bridges SeizureTransformer outputs to clinical metrics.
    This is what the April 2025 paper should have included!
    """
    
    def __init__(self):
        self.taes_scorer = taes.Taes()
        self.sensitivity_thresholds = [0.80, 0.85, 0.90, 0.95]
    
    def evaluate(self, predictions, ground_truth, duration_hours):
        """
        Compute all clinical metrics that SeizureTransformer didn't report.
        
        Args:
            predictions: List of (start, end, confidence) tuples
            ground_truth: List of (start, end) tuples  
            duration_hours: Total recording duration
            
        Returns:
            Dict with FA/24h, TAES, ATWV metrics
        """
        metrics = {}
        
        # Compute at different sensitivity thresholds
        for sens in self.sensitivity_thresholds:
            filtered_preds = self._filter_by_confidence(predictions, sens)
            fa_24h = self._compute_fa_24h(filtered_preds, ground_truth, duration_hours)
            metrics[f'fa_24h_at_{int(sens*100)}'] = fa_24h
        
        # TAES scoring (temporal alignment quality)
        metrics['taes'] = self._compute_taes(predictions, ground_truth)
        
        # ATWV (beta=999.9 for seizure detection)
        metrics['atwv'] = self._compute_atwv(predictions, ground_truth)
        
        return metrics
```

### Step 2: Create SeizureTransformer Wrapper
```python
# src/brain_go_brrr/evaluation/seizure_transformer_wrapper.py

import torch
from reference_repos.SeizureTransformer.wu_2025.src.wu_2025.architecture import SeizureTransformer

class SeizureTransformerWithMetrics:
    """
    Wraps SeizureTransformer with proper clinical evaluation.
    """
    
    def __init__(self, weights_path='data/models/pretrained/seizure_transformer_wu2025.pth'):
        self.model = self._load_model(weights_path)
        self.evaluator = TUSZClinicalEvaluator()
        self.post_processor = AdvancedPostProcessor()
    
    def predict_and_evaluate(self, eeg_data, ground_truth):
        """
        Full pipeline: Model → Post-process → Clinical Metrics
        """
        # Step 1: Get raw predictions
        raw_predictions = self.model(eeg_data)
        
        # Step 2: Apply sophisticated post-processing
        processed_predictions = self.post_processor.apply(
            raw_predictions,
            min_duration=1.0,  # seconds
            merge_gap=2.0,     # seconds
            hysteresis=(0.3, 0.7)
        )
        
        # Step 3: Compute REAL metrics (what the paper didn't do)
        metrics = self.evaluator.evaluate(
            processed_predictions,
            ground_truth,
            duration_hours=len(eeg_data) / (256 * 3600)
        )
        
        return processed_predictions, metrics
```

---

## Phase 2: Advanced Post-Processing (Day 3)

### Critical Components
```python
class AdvancedPostProcessor:
    """
    Post-processing that actually makes the difference between
    F1=0.43 (SeizureTransformer) and clinically viable FA/24h < 10.
    """
    
    def apply(self, predictions, min_duration=1.0, merge_gap=2.0, hysteresis=(0.3, 0.7)):
        """
        Three-stage post-processing pipeline.
        
        1. Hysteresis thresholding (dual threshold for stability)
        2. Gap merging (combine nearby events)
        3. Duration filtering (remove short spurious detections)
        """
        # Implementation based on Picone 2021 best practices
```

---

## Phase 3: Run Comparative Analysis (Days 4-5)

### Experiment Design
```python
# experiments/tusz_clinical_evaluation.py

def main():
    # Load TUSZ test set
    test_data = load_tusz_test_set()
    
    # Initialize models
    seizure_transformer = SeizureTransformerWithMetrics()
    
    # Run evaluation
    results = {}
    for patient in test_data:
        preds, metrics = seizure_transformer.predict_and_evaluate(
            patient.eeg_data,
            patient.annotations
        )
        results[patient.id] = metrics
    
    # Report what SeizureTransformer didn't
    print("=== Clinical Metrics (Never Before Reported) ===")
    print(f"FA/24h @ 95% sensitivity: {np.mean([r['fa_24h_at_95'] for r in results.values()])}")
    print(f"TAES score: {np.mean([r['taes'] for r in results.values()])}")
    print(f"ATWV: {np.mean([r['atwv'] for r in results.values()])}")
    
    # Compare to their reported metrics
    print("\n=== Competition Metrics (What they reported) ===")
    print(f"F1: 0.43 (from paper)")
    print(f"FP/day: 1 (but at what sensitivity?)")
```

---

## Expected Outcomes

### What We'll Likely Find
1. **FA/24h @ 95% sensitivity**: Probably > 50 (clinically unacceptable)
2. **TAES**: < 0.3 (poor temporal alignment)
3. **ATWV**: < 0.5 (below clinical viability threshold)

### The Publication Opportunity
**Title**: "Clinical Validation of SeizureTransformer: Bridging the Gap Between Competition Metrics and Medical Requirements"

**Key Findings**:
- First clinical evaluation of April 2025 SOTA
- Expose gap between F1 score and clinical metrics
- Demonstrate importance of post-processing
- Provide reproducible evaluation framework

---

## File Locations Summary

```bash
# NEDC Eval (Evaluation Metrics)
reference_repos/nedc_eeg_eval_v6.0.0/  ✅ Downloaded and extracted

# SeizureTransformer (Model)
reference_repos/SeizureTransformer/     ✅ Cloned from GitHub
data/models/pretrained/seizure_transformer_wu2025.pth  ✅ 169MB weights

# Our Integration (To Build)
src/brain_go_brrr/evaluation/          ← Create this
  ├── tusz_wrapper.py                  ← Clinical evaluator
  ├── seizure_transformer_wrapper.py   ← Model wrapper
  └── post_processing.py               ← Advanced post-processing
```

---

## Next Immediate Steps

1. ✅ NEDC Eval downloaded and in correct location
2. ⏳ Create `src/brain_go_brrr/evaluation/` module
3. ⏳ Implement TUSZClinicalEvaluator wrapper
4. ⏳ Test on sample TUSZ predictions
5. ⏳ Run full evaluation and report missing metrics

---

**This is revolutionary because NO ONE has done this for the April 2025 model yet!**