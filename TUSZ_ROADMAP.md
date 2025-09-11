# TUSZ Implementation Roadmap: SeizureTransformer-First Strategy

## Executive Decision Summary

After comprehensive analysis of the codebase, reference implementations, and published results, we are adopting a **SeizureTransformer-first strategy** for TUSZ temporal seizure detection.

**Key Decision Points:**
1. SeizureTransformer has **proven 87.6% AUROC on TUSZ** (Wu et al., 2025)
2. EEGPT has **no temporal detection capability** (window-level features only)
3. Model weights already downloaded: `data/models/pretrained/seizure_transformer_wu2025.pth`
4. Docker fallback available: `docker pull yujjio/seizure_transformer`

## Document Organization

### Planning & Specification Documents
- **TUSZ_SPEC.md**: Clinical requirements, metrics, dual-model architecture theory
- **TUSZ_IMPLEMENTATION.md**: Detailed implementation patterns, code examples, API design

### This Document (TUSZ_ROADMAP.md)
- **Purpose**: Concrete execution plan for SeizureTransformer wrapper
- **Scope**: Next 4-6 weeks of development
- **Focus**: Getting to production-ready TUSZ detection

## Phase 1: SeizureTransformer Wrapper (Weeks 1-2)

### Week 1: Core Integration
```python
# Location: src/brain_go_brrr/infra/ml_models/seizure_transformer_wrapper.py

class SeizureTransformerWrapper:
    """Production wrapper for Wu 2025 SeizureTransformer model"""
    
    def __init__(self, checkpoint_path: Path):
        self.model = self._load_checkpoint(checkpoint_path)
        self.preprocessor = TUSZPreprocessor()
        self.postprocessor = SeizurePostProcessor()
    
    def predict_timestep(self, eeg: np.ndarray) -> np.ndarray:
        """Direct time-step level predictions"""
        # 1. Preprocess to 256Hz, 19 channels
        # 2. Run inference 
        # 3. Return per-timestep probabilities
```

**Deliverables:**
- [ ] Load `seizure_transformer_wu2025.pth` checkpoint
- [ ] Implement 19-channel TCP montage mapping
- [ ] 256Hz resampling pipeline
- [ ] Basic inference working on sample data

### Week 2: NEDC Evaluation Integration
```python
# Location: src/brain_go_brrr/infra/evaluation/nedc_evaluator.py

class NEDCClinicalEvaluator:
    """NEDC Eval v6.0.0 compliant metrics"""
    
    def compute_fa_per_24h(self, predictions, labels, sensitivity=0.90):
        """Clinical gold standard metric"""
        
    def compute_taes(self, predictions, labels):
        """Time-aligned event scoring"""
        
    def compute_atwv(self, predictions, labels, beta=0.1):
        """Actual term-weighted value"""
```

**Deliverables:**
- [ ] Port NEDC Eval v6.0.0 scoring functions
- [ ] Validate against reference implementation
- [ ] Generate clinical reports at multiple sensitivities
- [ ] Interim baseline: ≤10 FA/24h at 85% sensitivity (final target ≤5 at 90%)

## Phase 2: TUSZ Dataset Pipeline (Week 3)

### Dataset Integration
```python
# Location: src/brain_go_brrr/infra/data/tusz_dataset.py

class TUSZDataset(EEGDataset):
    """TUSZ v2.0.1 dataset with official splits"""
    
    OFFICIAL_SPLITS = {
        'train': '01_tcp_ar',  # 80% 
        'dev': '02_tcp_ar',    # 10%
        'test': '03_tcp_ar'    # 10% (never touch during development)
    }
    
    def __init__(self, split='train'):
        # Use official NEDC splits
        # Patient-level separation (no leakage)
        # Handle class imbalance (2% seizure, 98% background)
```

**Deliverables:**
- [ ] Download TUSZ v2.0.1 (60GB compressed)
- [ ] Implement efficient data loader with caching
- [ ] Validate channel mappings and sampling rates
- [ ] Test on dev set only (never test set until final)

## Phase 3: Production Optimization (Week 4)

### Performance Targets
```python
# Real-time constraint: Process 1 hour in < 15 minutes (4x speedup)
# Memory constraint: < 8GB for edge deployment

optimization_strategies = {
    'quantization': 'INT8 for 2x speedup',
    'batching': 'Process multiple 60s windows in parallel',
    'caching': 'Pre-compute frequently used features',
    'onnx': 'Export for cross-platform deployment'
}
```

**Deliverables:**
- [ ] Benchmark inference speed (target: <4s per hour)
- [ ] Implement batch processing
- [ ] Memory profiling and optimization
- [ ] ONNX export for deployment flexibility

## Phase 4: Clinical Validation (Weeks 5-6)

### Robustness Testing
```python
# Test against various conditions:
test_scenarios = {
    'artifacts': 'Movement, electrode pop, line noise',
    'montages': 'Different channel configurations', 
    'sampling_rates': '128Hz, 256Hz, 512Hz, 1000Hz',
    'patient_variability': 'Age groups, seizure types'
}
```

**Deliverables:**
- [ ] Cross-dataset validation (if available)
- [ ] Failure mode analysis
- [ ] Generate clinical reports
- [ ] Document limitations and edge cases

## Success Metrics

### Minimum Viable Product (Week 2)
- [x] Model weights loaded: `seizure_transformer_wu2025.pth`
- [ ] Basic inference working
- [ ] AUROC ≥ 0.85 on dev set
- [ ] FA/24h ≤ 10 at 85% sensitivity

### Production Ready (Week 4)
- [ ] FA/24h ≤ 5 at 90% sensitivity  
- [ ] ATWV ≥ 0.40
- [ ] Real-time processing (4x faster than recording)
- [ ] Clinical interpretability features

### Gold Standard (Week 6)
- [ ] Match published performance (0.876 AUROC)
- [ ] FA/24h ≤ 1 at 95% sensitivity
- [ ] Patient-specific adaptation ready
- [ ] Full NEDC compliance

## Future Research Directions (Post-MVP)

### EEGPT Enhancement Experiments
Once SeizureTransformer baseline is stable, explore:

```python
# EXPERIMENTAL - Not for initial implementation
class HybridSeizureDetector:
    """Combine SeizureTransformer with EEGPT features"""
    
    def __init__(self):
        self.seizure_transformer = SeizureTransformerWrapper()
        self.eegpt = EEGPTWrapper()  # Our existing 83% AUROC model
        
    def detect_with_context(self, eeg):
        # 1. Get timestep predictions from SeizureTransformer
        # 2. Extract EEGPT features for suspicious regions
        # 3. Ensemble or use EEGPT for confidence calibration
        # 4. Research only - not for production timeline
```

### Why EEGPT+BiLSTM is a Future Research Project
- No proven architecture for this combination
- Would require 3-6 months of experimentation
- High risk of suboptimal performance
- Better as PhD research than production deployment

## Risk Mitigation

### Primary Risks & Mitigations
1. **Model doesn't load**: Use Docker container as fallback
2. **Performance below target**: Already have proven baseline to match
3. **Speed too slow**: Pre-implement optimization strategies
4. **Integration issues**: Well-documented architecture, clear interfaces

## Implementation Notes

### Current Status
- ✅ Model weights downloaded: `data/models/pretrained/seizure_transformer_wu2025.pth`
- ✅ Architecture decision made: SeizureTransformer-first
- ✅ Documentation complete: SPEC, IMPLEMENTATION, ROADMAP
- ⏳ Awaiting senior approval to begin implementation

### Next Immediate Steps
1. Create `src/brain_go_brrr/infra/ml_models/seizure_transformer_wrapper.py`
2. Load checkpoint and verify model architecture
3. Run inference on sample TUSZ data
4. Compare outputs with published results

## References

1. **SeizureTransformer**: Wu et al., 2025 - arXiv:2504.00336
2. **TUSZ Dataset**: v2.0.1 from Temple University Hospital
3. **NEDC Eval**: v6.0.0 evaluation framework
4. **Our TUAB Success**: 83% AUROC with EEGPT (proven feature extraction)

---

**Document Status**: READY FOR SENIOR REVIEW
**Strategy**: SeizureTransformer wrapper → NEDC validation → Production optimization
**Timeline**: 4-6 weeks to production-ready
**Risk Level**: LOW (following proven architecture)
