# MNE-Python Integration Strategy for Brain-Go-Brrr

## Executive Summary

This document outlines a comprehensive strategy to integrate MNE-Python's advanced preprocessing and analysis capabilities into the Brain-Go-Brrr training pipeline, with the goal of improving TUAB abnormality detection accuracy from ~56% to the target 87% AUROC.

## Current State Analysis

### What We Have
- **MNE Usage**: Limited to basic I/O and preprocessing in application layer
- **Training Pipeline**: Direct NumPy/PyTorch loading without MNE preprocessing
- **Accuracy**: ~56% on TUAB abnormality detection (target: 87%)
- **Autoreject**: Integrated but not used in training data preparation

### Gap Analysis
1. **No artifact rejection in training data** - potentially training on noisy samples
2. **No advanced filtering** - missing zero-phase, adaptive filters
3. **No data augmentation** - limited training diversity
4. **No spectral features** - missing frequency domain information
5. **No quality metrics** - training on all data regardless of quality

## Integration Objectives

### Primary Goals
1. **Improve Training Data Quality**: Use MNE's artifact detection to clean training data
2. **Enhance Feature Extraction**: Add spectral and connectivity features
3. **Implement Data Augmentation**: Increase training diversity
4. **Standardize Preprocessing**: Consistent pipeline across training and inference

### Success Metrics
- AUROC improvement from 56% → 87%
- Training stability (no NaN losses)
- Faster convergence (fewer epochs needed)
- Better generalization (validation performance)

## Technical Integration Plan

### Phase 1: Data Quality Enhancement (Weeks 1-2)

#### 1.1 Artifact Detection Pipeline
```python
# New preprocessing module: experiments/eegpt_linear_probe/preprocessing/mne_pipeline.py
- annotate_muscle_zscore() - detect muscle artifacts
- annotate_movement() - detect movement artifacts  
- annotate_bad_channels() - automatic bad channel detection
- compute_psd() - power spectral density for quality metrics
```

#### 1.2 Quality Scoring System
```python
# Quality metrics for each EEG segment:
- SNR (Signal-to-Noise Ratio)
- Artifact contamination percentage
- Channel correlation matrix
- Spectral entropy
```

#### 1.3 Integration with TUAB Dataset
- Modify `TUABMemoryMappedDataset` to include MNE preprocessing
- Cache preprocessed data with quality scores
- Filter training samples by quality threshold

### Phase 2: Advanced Preprocessing (Weeks 2-3)

#### 2.1 Filtering Enhancements
```python
# MNE filtering advantages:
- Zero-phase filters (no temporal distortion)
- Adaptive notch filters (power line removal)
- Optimal FIR/IIR selection
- Transition band optimization
```

#### 2.2 Reference Schemes
```python
# Implement multiple referencing options:
- Common Average Reference (CAR)
- Laplacian referencing (spatial filtering)
- Bipolar montages for specific analyses
- REST reference (Reference Electrode Standardization)
```

#### 2.3 Baseline Correction
- Implement proper baseline correction for epochs
- Z-score normalization with robust statistics
- Detrending for drift removal

### Phase 3: Feature Engineering (Weeks 3-4)

#### 3.1 Spectral Features
```python
# Complement EEGPT embeddings with:
- Band power (delta, theta, alpha, beta, gamma)
- Spectral edge frequency
- Peak frequency per band
- Spectral entropy
- Power ratios (e.g., theta/beta)
```

#### 3.2 Connectivity Features
```python
# Inter-channel relationships:
- Phase-locking value (PLV)
- Coherence matrices
- Mutual information
- Graph theory metrics
```

#### 3.3 Time-Frequency Features
```python
# Wavelet and multitaper analysis:
- Morlet wavelet decomposition
- Multitaper spectrograms
- Event-related spectral perturbation (ERSP)
```

### Phase 4: Data Augmentation (Week 4)

#### 4.1 MNE-Based Augmentations
```python
# Training data diversity:
- Temporal shifts (circular shift)
- Amplitude scaling (within physiological range)
- Channel dropout (simulate missing channels)
- Noise injection (colored noise matching EEG spectrum)
- Mixup between samples of same class
```

#### 4.2 Synthetic Data Generation
```python
# Generate synthetic pathological patterns:
- Spike-and-wave complexes
- Slowing patterns
- Asymmetric activity
- Based on MNE's simulation module
```

## Implementation Architecture

### Module Structure
```
experiments/eegpt_linear_probe/
├── mne_integration/
│   ├── __init__.py
│   ├── preprocessing.py      # Core MNE preprocessing pipeline
│   ├── quality_metrics.py    # Data quality scoring
│   ├── feature_extraction.py # Spectral/connectivity features
│   ├── augmentation.py       # Data augmentation strategies
│   └── artifact_rejection.py # Advanced artifact detection
├── datasets/
│   └── tuab_mne_dataset.py  # MNE-enhanced TUAB dataset
└── configs/
    └── mne_tuab_config.yaml  # MNE preprocessing parameters
```

### Integration Points

1. **Dataset Level**
   - Subclass `TUABMemoryMappedDataset` → `TUABMNEDataset`
   - Add MNE preprocessing in `__getitem__()`
   - Cache preprocessed segments

2. **Training Script**
   - Add `--use-mne-preprocessing` flag
   - Include quality-based sample weighting
   - Log preprocessing metrics

3. **Configuration**
   - MNE-specific parameters in YAML
   - Preprocessing profiles (minimal, standard, aggressive)
   - Quality thresholds

## MNE-Autoreject Synergy

### Combined Pipeline
```python
# Optimal artifact rejection flow:
1. MNE annotate_muscle_zscore() → detect muscle artifacts
2. MNE annotate_movement() → detect movement
3. Autoreject local → channel-specific thresholds
4. MNE interpolate_bads() → repair bad channels
5. Quality scoring → filter training data
```

### Benefits
- Autoreject handles channel-specific artifacts
- MNE handles global artifacts (muscle, movement)
- Combined approach more robust than either alone

## Risk Mitigation

### Potential Issues & Solutions

1. **Processing Speed**
   - Risk: MNE preprocessing slows training
   - Solution: Parallel processing, caching, GPU acceleration where possible

2. **Memory Usage**
   - Risk: MNE objects consume more RAM
   - Solution: Lazy loading, memory mapping, batch processing

3. **Compatibility**
   - Risk: MNE version conflicts
   - Solution: Pin versions, comprehensive testing

4. **Overfitting**
   - Risk: Too much preprocessing removes important patterns
   - Solution: Validation on held-out data, ablation studies

## Validation Strategy

### A/B Testing
- Train parallel models with/without MNE preprocessing
- Compare on same validation set
- Track metrics: AUROC, sensitivity, specificity

### Ablation Studies
- Test each preprocessing step independently
- Measure contribution to accuracy
- Identify optimal combination

### Cross-Dataset Validation
- Test on TUEV dataset
- Ensure preprocessing generalizes
- Avoid dataset-specific overfitting

## Timeline & Milestones

### Week 1-2: Foundation
- [ ] Implement quality metrics
- [ ] Basic artifact detection
- [ ] Integration with dataset

### Week 2-3: Advanced Features  
- [ ] Spectral feature extraction
- [ ] Advanced filtering
- [ ] Reference schemes

### Week 3-4: Augmentation & Testing
- [ ] Data augmentation pipeline
- [ ] A/B testing framework
- [ ] Performance benchmarking

### Week 4-5: Optimization & Documentation
- [ ] Hyperparameter tuning
- [ ] Final performance evaluation
- [ ] Documentation and deployment

## Expected Outcomes

### Performance Improvements
- **Accuracy**: 56% → 75-87% AUROC
- **Training Speed**: 20-30% faster convergence
- **Stability**: Elimination of NaN losses
- **Generalization**: Better cross-dataset performance

### Technical Benefits
- Standardized preprocessing pipeline
- Reproducible results
- Better interpretability
- Production-ready code

## Conclusion

Integrating MNE-Python's advanced capabilities addresses the current gaps in our training pipeline. By improving data quality, adding complementary features, and implementing augmentation, we expect significant accuracy improvements toward the 87% target.

The phased approach allows iterative validation and risk mitigation while maintaining backward compatibility. This integration will establish Brain-Go-Brrr as a state-of-the-art EEG analysis platform.

## References

1. Gramfort et al. (2013). MEG and EEG data analysis with MNE-Python
2. Jas et al. (2017). Autoreject: Automated artifact rejection for MEG and EEG
3. EEGPT Paper - Target performance metrics
4. MNE-Python Documentation v1.7.0

---

*Document prepared for external auditor review*  
*Last updated: August 25, 2025*