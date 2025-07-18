---
name: Enhanced Autoreject with Bayesian Optimization
about: Implement Bayesian optimization for artifact rejection thresholds
title: 'feat: Add Bayesian optimization to Autoreject for adaptive thresholds'
labels: enhancement, machine-learning
assignees: ''
---

## Problem Statement
Current Autoreject implementation uses fixed thresholds for artifact detection. We need adaptive thresholds using Bayesian optimization to improve artifact rejection accuracy across diverse EEG recordings.

## Requirements
1. Implement Bayesian optimization for threshold tuning
2. Create cross-validation framework for threshold selection
3. Add channel-specific threshold adaptation
4. Support different optimization objectives (precision, recall, F1)
5. Cache optimized thresholds for similar recordings

## Technical Approach
```python
# Leverage scikit-optimize for Bayesian optimization
from skopt import BayesSearchCV
from skopt.space import Real

# Define search space for thresholds
search_spaces = {
    'threshold_muscle': Real(0.5, 5.0, prior='log-uniform'),
    'threshold_eog': Real(0.3, 3.0, prior='log-uniform'),
    'threshold_channel': Real(0.1, 2.0, prior='log-uniform')
}
```

## Implementation Details
- Extend `services/qc_flagger.py` with `BayesianAutoReject` class
- Use reference data from `/reference_repos/autoreject/` for validation
- Implement parallel threshold search using joblib
- Add metrics tracking for optimization history
- Create visualization of threshold convergence

## Acceptance Criteria
- [ ] Bayesian optimization improves artifact detection F1 score by >5%
- [ ] Optimization completes within 60 seconds for 20-minute EEG
- [ ] Channel-specific thresholds show meaningful variation
- [ ] Comprehensive tests with mocked optimization
- [ ] Documentation includes threshold tuning guide

@claude please implement Bayesian optimization for Autoreject thresholds following TDD principles. Reference the autoreject paper in `/reference_repos/autoreject/` for the mathematical framework. Start with tests for the optimization pipeline.
