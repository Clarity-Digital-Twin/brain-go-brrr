# 🚀 Brain-Go-Brrr Implementation Plan

## 📚 Literature-Informed Development Strategy

This plan strategically incorporates insights from our literature repository to build a production-ready EEG analysis system.

## 🎯 Phase 1: MVP - Auto-QC + Risk Flagger (Week 1)

### Day 1-2: EEGPT Integration Foundation
**Literature Reference**: EEGPT paper (page 3-5 figures)
- [ ] Create EEGPT model loader in `src/brain_go_brrr/models/eegpt_model.py`
  - Load checkpoint from `/data/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt`
  - Implement preprocessing: 256Hz resampling, 4-second windowing
  - Handle channel mapping with adaptive spatial filter
- [ ] Write comprehensive tests for model loading and preprocessing
- [ ] Verify model outputs match expected dimensions (1024 samples → features)

### Day 3-4: Abnormal Detection Pipeline
**Literature Reference**: Abnormal-EEG paper (94.63% accuracy benchmark)
- [ ] Implement abnormality detection in `services/abnormal_detector.py`
  - Window-based feature extraction using EEGPT
  - Aggregate predictions across windows
  - Target: >80% balanced accuracy
- [ ] Update `qc_flagger.py` to use real EEGPT inference
- [ ] Add confidence scoring and clinical thresholds

### Day 5: FastAPI Endpoint
**Literature Reference**: ROUGH_DRAFT.md MVP specification
- [ ] Create `/api/v1/eeg/analyze` endpoint
- [ ] Return JSON:
  ```json
  {
    "bad_channels": ["T3", "O2"],
    "bad_pct": 21,
    "abnormal_prob": 0.83,
    "flag": "Expedite read",
    "confidence": 0.92,
    "processing_time": 12.3
  }
  ```
- [ ] Add PDF report generation with matplotlib

## 🧪 Phase 2: Enhanced Quality Control (Week 2)

### Day 6-7: Autoreject Integration
**Literature Reference**: Autoreject paper figures (page 22-27)
- [ ] Fix channel position handling for diverse datasets
- [ ] Implement Bayesian optimization for thresholds
- [ ] Create artifact rejection pipeline
- [ ] Target: Match human expert agreement (87.5%)

### Day 8-9: Sleep Analysis Enhancement
**Literature Reference**: YASA paper (87.46% accuracy)
- [ ] Fix hypnogram generation issues
- [ ] Add sleep event detection (spindles, slow waves)
- [ ] Implement comprehensive sleep metrics
- [ ] Create sleep quality scoring

### Day 10: Integration Testing
- [ ] End-to-end pipeline tests
- [ ] Performance benchmarking
- [ ] Clinical validation with test datasets

## 🔧 Phase 3: Production Readiness (Week 3)

### Day 11-12: Code Quality Fixes
**Reference**: "Bad-hacky-shit" audit findings
- [ ] Replace home-rolled IoC container with proper DI
- [ ] Remove hardcoded paths, use configuration
- [ ] Fix type annotations and mypy errors
- [ ] Achieve 80% test coverage

### Day 13-14: Event Detection
**Literature Reference**: Epileptiform discharge paper
- [ ] Implement spike/sharp wave detection
- [ ] Add GPED/PLED pattern recognition
- [ ] Create time-stamped event list
- [ ] Integrate with main pipeline

### Day 15: Deployment Preparation
- [ ] Docker containerization
- [ ] CI/CD pipeline setup
- [ ] Documentation and API specs
- [ ] Performance optimization

## 📊 Key Performance Targets

Based on literature benchmarks:
- **Abnormal Detection**: >80% balanced accuracy (BioSerenity-E1: 94.63%)
- **Sleep Staging**: >85% accuracy (YASA: 87.46%)
- **Artifact Rejection**: >85% expert agreement (Autoreject: 87.5%)
- **Processing Speed**: <2 min for 20-min EEG
- **API Response**: <100ms latency

## 🔍 Strategic Literature Checkpoints

### Before Each Major Component:
1. **EEGPT Implementation** → Review figures 3-5 for architecture
2. **Abnormal Detection** → Check performance tables (page 22)
3. **Sleep Analysis** → Reference YASA confusion matrices
4. **Event Detection** → Study epileptiform patterns
5. **Quality Control** → Autoreject validation curves

### Reference Repos Usage:
- `reference_repos/EEGPT/` - Model architecture and training code
- `reference_repos/autoreject/` - Artifact rejection algorithms
- `reference_repos/yasa/` - Sleep staging implementation
- `reference_repos/mne-python/` - EEG preprocessing utilities

## 🚦 Success Criteria

### MVP (Week 1):
- ✅ EEGPT model loads and runs inference
- ✅ Abnormality detection achieves >80% accuracy
- ✅ API endpoint returns proper JSON/PDF
- ✅ Processing time <30 seconds

### Full System (Week 3):
- ✅ All modules integrated and tested
- ✅ Meets performance benchmarks
- ✅ Clean code with 80% coverage
- ✅ Production-ready with documentation

## 📝 Daily Workflow

1. **Morning**: Review relevant literature section
2. **Coding**: TDD approach - write tests first
3. **Integration**: Test with Sleep-EDF data
4. **Review**: Check against paper benchmarks
5. **Commit**: Update progress in PROJECT_STATUS.md

## 🎨 Architecture Decisions

Based on literature and current codebase:
- Use EEGPT as feature extractor (frozen weights)
- Linear probing for task-specific heads
- Service-oriented architecture (not microservices)
- PostgreSQL + TimescaleDB for time-series data
- Redis for job queuing

## 🔗 Next Steps

1. Start with `test_eegpt_integration.py` (TDD)
2. Implement EEGPT model loader
3. Update QC flagger with real inference
4. Create FastAPI endpoint
5. Iterate based on test results

Remember: This is medical-adjacent software. Accuracy matters. Test everything.
