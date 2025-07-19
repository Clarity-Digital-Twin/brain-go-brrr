# EEGPT + YASA Integration Plan

## Current State

- YASA and EEGPT run as separate, parallel services
- No feature sharing between models
- Missing the core vision: EEGPT as universal feature extractor

## Target Architecture

```
Raw EEG → Preprocessing → EEGPT Encoder → 512-dim Embeddings
                                              ↓
                                    ┌─────────┴──────────┐
                                    │                    │
                              YASA + Embeddings    Abnormality Head
                               (Enhanced Sleep)     (Classification)
```

## TDD Implementation Plan

### Phase 1: Add Smoothing to YASA ✅

1. ✅ Write tests for temporal smoothing
2. ⏳ Implement `_smooth_hypnogram()` method
3. ⏳ Add `apply_smoothing` parameter to `stage_sleep()`
4. ⏳ Run tests and verify

### Phase 2: Create Unified Feature Extractor

1. Write tests for `EEGPTFeatureExtractor` service
   - Test embedding extraction for various window sizes
   - Test caching of embeddings
   - Test sharing embeddings across services
2. Implement feature extractor service
3. Integrate with existing pipeline

### Phase 3: Enhance YASA with EEGPT Features

1. Write tests for enhanced sleep staging
   - Test YASA with EEGPT embeddings
   - Test performance improvement
   - Test fallback to YASA-only mode
2. Modify `SleepAnalyzer` to accept embeddings
3. Create feature fusion layer

### Phase 4: Create Integrated Pipeline

1. Write integration tests for full pipeline
   - Test end-to-end processing
   - Test cross-task information flow
   - Test performance benchmarks
2. Implement `IntegratedEEGPipeline` class
3. Update existing services to use shared features

### Phase 5: Validate with Real Data

1. Test on Sleep-EDF dataset
2. Benchmark performance vs standalone services
3. Document improvements

## Key Integration Points

### 1. Feature Extraction Service

```python
class EEGPTFeatureExtractor:
    """Centralized EEGPT feature extraction."""

    def extract_embeddings(self, raw: mne.io.Raw) -> np.ndarray:
        """Extract 512-dim embeddings for all windows."""
        # Returns: (n_windows, 512) embeddings
```

### 2. Enhanced Sleep Analyzer

```python
class SleepAnalyzer:
    def stage_sleep(self, raw, embeddings=None):
        """Stage sleep with optional EEGPT embeddings."""
        if embeddings is not None:
            # Combine YASA features with EEGPT embeddings
            features = self._combine_features(yasa_features, embeddings)
```

### 3. Integrated Pipeline

```python
class IntegratedEEGPipeline:
    """Unified pipeline with feature sharing."""

    def process(self, raw):
        # Extract features once
        embeddings = self.feature_extractor.extract_embeddings(raw)

        # Share across all tasks
        sleep_results = self.sleep_analyzer.stage_sleep(raw, embeddings)
        abnormal_results = self.abnormal_detector.detect(embeddings)

        # Cross-task enhancement
        return self._integrate_results(sleep_results, abnormal_results)
```

## Success Metrics

- [ ] EEGPT embeddings extracted once and shared
- [ ] YASA can optionally use EEGPT features
- [ ] Integration tests pass with real data
- [ ] Performance improvement documented
- [ ] No orphan code - everything connected

## Next Step

Complete Phase 1 (smoothing), then move to Phase 2 (unified feature extractor)
