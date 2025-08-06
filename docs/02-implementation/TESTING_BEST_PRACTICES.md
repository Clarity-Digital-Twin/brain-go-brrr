# ML Testing Best Practices for Brain-Go-Brrr

## Core Principles

### 1. **DON'T MOCK ML MODELS** (Eugene Yan's Rule)
- ❌ **Bad**: Complex mocks that simulate model behavior
- ✅ **Good**: Real small data samples with real models
- **Why**: Mocks can't capture the complexity of ML models and lead to false confidence

### 2. **Test Hierarchy (Bottom to Top)**
1. **Unit Tests**: Core logic, preprocessing, data transforms
2. **Integration Tests**: Real data + real models on small samples
3. **End-to-End Tests**: Full pipeline with real data
4. **Performance Tests**: Benchmarks with real models

### 3. **What to Test vs What NOT to Test**

#### ✅ **DO Test**
- **Data preprocessing logic**: Filtering, normalization, windowing
- **Business logic**: Thresholds, classifications, aggregations
- **I/O operations**: File loading, saving, API responses
- **Error handling**: Invalid inputs, missing files, malformed data
- **Performance**: Memory usage, processing time on known data sizes

#### ❌ **DON'T Test**
- **External library behavior**: MNE, PyTorch, NumPy internals
- **Model architecture details**: Layer weights, activation functions
- **Complex ML model internals**: Use integration tests instead

### 4. **Testing Strategy by Component**

#### **Preprocessing (Unit Tests)**
```python
def test_eeg_filtering():
    # Use synthetic data
    raw_data = create_synthetic_eeg(n_channels=19, duration=30)
    filtered = apply_bandpass_filter(raw_data, low=0.5, high=50)
    assert filtered.shape == raw_data.shape
    assert np.mean(np.abs(filtered)) < np.mean(np.abs(raw_data))  # Energy reduced
```

#### **ML Models (Integration Tests with Real Data)**
```python
def test_abnormality_detection():
    # Use REAL fixtures with known labels
    normal_eeg = load_fixture("tuab_001_norm_5s.fif")
    abnormal_eeg = load_fixture("tuab_003_abnorm_5s.fif")

    detector = AbnormalityDetector()
    normal_score = detector.predict(normal_eeg)
    abnormal_score = detector.predict(abnormal_eeg)

    assert normal_score < abnormal_score  # Real behavioral test
```

#### **I/O Operations (Mock External Dependencies)**
```python
@patch('requests.post')
def test_api_upload(mock_post):
    mock_post.return_value.json.return_value = {"success": True}
    result = upload_results_to_api(test_data)
    assert result["success"] is True
```

### 5. **Fixture Strategy**

#### **Real EEG Data Fixtures** (Primary)
- Location: `tests/fixtures/eeg/`
- 9 real .fif files with known labels (normal/abnormal)
- Fast 5-second versions for unit tests
- 30-second versions for integration tests

#### **Synthetic Data** (Secondary)
- For testing edge cases (empty data, single channels, etc.)
- For performance testing (large arrays)
- Generated on-demand, not stored

#### **NO Complex Model Mocks**
- Deleted: `tests/fixtures/mock_eegpt.py` (over-engineered)
- Keep: `simple_mock_eegpt.py` for pure I/O testing only

### 6. **Test Categories and Markers**

```python
# Fast unit tests (< 1 second)
@pytest.mark.unit
def test_preprocessing_logic():
    pass

# Integration tests with real data (< 10 seconds)
@pytest.mark.integration
def test_full_pipeline():
    pass

# Slow tests with large models (> 10 seconds)
@pytest.mark.slow
def test_performance_benchmark():
    pass

# Tests requiring external dependencies
@pytest.mark.external
def test_with_sleep_edf_data():
    pass
```

### 7. **Dependency Management in Tests**

#### **Required Dependencies**
- Test should skip gracefully if optional deps missing
```python
yasa = pytest.importorskip("yasa")  # Skip if YASA not installed
```

#### **Mock External I/O Only**
```python
# ✅ Good: Mock file system, network calls
@patch('pathlib.Path.exists')
def test_file_validation(mock_exists):
    pass

# ❌ Bad: Mock model predictions
@patch('eegpt_model.predict')
def test_abnormality_detection(mock_predict):  # DON'T DO THIS
    pass
```

### 8. **Test Data Requirements**

#### **Small and Fast**
- EEG segments: 5-30 seconds max
- 19 channels max (standard 10-20 system)
- Known ground truth labels

#### **Deterministic**
- Same input always produces same output
- Use fixed random seeds where needed
- Avoid time-dependent tests

#### **Realistic**
- Real EEG data with actual artifacts
- Actual pathological patterns
- Representative sampling rates (256 Hz)

### 9. **Performance Testing**

```python
@pytest.mark.performance
def test_processing_speed():
    start_time = time.time()
    result = process_20_minute_eeg(real_eeg_data)
    duration = time.time() - start_time

    assert duration < 120  # Must complete in 2 minutes
    assert result is not None
```

### 10. **Error Cases to Test**

#### **Data Quality Issues**
- Missing channels
- Corrupted files
- Wrong sampling rates
- Too short recordings

#### **Edge Cases**
- Empty arrays
- Single-channel data
- Extreme values (artifacts)
- Memory limits

### 11. **Test Organization**

```
tests/
├── unit/           # Fast isolated tests
├── integration/    # Real data + real models
├── e2e/           # Full pipeline tests
├── benchmarks/    # Performance tests
└── fixtures/
    ├── eeg/       # Real EEG data (9 files)
    └── synthetic/ # Generated test data
```

### 12. **CI/CD Test Strategy**

#### **Every Commit (Fast Suite)**
```bash
pytest tests/unit/ -m "not slow"  # < 30 seconds total
```

#### **Pull Requests (Integration Suite)**
```bash
pytest tests/unit/ tests/integration/ -m "not slow"  # < 5 minutes
```

#### **Nightly (Full Suite)**
```bash
pytest tests/ --benchmark-only  # All tests + benchmarks
```

### 13. **Common Anti-Patterns to Avoid**

#### ❌ **Over-Mocking ML Models**
```python
# DON'T: Complex mock trying to simulate EEGPT
mock_model.extract_features.return_value = simulate_realistic_features()
```

#### ❌ **Non-Deterministic Tests**
```python
# DON'T: Tests that randomly pass/fail
score = model.predict(random_data)
assert score > 0.5  # Could fail randomly
```

#### ❌ **Testing External Libraries**
```python
# DON'T: Test that NumPy works
assert np.mean([1, 2, 3]) == 2
```

#### ❌ **Slow Unit Tests**
```python
# DON'T: Load 20GB model in unit test
def test_preprocessing():
    model = load_huge_eegpt_model()  # Takes 5 minutes
```

### 14. **Success Metrics**

- **Unit tests**: 100% pass rate, < 30 seconds total
- **Integration tests**: > 95% pass rate, < 5 minutes total
- **Test coverage**: > 80% for core business logic
- **Flakiness**: < 1% failure rate on repeated runs

---

## TL;DR - Quick Rules

1. **Use real EEG data fixtures** - not complex mocks
2. **Test business logic, not ML internals**
3. **Mock I/O, not models**
4. **Keep unit tests fast** (< 1 second each)
5. **Skip tests gracefully** if dependencies missing
6. **Use pytest markers** for test categories
7. **Delete brittle mock systems** that don't work

Following these practices ensures reliable, maintainable tests that actually catch bugs in medical EEG analysis systems.
