# Test-Driven Development (TDD) Specifications for Brain-Go-Brrr

## Executive Summary

This document provides comprehensive Test-Driven Development specifications for the Brain-Go-Brrr EEG analysis pipeline. Following the Red-Green-Refactor cycle, we define test specifications for each component BEFORE implementation, ensuring 100% test coverage and robust quality assurance.

## Core TDD Principles

1. **Red**: Write a failing test first
2. **Green**: Write minimal code to pass the test
3. **Refactor**: Improve code quality while keeping tests green
4. **Coverage**: Maintain >95% test coverage across all modules
5. **Fast**: Tests must run in <30 seconds for rapid iteration

## Test Categories

### 1. Unit Tests
- Isolated component testing
- Mock external dependencies
- Sub-second execution time
- Located in `tests/unit/`

### 2. Integration Tests
- Component interaction testing
- Real dependencies where possible
- Located in `tests/integration/`

### 3. End-to-End Tests
- Full pipeline validation
- Real EEG data processing
- Located in `tests/e2e/`

### 4. Performance Tests
- Benchmark critical operations
- Memory usage profiling
- Located in `tests/performance/`

## Module-Specific Test Specifications

### 1. EEGPT Model Tests

#### 1.1 Model Loading Tests
```python
# tests/unit/test_eegpt_model.py

def test_eegpt_loads_from_checkpoint():
    """Test EEGPT model loads successfully from checkpoint."""
    # Given: Valid checkpoint path
    checkpoint_path = Path("data/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt")
    
    # When: Loading model
    model = EEGPTModel.from_checkpoint(checkpoint_path)
    
    # Then: Model has expected architecture
    assert model.num_parameters() == 10_417_024
    assert model.patch_size == 64
    assert model.window_samples == 2048  # 8s @ 256Hz

def test_eegpt_handles_missing_checkpoint():
    """Test graceful failure when checkpoint missing."""
    # Given: Invalid checkpoint path
    checkpoint_path = Path("nonexistent.ckpt")
    
    # When/Then: Should raise FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        EEGPTModel.from_checkpoint(checkpoint_path)

def test_eegpt_channel_mapping():
    """Test automatic channel name conversion (old->modern)."""
    # Given: EEG data with old naming
    raw_data = create_mock_raw(channels=["T3", "T4", "T5", "T6"])
    
    # When: Processing through model
    model = EEGPTModel()
    processed = model.preprocess(raw_data)
    
    # Then: Channels renamed to modern convention
    assert "T7" in processed.ch_names
    assert "T8" in processed.ch_names
    assert "P7" in processed.ch_names
    assert "P8" in processed.ch_names
```

#### 1.2 Feature Extraction Tests
```python
def test_eegpt_extracts_features():
    """Test EEGPT extracts meaningful features from EEG."""
    # Given: 8-second EEG window
    eeg_data = create_synthetic_eeg(duration=8.0, sfreq=256, n_channels=20)
    
    # When: Extracting features
    model = EEGPTModel()
    features = model.extract_features(eeg_data)
    
    # Then: Features have expected shape
    assert features.shape == (1, 768)  # Transformer embedding dimension
    assert not torch.isnan(features).any()
    assert features.abs().mean() > 0.01  # Non-trivial features

def test_eegpt_batch_processing():
    """Test EEGPT processes multiple windows efficiently."""
    # Given: Multiple EEG windows
    batch_size = 32
    windows = [create_synthetic_eeg(8.0, 256, 20) for _ in range(batch_size)]
    
    # When: Batch processing
    model = EEGPTModel()
    start_time = time.time()
    features = model.extract_features_batch(windows)
    elapsed = time.time() - start_time
    
    # Then: Efficient batch processing
    assert features.shape == (batch_size, 768)
    assert elapsed < 5.0  # Process 32 windows in <5 seconds
```

### 2. AutoReject Integration Tests

#### 2.1 Bad Channel Detection Tests
```python
# tests/unit/test_autoreject.py

def test_autoreject_detects_bad_channels():
    """Test AutoReject identifies bad channels correctly."""
    # Given: EEG with known bad channels
    raw = create_mock_raw_with_artifacts(
        bad_channels=["Fp1", "O2"],
        artifact_amplitude=500e-6  # 500 µV
    )
    
    # When: Running AutoReject
    ar = AutoRejectWrapper()
    cleaned, bad_channels = ar.process(raw)
    
    # Then: Bad channels detected
    assert "Fp1" in bad_channels
    assert "O2" in bad_channels
    assert len(bad_channels) == 2

def test_autoreject_repairs_bad_epochs():
    """Test AutoReject repairs bad epochs via interpolation."""
    # Given: Epoched data with artifacts
    epochs = create_epochs_with_artifacts(n_epochs=100, n_bad=10)
    
    # When: Processing epochs
    ar = AutoRejectWrapper()
    cleaned_epochs = ar.repair_epochs(epochs)
    
    # Then: Bad epochs repaired
    assert len(cleaned_epochs) >= 90  # At most 10% rejected
    assert cleaned_epochs.get_data().max() < 200e-6  # Artifacts removed

def test_autoreject_performance_target():
    """Test AutoReject achieves 87.5% agreement with experts."""
    # Given: Expert-annotated test set
    test_data = load_expert_annotated_eeg()
    
    # When: Running AutoReject
    ar = AutoRejectWrapper()
    predictions = ar.predict_artifacts(test_data)
    
    # Then: High agreement with experts
    agreement = calculate_agreement(predictions, test_data.expert_labels)
    assert agreement >= 0.875  # 87.5% target from literature
```

### 3. Abnormality Detection Tests

#### 3.1 Binary Classification Tests
```python
# tests/unit/test_abnormal_detection.py

def test_abnormal_detection_accuracy():
    """Test abnormal detection meets performance targets."""
    # Given: TUAB test set
    test_loader = create_tuab_test_loader()
    
    # When: Running inference
    detector = AbnormalityDetector()
    predictions = []
    labels = []
    
    for batch in test_loader:
        pred = detector.predict(batch.eeg_data)
        predictions.extend(pred)
        labels.extend(batch.labels)
    
    # Then: Meets performance targets
    accuracy = balanced_accuracy_score(labels, predictions)
    auroc = roc_auc_score(labels, predictions)
    
    assert accuracy >= 0.80  # 80% balanced accuracy target
    assert auroc >= 0.93     # AUROC ≥ 0.93 target

def test_abnormal_detection_confidence():
    """Test confidence scores are well-calibrated."""
    # Given: Mix of normal and abnormal EEG
    normal_eeg = load_normal_eeg_sample()
    abnormal_eeg = load_abnormal_eeg_sample()
    
    # When: Getting confidence scores
    detector = AbnormalityDetector()
    normal_conf = detector.predict_proba(normal_eeg)[0, 0]  # P(normal)
    abnormal_conf = detector.predict_proba(abnormal_eeg)[0, 1]  # P(abnormal)
    
    # Then: High confidence for clear cases
    assert normal_conf > 0.8
    assert abnormal_conf > 0.8
```

#### 3.2 Triage Priority Tests
```python
def test_triage_priority_assignment():
    """Test correct triage priority assignment."""
    # Given: EEG samples with varying abnormality levels
    test_cases = [
        (create_normal_eeg(), "routine"),
        (create_mildly_abnormal_eeg(), "expedite"),
        (create_severely_abnormal_eeg(), "urgent")
    ]
    
    # When/Then: Correct triage assignment
    detector = AbnormalityDetector()
    for eeg, expected_priority in test_cases:
        priority = detector.assign_triage_priority(eeg)
        assert priority == expected_priority

def test_triage_thresholds():
    """Test triage thresholds are clinically appropriate."""
    # Given: Detector with configured thresholds
    detector = AbnormalityDetector(
        routine_threshold=0.3,
        expedite_threshold=0.7,
        urgent_threshold=0.9
    )
    
    # When: Testing threshold behavior
    # Then: Monotonic priority assignment
    for conf in [0.2, 0.5, 0.8, 0.95]:
        priority = detector._confidence_to_priority(conf)
        if conf < 0.3:
            assert priority == "routine"
        elif conf < 0.7:
            assert priority == "expedite"
        else:
            assert priority == "urgent"
```

### 4. Sleep Analysis Tests

#### 4.1 YASA Integration Tests
```python
# tests/unit/test_sleep_analysis.py

def test_yasa_sleep_staging():
    """Test YASA achieves expected accuracy on Sleep-EDF."""
    # Given: Sleep-EDF test recording
    edf_path = Path("data/datasets/external/sleep-edf/test/SC4001E0-PSG.edf")
    raw = mne.io.read_raw_edf(edf_path)
    
    # When: Running YASA
    analyzer = SleepAnalyzer()
    hypnogram = analyzer.predict_stages(raw)
    
    # Then: Reasonable stage distribution
    stage_counts = pd.value_counts(hypnogram)
    assert "N2" in stage_counts  # Most common stage
    assert len(hypnogram) == len(raw) // (30 * raw.info['sfreq'])  # 30s epochs

def test_sleep_metrics_calculation():
    """Test sleep metrics calculation accuracy."""
    # Given: Known hypnogram
    hypnogram = create_test_hypnogram(
        total_minutes=480,  # 8 hours
        sleep_minutes=420,  # 7 hours sleep
        rem_minutes=90,     # 1.5 hours REM
        n3_minutes=120      # 2 hours deep sleep
    )
    
    # When: Calculating metrics
    analyzer = SleepAnalyzer()
    metrics = analyzer.calculate_metrics(hypnogram)
    
    # Then: Accurate metrics
    assert abs(metrics["sleep_efficiency"] - 87.5) < 0.1
    assert abs(metrics["rem_percentage"] - 21.4) < 0.1
    assert abs(metrics["n3_percentage"] - 28.6) < 0.1

def test_yasa_performance_benchmark():
    """Test YASA meets 87.46% accuracy target."""
    # Given: Annotated sleep test set
    test_set = load_sleep_edf_test_set()
    
    # When: Running YASA on all recordings
    analyzer = SleepAnalyzer()
    accuracies = []
    
    for recording in test_set:
        pred = analyzer.predict_stages(recording.raw)
        acc = accuracy_score(recording.true_stages, pred)
        accuracies.append(acc)
    
    # Then: Meets performance target
    mean_accuracy = np.mean(accuracies)
    assert mean_accuracy >= 0.87  # 87.46% target from YASA paper
```

### 5. Event Detection Tests

#### 5.1 Epileptiform Discharge Detection
```python
# tests/unit/test_event_detection.py

def test_epileptiform_detection():
    """Test detection of epileptiform discharges."""
    # Given: EEG with synthetic spikes
    eeg_with_spikes = create_eeg_with_epileptiform_spikes(
        n_spikes=10,
        spike_amplitude=100e-6,
        spike_duration=0.07  # 70ms
    )
    
    # When: Detecting events
    detector = EventDetector(event_type="epileptiform")
    events = detector.detect(eeg_with_spikes)
    
    # Then: Spikes detected
    assert len(events) >= 8  # 80% sensitivity
    assert all(e.confidence > 0.7 for e in events)
    assert all(0.05 <= e.duration <= 0.1 for e in events)

def test_pled_gped_detection():
    """Test PLED/GPED pattern detection."""
    # Given: EEG with periodic discharges
    eeg_with_pleds = create_eeg_with_periodic_pattern(
        pattern_type="PLED",
        frequency=1.5,  # 1.5 Hz
        duration=30.0   # 30 seconds
    )
    
    # When: Detecting periodic patterns
    detector = EventDetector(event_type="periodic")
    patterns = detector.detect_periodic_patterns(eeg_with_pleds)
    
    # Then: Pattern identified
    assert len(patterns) > 0
    assert any(p.pattern_type == "PLED" for p in patterns)
    assert any(1.0 <= p.frequency <= 2.0 for p in patterns)
```

### 6. Performance and Scalability Tests

#### 6.1 Processing Speed Tests
```python
# tests/performance/test_pipeline_speed.py

def test_pipeline_processing_speed():
    """Test pipeline meets <2 min for 20-min EEG target."""
    # Given: 20-minute EEG recording
    eeg_20min = create_realistic_eeg(duration_minutes=20)
    
    # When: Processing through full pipeline
    pipeline = FullPipeline()
    start_time = time.time()
    results = pipeline.process(eeg_20min)
    elapsed = time.time() - start_time
    
    # Then: Meets performance target
    assert elapsed < 120  # <2 minutes
    assert results.quality_report is not None
    assert results.abnormality_score is not None
    assert results.sleep_stages is not None

def test_concurrent_processing():
    """Test handling 50 concurrent analyses."""
    # Given: 50 EEG recordings
    recordings = [create_realistic_eeg(5) for _ in range(50)]
    
    # When: Processing concurrently
    pipeline = FullPipeline(max_workers=10)
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(pipeline.process, rec) for rec in recordings]
        results = [f.result(timeout=300) for f in futures]
    
    # Then: All complete successfully
    assert len(results) == 50
    assert all(r.status == "completed" for r in results)
    assert time.time() - start_time < 300  # <5 minutes for all

def test_memory_usage():
    """Test memory usage stays within bounds."""
    # Given: Large EEG file
    large_eeg = create_realistic_eeg(duration_minutes=60)
    
    # When: Processing with memory monitoring
    pipeline = FullPipeline()
    
    import tracemalloc
    tracemalloc.start()
    
    results = pipeline.process(large_eeg)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Then: Reasonable memory usage
    assert peak < 4 * 1024 * 1024 * 1024  # <4GB peak memory
```

### 7. API Endpoint Tests

#### 7.1 FastAPI Integration Tests
```python
# tests/integration/test_api.py

@pytest.mark.asyncio
async def test_analysis_endpoint():
    """Test /analyze endpoint functionality."""
    # Given: Test client and EEG file
    async with AsyncClient(app=app, base_url="http://test") as client:
        files = {"file": ("test.edf", create_test_edf_bytes(), "application/octet-stream")}
        
        # When: Uploading for analysis
        response = await client.post(
            "/api/v1/eeg/analyze",
            files=files,
            data={"analysis_type": "full"}
        )
        
        # Then: Analysis queued successfully
        assert response.status_code == 202
        assert "job_id" in response.json()
        assert response.json()["status"] == "queued"

@pytest.mark.asyncio
async def test_status_endpoint():
    """Test job status checking."""
    # Given: Completed job
    job_id = "test-job-123"
    await redis_client.set(f"job:{job_id}", json.dumps({
        "status": "completed",
        "progress": 100,
        "results": {"abnormality_score": 0.85}
    }))
    
    # When: Checking status
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get(f"/api/v1/jobs/{job_id}")
    
    # Then: Status returned correctly
    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert response.json()["results"]["abnormality_score"] == 0.85

@pytest.mark.asyncio 
async def test_api_response_time():
    """Test API responds within 100ms."""
    # Given: Small EEG file
    small_eeg = create_test_edf_bytes(duration=10)
    
    # When: Making request
    async with AsyncClient(app=app, base_url="http://test") as client:
        start = time.time()
        response = await client.post(
            "/api/v1/eeg/analyze",
            files={"file": ("test.edf", small_eeg, "application/octet-stream")}
        )
        elapsed = time.time() - start
    
    # Then: Fast response
    assert elapsed < 0.1  # <100ms
    assert response.status_code == 202
```

### 8. Error Handling Tests

#### 8.1 Graceful Failure Tests
```python
# tests/unit/test_error_handling.py

def test_handles_corrupted_edf():
    """Test graceful handling of corrupted EDF files."""
    # Given: Corrupted EDF
    corrupted_edf = create_corrupted_edf()
    
    # When: Processing
    pipeline = FullPipeline()
    result = pipeline.process(corrupted_edf)
    
    # Then: Graceful failure
    assert result.status == "failed"
    assert "corrupted" in result.error_message.lower()
    assert result.partial_results is None

def test_handles_missing_channels():
    """Test handling of recordings with missing channels."""
    # Given: EEG with only 10 channels (minimum is 19)
    insufficient_eeg = create_mock_raw(n_channels=10)
    
    # When: Processing
    pipeline = FullPipeline()
    result = pipeline.process(insufficient_eeg)
    
    # Then: Informative error
    assert result.status == "failed"
    assert "insufficient channels" in result.error_message.lower()
    assert "10 channels found, 19 required" in result.error_message

def test_handles_wrong_sampling_rate():
    """Test automatic resampling for non-256Hz data."""
    # Given: 1000Hz EEG data
    high_sfreq_eeg = create_mock_raw(sfreq=1000)
    
    # When: Processing
    pipeline = FullPipeline()
    result = pipeline.process(high_sfreq_eeg)
    
    # Then: Successful processing after resampling
    assert result.status == "completed"
    assert result.preprocessing_notes["resampled_from"] == 1000
    assert result.preprocessing_notes["resampled_to"] == 256
```

## Test Data Fixtures

### Synthetic Data Generation
```python
# tests/fixtures/synthetic_data.py

@pytest.fixture
def synthetic_normal_eeg():
    """Generate synthetic normal EEG."""
    return create_synthetic_eeg(
        duration=30.0,
        sfreq=256,
        n_channels=20,
        noise_level=10e-6,
        alpha_power=20e-6,
        beta_power=5e-6
    )

@pytest.fixture
def synthetic_abnormal_eeg():
    """Generate synthetic abnormal EEG."""
    eeg = create_synthetic_eeg(duration=30.0)
    # Add slowing
    add_delta_slowing(eeg, power=50e-6)
    # Add spikes
    add_epileptiform_spikes(eeg, n_spikes=5)
    return eeg

@pytest.fixture(scope="session")
def cached_tuab_test_set():
    """Load cached TUAB test set."""
    cache_path = Path("tests/data/tuab_test_cached.pkl")
    if cache_path.exists():
        return pickle.load(cache_path.open("rb"))
    else:
        # Generate and cache
        test_set = load_tuab_test_subset(n_samples=100)
        pickle.dump(test_set, cache_path.open("wb"))
        return test_set
```

## Continuous Integration Configuration

### GitHub Actions Workflow
```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.11, 3.12]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/uv
        key: ${{ runner.os }}-uv-${{ hashFiles('**/pyproject.toml') }}
    
    - name: Install dependencies
      run: |
        pip install uv
        uv sync --dev
    
    - name: Run tests with coverage
      run: |
        uv run pytest -xvs --cov=brain_go_brrr --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: true
```

## Test Execution Strategy

### 1. Local Development
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=brain_go_brrr --cov-report=html

# Run specific test module
uv run pytest tests/unit/test_eegpt_model.py -xvs

# Run in watch mode
uv run pytest-watch

# Run only fast tests
uv run pytest -m "not slow"
```

### 2. Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest-check
        name: pytest-check
        entry: uv run pytest -x --tb=short
        language: system
        pass_filenames: false
        always_run: true
```

### 3. Performance Benchmarking
```bash
# Run performance tests
uv run pytest tests/performance/ --benchmark-only

# Generate performance report
uv run pytest tests/performance/ --benchmark-autosave

# Compare with baseline
uv run pytest tests/performance/ --benchmark-compare=0001
```

## Test Coverage Requirements

### Module Coverage Targets
- Core modules: ≥95% coverage
- API endpoints: 100% coverage
- Error handling: 100% coverage
- Integration points: ≥90% coverage

### Coverage Enforcement
```toml
# pyproject.toml
[tool.coverage.run]
source = ["brain_go_brrr"]
omit = ["*/tests/*", "*/migrations/*"]

[tool.coverage.report]
fail_under = 90
show_missing = true
skip_covered = false
```

## Testing Best Practices

1. **Test Naming**: Use descriptive names that explain what is being tested
2. **Arrange-Act-Assert**: Follow AAA pattern consistently
3. **One Assertion**: Each test should verify one behavior
4. **Test Data**: Use factories and fixtures for test data
5. **Mocking**: Mock external services and I/O operations
6. **Deterministic**: Tests must be reproducible
7. **Fast Feedback**: Keep unit tests under 100ms each
8. **Documentation**: Document complex test scenarios

## Conclusion

This TDD specification ensures that Brain-Go-Brrr is built with quality and reliability from the ground up. By writing tests first, we guarantee that our implementation meets all requirements and maintains high quality throughout development.