# Testing Guide

## Test Structure

```
tests/
├── unit/           # Fast, isolated tests
├── integration/    # Component interaction tests
├── api/           # API endpoint tests
├── benchmarks/    # Performance tests
└── smoke/         # Basic functionality checks
```

## Running Tests

### Quick Test (Unit Only)

```bash
# Fast feedback during development
uv run pytest tests/unit -q
```

### Full Test Suite

```bash
# All tests with coverage
make test

# Or manually
uv run pytest tests --cov=brain_go_brrr
```

### Specific Components

```bash
# Sleep analysis tests
uv run pytest tests/unit/test_sleep_analysis.py

# Quality control tests
uv run pytest tests/unit/test_quality_controller.py

# API tests
uv run pytest tests/api/
```

## Test Categories

### Unit Tests (454 passing)

Fast, isolated tests for individual components:

```python
def test_eegpt_feature_extraction():
    """Test EEGPT extracts correct feature dimensions."""
    model = EEGPTModel(auto_load=False)
    data = np.random.randn(20, 1024)  # 20 channels, 1024 samples

    features = model.extract_features(data)

    assert features.shape == (1, 4, 512)  # 1 window, 4 tokens, 512 dims
```

### Integration Tests

Test component interactions with real data:

```python
@pytest.mark.integration
def test_full_pipeline():
    """Test complete analysis pipeline."""
    raw = mne.io.read_raw_edf("data/sample.edf")

    # QC -> Features -> Analysis
    qc_report = run_quality_control(raw)
    features = extract_eegpt_features(raw)
    results = analyze_abnormality(features)

    assert results["confidence"] > 0.5
```

### API Tests

Test REST endpoints:

```python
def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```

### Performance Benchmarks

```python
@pytest.mark.benchmark
def test_eegpt_inference_speed(benchmark):
    """Benchmark EEGPT inference time."""
    model = EEGPTModel()
    data = np.random.randn(20, 1024)

    result = benchmark(model.extract_features, data)

    assert benchmark.stats["mean"] < 0.1  # <100ms
```

## Test Fixtures

### Common Fixtures

```python
@pytest.fixture
def mock_eeg_data():
    """Generate realistic mock EEG data."""
    n_channels = 19
    n_samples = 256 * 60  # 1 minute at 256Hz
    return np.random.randn(n_channels, n_samples) * 50e-6

@pytest.fixture
def sample_raw():
    """Load sample EEG recording."""
    return mne.io.read_raw_edf("tests/data/sample.edf")
```

## Mocking External Services

```python
@patch("brain_go_brrr.infra.external.yasa_adapter.yasa")
def test_sleep_staging_with_mock(mock_yasa):
    """Test sleep staging with mocked YASA."""
    mock_yasa.sleep_stage.return_value = np.array([0, 1, 2, 3, 4])

    results = stage_sleep(mock_eeg_data)

    assert "stages" in results
    mock_yasa.sleep_stage.assert_called_once()
```

## Coverage Requirements

- Minimum coverage: 60%
- Target coverage: 80%
- Critical paths: 95%

Check coverage:

```bash
# Generate coverage report
uv run pytest --cov=brain_go_brrr --cov-report=html

# View report
open htmlcov/index.html
```

## CI/CD Integration

Tests run automatically on:
- Every push
- Pull requests
- Scheduled nightly runs

GitHub Actions workflow:

```yaml
- name: Run Tests
  run: |
    uv run pytest tests/unit tests/smoke -q
    uv run pytest tests/api --tb=short
    uv run pytest tests/integration -m "not slow"
```

## Test Markers

```python
# Skip slow tests
@pytest.mark.slow
def test_full_dataset_processing():
    pass

# GPU required
@pytest.mark.gpu
def test_cuda_operations():
    pass

# External data required
@pytest.mark.data
def test_with_real_eeg():
    pass
```

Run specific markers:

```bash
# Skip slow tests
pytest -m "not slow"

# Only GPU tests
pytest -m gpu

# Integration tests
pytest -m integration --run-integration
```

## Writing New Tests

### Test Structure

```python
def test_feature_name():
    """Test description.

    Given: Initial conditions
    When: Action performed
    Then: Expected outcome
    """
    # Arrange
    data = create_test_data()

    # Act
    result = function_under_test(data)

    # Assert
    assert result.shape == expected_shape
    assert result.mean() == pytest.approx(0.0, abs=0.1)
```

### Test Naming

- `test_` prefix required
- Descriptive names: `test_sleep_staging_detects_rem_sleep`
- Group related tests in classes

## Debugging Failed Tests

```bash
# Verbose output
pytest -xvs tests/unit/test_failing.py

# Stop on first failure
pytest -x

# Run specific test
pytest tests/unit/test_module.py::TestClass::test_method

# Debug with pdb
pytest --pdb
```

## Performance Testing

```bash
# Run benchmarks
pytest tests/benchmarks --benchmark-only

# Compare results
pytest-benchmark compare 0001 0002

# Save results
pytest --benchmark-autosave
```

## Test Data

Sample data location:
```
tests/data/
├── sample.edf       # 1-minute EEG sample
├── sleep.edf        # 8-hour sleep recording
└── abnormal.edf     # Abnormal EEG patterns
```

## Common Issues

### Import Errors

```bash
# Set Python path
export PYTHONPATH=$PYTHONPATH:src

# Or use pytest.ini
```

### Fixture Not Found

```python
# Add conftest.py in test directory
# tests/conftest.py
import pytest

@pytest.fixture(scope="session")
def shared_resource():
    return expensive_setup()
```

### Async Test Issues

```python
@pytest.mark.asyncio
async def test_async_endpoint():
    async with AsyncClient(app) as client:
        response = await client.get("/api/v1/async")
        assert response.status_code == 200
```

## Best Practices

1. **Keep tests fast** - mock external dependencies
2. **Test behavior, not implementation** - focus on outputs
3. **Use fixtures** - avoid duplication
4. **Clear assertions** - one logical assertion per test
5. **Descriptive names** - tests are documentation
6. **Isolate tests** - no dependencies between tests
7. **Clean up** - reset state after tests
