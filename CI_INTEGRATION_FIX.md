# CI Integration Test Fix - Complete Implementation

## Problem Identified

Your integration tests were failing because they expected real data files that don't exist in CI:
- `data/datasets/` - TUAB/TUEV datasets
- `data/models/` - Pretrained model checkpoints
- `data/datasets/external/sleep-edf/` - Sleep-EDF recordings

## Solution Applied (First Principles)

### 1. Separation of Concerns

**Principle**: Tests should be explicit about their dependencies

We separated integration tests into two categories:
- **CI-friendly integration**: Services + pipelines with synthetic/fixture data
- **Data-backed integration**: Requires real datasets (run locally or scheduled)

### 2. Implementation

#### Added `@pytest.mark.data` Marker
```python
# pytest.ini
markers =
    integration: marks tests as integration tests (services/pipelines)
    data: marks tests requiring real datasets (TUAB/TUEV/Sleep-EDF)
    gpu: marks tests requiring GPU
```

#### Updated conftest.py
```python
def pytest_collection_modifyitems(config, items):
    # Skip data tests unless --run-data AND BGB_DATA_ROOT exists
    run_data = config.getoption("--run-data", default=False)
    data_root = os.environ.get("BGB_DATA_ROOT", "")
    has_data = bool(data_root and Path(data_root).exists())

    if not (run_data and has_data):
        skip_data = pytest.mark.skip(reason="need BGB_DATA_ROOT and --run-data")
        for item in items:
            if "data" in item.keywords:
                item.add_marker(skip_data)

    # Skip GPU tests when no CUDA
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="no CUDA in CI")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
```

#### Marked Data-Dependent Tests
```python
# tests/integration/test_real_tuab_autoreject.py
@pytest.mark.data
def test_load_real_edf_with_autoreject(self, real_tuab_file):
    # Needs real TUAB files

# tests/integration/test_parallel_pipeline.py
@pytest.mark.data
def test_with_real_sleep_edf(self):
    # Needs Sleep-EDF dataset

# tests/integration/test_eegpt_integration.py
@pytest.mark.data
def test_end_to_end_pipeline(self, model_path):
    # Needs pretrained model + Sleep-EDF
```

#### Updated Makefile Targets
```makefile
# CI-friendly (no GPU, no data)
test-integration:
    $(PYTEST) tests --run-integration -m "integration and not gpu and not data" -v

# Data-backed (for local/scheduled runs)
test-integration-data:
    @if [ -z "$$BGB_DATA_ROOT" ]; then echo "Need BGB_DATA_ROOT"; exit 0; fi
    $(PYTEST) tests --run-integration --run-data -m "integration and data" -v
```

## Result

### What CI Runs Now
- ✅ Integration tests that use fixtures/synthetic data
- ✅ Service integration (Redis, API endpoints)
- ✅ Pipeline tests with generated EEG
- ⏭️ Skips tests needing real datasets
- ⏭️ Skips GPU tests (no CUDA in CI)

### What Still Works Locally
```bash
# With real data
export BGB_DATA_ROOT=/path/to/data
make test-integration-data  # Runs data-backed tests

# With GPU
CUDA_VISIBLE_DEVICES=0 pytest tests -m "gpu" --run-integration
```

## Verification

```bash
# Check what runs in CI (should exclude data tests)
pytest tests --collect-only -m "integration and not gpu and not data" --run-integration

# Check what's skipped
pytest tests --co -m "data" -q
# Should show: "58 deselected" or similar
```

## Why This Is Professional

1. **Explicit Contracts**: Tests declare dependencies via markers
2. **No Hidden Failures**: Data tests skip cleanly with clear reasons
3. **Single Source of Truth**: Makefile defines all test strategies
4. **Progressive Testing**: CI gets fast feedback, full suite runs on schedule
5. **Environment Aware**: Tests adapt to available resources

## Next Steps

1. **Add Scheduled Workflow** (optional):
   ```yaml
   # .github/workflows/nightly.yml
   on:
     schedule:
       - cron: '0 3 * * *'  # 3 AM UTC daily
   jobs:
     data-tests:
       runs-on: self-hosted  # Or use large runner with data mounted
       env:
         BGB_DATA_ROOT: /data
       steps:
         - run: make test-integration-data
   ```

2. **Use Fixtures for More Tests**:
   - Convert tests to use `tiny_edf` fixture when possible
   - Add more synthetic data generators
   - Reduce dependency on real files

3. **Document Data Requirements**:
   ```markdown
   # DATA_REQUIREMENTS.md
   - TUAB dataset: 2.5GB, place in data/datasets/tuab/
   - EEGPT checkpoint: 150MB, place in data/models/pretrained/
   - Sleep-EDF: 8GB, place in data/datasets/external/sleep-edf/
   ```

The CI is now **deterministic and green** while preserving full test coverage when data is available!
