# Test Suite Organization

## Directory Structure

```
tests/
├── README.md                 # This file
├── conftest.py              # Main pytest configuration and shared fixtures
├── fakes.py                 # Test doubles (lightweight fakes)
├── _mocks.py                # Mock objects for testing
├── _test_utils.py           # Shared test utilities
├── import_mapping.md        # Import guide for tests
│
├── unit/                    # Unit tests (fast, isolated)
│   ├── test_*.py           # Individual unit tests
│   └── ...
│
├── integration/             # Integration tests (require data/models)
│   ├── test_sleep_enhanced.py
│   ├── test_yasa_channel_aliasing.py
│   └── ...
│
├── api/                     # API endpoint tests
│   ├── test_*.py
│   └── ...
│
├── benchmarks/              # Performance benchmarks
│   ├── test_*.py
│   └── ...
│
└── fixtures/                # Test fixtures and data
    ├── tuab_fixtures.py     # TUAB dataset fixtures
    ├── benchmark_data.py    # Benchmark data fixtures
    ├── cache_fixtures.py    # Cache-related fixtures
    ├── synthetic_data.py    # Synthetic data generators
    └── metrics/
        └── accuracy_metrics.json  # Stored test metrics
```

## Test Categories

### Unit Tests (`tests/unit/`)
- Fast (<0.1s per test)
- No external dependencies
- No file I/O or network access
- Use fakes and mocks

### Integration Tests (`tests/integration/`)
- May require real data files
- Test component interactions
- Marked with `@pytest.mark.integration`
- Run with `--run-integration` flag

### API Tests (`tests/api/`)
- Test FastAPI endpoints
- Use TestClient
- Mock external services

### Benchmarks (`tests/benchmarks/`)
- Performance measurements
- Marked with `@pytest.mark.benchmark`
- Run separately in CI

## Running Tests

```bash
# Run all unit tests (default)
make test

# Run with coverage
make test-unit-cov

# Run integration tests
pytest tests --run-integration

# Run specific test file
pytest tests/unit/test_window_extractor.py

# Run benchmarks
pytest tests/benchmarks -m benchmark
```

## Test Markers

- `@pytest.mark.unit` - Unit tests (default)
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Tests taking >5 seconds
- `@pytest.mark.network` - Tests requiring network (set BGB_ALLOW_NET=1)
- `@pytest.mark.gpu` - Tests requiring GPU
- `@pytest.mark.benchmark` - Performance benchmarks

## Fixtures

Common fixtures are defined in:
- `conftest.py` - Main fixtures
- `fixtures/*.py` - Specialized fixtures

Key fixtures:
- `mock_eeg_data` - Synthetic EEG data
- `fake_redis` - In-memory Redis fake
- `client` - FastAPI test client
- `project_root` - Project root path

## Coverage

Current coverage: **62.14%**
Target: **60%** (minimum)

Run coverage report:
```bash
make test-fast-cov
```

## Best Practices

1. **Use fakes over mocks** when possible (see `fakes.py`)
2. **Set seeds** for reproducibility (`np.random.seed(42)`)
3. **Parametrize tests** to reduce duplication
4. **Name tests clearly** - what they test, not how
5. **Keep tests fast** - mock/fake heavy operations
6. **Test one thing** per test function
