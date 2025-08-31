# Coverage Strategy

## Philosophy

We use a **dual-coverage** approach that measures unit and integration coverage separately. This is standard practice at companies like Google, Meta, and Netflix.

## Why Split Coverage?

Unit tests and integration tests serve different purposes:
- **Unit tests**: Fast, isolated, deterministic - test business logic
- **Integration tests**: Slower, use real services/data - test system behavior

Mixing them in one coverage metric creates false signals. A module that can only be tested via integration (e.g., database adapter) will drag down unit coverage unfairly.

## Coverage Jobs

### 1. Unit Coverage (`make test-all-cov`)
- **Config**: `.coveragerc.unit`
- **Threshold**: 70% (will raise to 75% as we add more unit tests)
- **Excludes**: 
  - Data loaders (`infra/data/*`)
  - API routes needing services (`api/routers/{sleep,jobs,queue}.py`)
  - Complex pipelines (`application/pipeline/*`)
  - Domain modules needing real data (`domain/sleep/analyzer.py`)
  - Cache implementations (`infra/cache.py`)
- **Run**: Every commit in CI

### 2. Data/Integration Coverage (`make test-data-cov`)
- **Config**: `.coveragerc.data`
- **Threshold**: 50% (integration tests are expensive)
- **Includes**: All the modules excluded from unit coverage
- **Run**: Nightly or on-demand with real/synthetic data

## Marker Semantics

- `@pytest.mark.unit`: Pure unit test, no I/O
- `@pytest.mark.integration`: Needs services/data
- `@pytest.mark.data`: Requires REAL datasets (not synthetic)
- `@pytest.mark.synth`: Can run with synthetic data
- `@pytest.mark.slow`: Takes >1 second
- `@pytest.mark.gpu`: Requires CUDA

## Running Tests

```bash
# Unit tests (fast, no data needed)
make test-all-cov

# Integration with synthetic data
BGB_ALLOW_SYNTH_TUAB=1 pytest -m "integration and synth"

# Integration with real data
export BGB_DATA_ROOT=/data
pytest -m "integration and data" --run-data

# Full data coverage
make test-data-cov
```

## Why This Is Professional

1. **Honest metrics**: Unit coverage measures what unit tests can actually test
2. **Fast feedback**: Unit tests run in <3 minutes, catch most bugs
3. **Comprehensive**: Integration tests catch system-level issues
4. **Clear boundaries**: No confusion about what each test type does
5. **CI-friendly**: Fast unit tests on every commit, slower integration tests nightly

## Future Improvements

- Raise unit threshold to 75% once we add more focused unit tests
- Add mutation testing for critical paths
- Add property-based testing for data structures
- Consider contract testing for API boundaries

## References

- [Google Testing Blog: Code Coverage Best Practices](https://testing.googleblog.com/2020/08/code-coverage-best-practices.html)
- [Martin Fowler: Test Coverage](https://martinfowler.com/bliki/TestCoverage.html)