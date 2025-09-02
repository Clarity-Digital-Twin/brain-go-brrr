# Test Suite Reorganization Plan

> ⚠️ **ARCHIVED DOCUMENT - DO NOT USE FOR CURRENT DEVELOPMENT**
> This document is preserved for historical reference only.
> For current documentation, see [docs/README.md](../../README.md)
> Archive date: September 2, 2025

---



## Executive Summary
✅ COMPLETED: Reorganized 93 unit tests from flat structure to mirror source code architecture
✅ FIXED: Moved misplaced API integration tests and behavior tests to correct locations
✅ RESULT: Clean test hierarchy following standard testing pyramid

## Migration Map

### Phase 1: Create Directory Structure (No Code Changes)

```bash
tests/unit/
├── api/
│   ├── routers/
│   └── # API-related tests
├── application/
│   ├── config/
│   ├── factories/
│   ├── jobs/
│   ├── pipeline/
│   ├── training/
│   └── use_cases/
├── domain/
│   ├── abnormal/
│   ├── ports/
│   ├── preprocessing/
│   │   └── features/
│   ├── quality/
│   └── sleep/
├── infra/
│   ├── adapters/
│   ├── cache/
│   ├── data/
│   ├── external/
│   ├── ml_models/
│   ├── preprocessing/
│   ├── redis/
│   └── serialization/
├── services/
├── utils/
│   └── logging/
├── visualization/
├── cli/
├── presentation/
└── models/  # Legacy
```

Notes:
- `application/pipeline/` currently has tests (e.g., orchestration) — keep this bucket.
- `models/` (Legacy) currently contains 2 tests; retain until legacy code is removed.
- API endpoint tests live under `tests/integration/api/` (top-level `tests/api/` removed).

### Phase 2: File Migration Plan

#### API Tests (10 files)
```
FROM: tests/unit/test_api_*.py
TO:   tests/unit/api/

test_api_app.py                → tests/unit/api/test_app.py
test_api_cache.py              → tests/unit/api/test_cache.py
test_api_cache_router.py       → tests/unit/api/routers/test_cache.py
test_api_dependencies.py       → tests/unit/api/test_dependencies.py
test_api_models.py             → tests/unit/api/test_models.py
test_api_resources_router.py   → tests/unit/api/routers/test_resources.py
test_api_routers_eegpt.py      → tests/unit/api/routers/test_eegpt.py
test_api_routers_resources_clean.py → CONSOLIDATE with test_resources.py
test_api_routes.py             → tests/unit/api/test_routes.py
test_api_schemas.py            → tests/unit/api/test_schemas.py
```

#### Domain Tests (25 files)
```
FROM: tests/unit/test_{domain}_*.py
TO:   tests/unit/domain/{subdomain}/

# Abnormality (5 files)
test_abnormality_accuracy.py        → tests/unit/domain/abnormal/test_accuracy.py
test_abnormality_detection_real.py  → tests/unit/domain/abnormal/test_detection_real.py
test_abnormality_preprocessor.py    → tests/unit/domain/abnormal/test_preprocessor.py
test_enhanced_abnormality.py        → tests/unit/domain/abnormal/test_enhanced.py
test_accuracy_metrics.py            → tests/unit/domain/abnormal/test_metrics.py

# Preprocessing (8 files)
test_core_preprocessing.py          → tests/unit/domain/preprocessing/test_core.py
test_preprocessing_pipeline.py      → tests/unit/domain/preprocessing/test_pipeline.py
test_preprocessing_real.py          → tests/unit/domain/preprocessing/test_real.py
test_flexible_preprocessing.py      → tests/unit/domain/preprocessing/test_flexible.py
test_nan_policy.py                  → tests/unit/domain/preprocessing/test_nan_policy.py
test_window_extractor.py            → tests/unit/domain/preprocessing/test_window_extractor.py
test_window_extractor_edge_cases.py → tests/unit/domain/preprocessing/test_window_extractor_edges.py
test_window_tail.py                 → tests/unit/domain/preprocessing/test_window_tail.py

# Sleep (4 files)
test_sleep_analysis.py              → tests/unit/domain/sleep/test_analysis.py
test_sleep_montage_detection.py     → tests/unit/domain/sleep/test_montage_detection.py
test_sleep_preprocessor.py          → tests/unit/domain/sleep/test_preprocessor.py
test_enhanced_sleep_analyzer.py     → tests/unit/domain/sleep/test_enhanced_analyzer.py

# Quality (1 file)
test_quality_controller.py          → tests/unit/domain/quality/test_controller.py

# Channels (2 files)
test_channels.py                    → tests/unit/domain/test_channels.py
test_channel_mapper.py              → tests/unit/domain/test_channel_mapper.py

# Exceptions (2 files)
test_core_exceptions.py             → tests/unit/domain/test_exceptions.py
test_exceptions_hierarchy.py        → tests/unit/domain/test_exceptions_hierarchy.py

# Ports (2 files)
test_domain_ports_behavior.py       → tests/unit/domain/ports/test_behavior.py
test_domain_settings.py             → tests/unit/domain/test_settings.py
```

#### Infrastructure Tests (40 files)
```
FROM: tests/unit/test_{infra}_*.py
TO:   tests/unit/infra/{subdomain}/

# ML Models / EEGPT (20 files)
test_eegpt_checkpoint_loading.py    → tests/unit/infra/ml_models/test_checkpoint_loading.py
test_eegpt_compat_coverage.py       → tests/unit/infra/ml_models/test_compat.py
test_eegpt_compat_strict.py         → CONSOLIDATE with test_compat.py
test_eegpt_contract_fuzzer.py       → tests/unit/infra/ml_models/test_contract_fuzzer.py
test_eegpt_extreme_discrimination.py → tests/unit/infra/ml_models/test_discrimination.py
test_eegpt_feature_extractor.py     → tests/unit/infra/ml_models/test_feature_extractor.py
test_eegpt_linear_probe.py          → tests/unit/infra/ml_models/test_linear_probe.py
test_eegpt_model_loading.py         → tests/unit/infra/ml_models/test_model_loading.py
test_eegpt_orchestration.py         → tests/unit/application/pipeline/test_orchestration.py
test_eegpt_pipeline.py              → tests/unit/infra/ml_models/test_pipeline.py
test_eegpt_preprocessing_small.py   → tests/unit/infra/ml_models/test_preprocessing.py
test_eegpt_probe_unified.py         → tests/unit/infra/ml_models/test_probe_unified.py
test_eegpt_summary_tokens.py        → tests/unit/infra/ml_models/test_summary_tokens.py
test_encoder_raw_output.py          → tests/unit/infra/ml_models/test_encoder_output.py
test_linear_probe.py                → tests/unit/infra/ml_models/test_probe.py
test_linear_probe_head.py           → CONSOLIDATE with test_probe.py
test_models_eegpt_model.py          → CONSOLIDATE with test_model_loading.py
test_models_eegpt_wrapper.py        → tests/unit/infra/ml_models/test_wrapper.py
test_models_linear_probe.py         → CONSOLIDATE with test_probe.py
test_robust_eegpt_probe.py          → tests/unit/infra/ml_models/test_robust_probe.py
test_rotary_embedding_fix.py        → tests/unit/infra/ml_models/test_rotary_embedding.py

# Data (5 files)
test_edf_loader.py                  → tests/unit/infra/data/test_edf_loader.py
test_edf_streaming.py               → tests/unit/infra/data/test_edf_streaming.py
test_edf_validator.py               → tests/unit/infra/data/test_edf_validator.py
test_collate_functions.py           → tests/unit/utils/test_collate.py

# External/YASA (4 files)
test_yasa_adapter.py                → tests/unit/infra/external/test_yasa_adapter.py
test_yasa_compliance.py             → tests/unit/infra/external/test_yasa_compliance.py
test_yasa_smoothing.py              → tests/unit/infra/external/test_yasa_smoothing.py

# Autoreject (2 files)
test_autoreject_adapter.py          → tests/unit/infra/adapters/test_autoreject.py
test_chunked_autoreject.py          → tests/unit/infra/preprocessing/test_chunked_autoreject.py

# Redis/Cache (3 files)
test_redis_caching.py               → tests/unit/infra/redis/test_caching.py
test_redis_pool.py                  → tests/unit/infra/redis/test_pool.py
test_cache_factory.py               → tests/unit/infra/cache/test_factory.py

# Serialization (5 files)
test_serialization.py               → tests/unit/infra/serialization/test_core.py
test_serialization_contract.py      → tests/unit/infra/serialization/test_contract.py
test_serialization_edge_cases.py    → tests/unit/infra/serialization/test_edge_cases.py
test_serialization_registry.py      → tests/unit/infra/serialization/test_registry.py
test_jobdata_serialization.py       → tests/unit/application/jobs/test_serialization.py

# Other Infrastructure (1 file)
test_safe_load.py                   → tests/unit/infra/test_safe_load.py
```

#### Application Tests (3 files)
```
test_async_base.py                  → tests/unit/application/test_async_base.py
test_job_store_behavior.py          → tests/unit/application/jobs/test_store_behavior.py
test_config.py                      → tests/unit/application/config/test_config.py
test_config_defaults.py             → tests/unit/application/config/test_defaults.py
```


#### CLI Tests (3 files)
```
test_cli.py                         → tests/unit/cli/test_cli.py
test_cli_commands.py                → tests/unit/cli/test_commands.py
test_cli_streaming.py               → tests/unit/cli/test_streaming.py
```

#### Utils Tests (3 files)
```
test_time_utils.py                  → tests/unit/utils/test_time.py
test_logger_singleton.py            → tests/unit/utils/logging/test_logger.py
test_logging_mask.py                → tests/unit/utils/logging/test_mask.py
```

#### Presentation Tests (2 files)
```
test_markdown_report.py             → tests/unit/presentation/test_markdown_report.py
test_pdf_report.py                  → tests/unit/presentation/test_pdf_report.py
```

#### Tests to Move to Integration (5 files)
```
test_abnormality_detection_real.py  → tests/integration/test_abnormality_real.py
test_data_pipeline_real.py          → tests/integration/test_data_pipeline_real.py
test_eegpt_real_inference.py        → tests/integration/test_eegpt_real_inference.py
test_preprocessing_real.py          → tests/integration/test_preprocessing_real.py
test_train_sleep_probe.py           → tests/integration/test_train_sleep_probe.py
```

#### Tests to Remove/Consolidate
```
test_coverage_boost_minimal.py      → DELETE (coverage hack)
test_coverage_boost_refactored.py   → DELETE (coverage hack)
test_improved_mocking.py            → DELETE (test infrastructure, not source)
test_classifier_compatibility.py    → REVIEW (might be integration test)
test_code_quality.py                → MOVE to tests/quality/
```

### Phase 3: Fix Test-to-Test Imports

Extract shared helpers before moving:
```python
# Move shared function from test_accuracy_metrics.py to tests/_test_utils.py
# Then update imports in test_abnormality_accuracy.py:
# Old: from tests.unit.test_accuracy_metrics import record_accuracy_metric
# New: from tests._test_utils import record_accuracy_metric
```

### Phase 4: Import Updates

After moving files, update all imports:
```python
# Old import
from tests.unit.test_eegpt_wrapper import MockEEGPT

# New import
from tests.unit.infra.ml_models.test_wrapper import MockEEGPT
```

### Phase 5: Consolidation

Merge duplicate/related test files:
1. Consolidate 3 linear probe test files into one
2. Merge coverage/strict compat tests
3. Combine related API router tests
4. Merge serialization edge cases into main test

### Phase 6: Documentation

Update:
1. `tests/README.md` with new structure
2. CI/CD paths if needed
3. Developer documentation
4. Test running instructions

## Execution Timeline

- **Hour 1**: Create directory structure
- **Hour 2-3**: Move API and domain tests
- **Hour 4-5**: Move infrastructure tests
- **Hour 6**: Update imports
- **Hour 7**: Run tests, fix breaks
- **Hour 8**: Documentation and cleanup

## Success Metrics

- ✅ All tests pass post-migration
- ✅ Test discovery time reduced
- ✅ Clear 1:1 mapping between source and tests
- ✅ No orphaned tests in unit/ root
- ✅ Easier to find related tests

## Risk Mitigation

1. **Create backup branch** before starting
2. **Move in small batches** (5-10 files at a time)
3. **Run tests after each batch** to catch breaks early
4. **Use git mv** to preserve history
5. **Update imports incrementally**

## Rollback Plan

If issues arise:
```bash
git checkout fix/test-suite-organization
git reset --hard HEAD~n  # where n = number of commits to undo
```

## Path Stability for Test Data

Centralize fixture paths to avoid breakage after moves:

```python
# In tests/conftest.py
import pytest
from pathlib import Path

@pytest.fixture
def fixtures_dir():
    """Root fixtures directory."""
    return Path(__file__).parent / "fixtures"

@pytest.fixture
def eeg_fixtures_dir(fixtures_dir):
    """EEG test data directory."""
    return fixtures_dir / "eeg"

@pytest.fixture
def metrics_file():
    """Metrics JSON file path."""
    return Path(__file__).parent / "test_accuracy_metrics.json"
```

## Post-Migration Verification

After completing migration:

```bash
# 1. Run all quality checks
make check-all

# 2. Quick test to verify discovery
pytest --co -q tests/unit/

# 3. Run subset to verify functionality
pytest -k 'eegpt or redis or edf' -q

# 4. Check for broken imports
rg -n 'from tests\.unit\.test_' tests/

# 5. Full test suite with coverage
make test-all-cov
```

Note: The entire integration subtree is marked via `tests/integration/conftest.py`.

## Phase 7: Final Corrections (COMPLETED)

After initial reorganization, discovered and fixed:

1. **Duplicate API Tests**:
   - `tests/api/` had 8 integration tests → Moved to `tests/integration/api/`
   - `tests/unit/api/` correctly has unit tests → Kept as-is
   - Eliminated confusing duplication

2. **Misnamed behavior Directory**:
   - `tests/behavior/test_all_components.py` → Moved to `tests/integration/test_end_to_end.py`
   - Removed empty behavior directory

3. **Final Structure**:
   ```
   tests/
   ├── unit/           # Pure unit tests (mocked, isolated)
   ├── integration/    # Integration tests (real dependencies)
   │   └── api/        # API integration tests using TestClient
   ├── smoke/          # Quick sanity checks
   ├── benchmarks/     # Performance tests
   └── quality/        # Code quality checks
   ```

## Next Steps

1. ✅ Review and approve plan
2. ✅ Create directory structure
3. ✅ Execute migration in phases
4. ✅ Run full test suite
5. ✅ Update documentation
6. ✅ Fix final organization issues
7. Submit PR for review
