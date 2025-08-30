# Test Suite Investigation & Analysis

## Current State Analysis

### Source Code Structure (`src/brain_go_brrr/`)
```
src/brain_go_brrr/
├── api/                 # FastAPI routes and REST endpoints
│   ├── routers/         # Individual route handlers
│   └── ...              # Auth, cache, models, schemas
├── application/         # Application layer (use cases)
│   ├── config/          # Configuration management
│   ├── factories/       # Factory patterns
│   ├── jobs/            # Job queue management
│   ├── pipeline/        # Pipeline orchestration
│   ├── training/        # Training workflows
│   └── use_cases/       # Business use cases
├── domain/              # Domain layer (business logic)
│   ├── abnormal/        # Abnormality detection
│   ├── ports/           # Port interfaces
│   ├── preprocessing/   # Preprocessing logic
│   ├── quality/         # Quality control
│   └── sleep/           # Sleep analysis
├── infra/               # Infrastructure layer
│   ├── adapters/        # External service adapters
│   ├── data/            # Data loaders and datasets
│   ├── external/        # External integrations
│   ├── ml_models/       # ML model implementations
│   ├── preprocessing/   # Infrastructure preprocessing
│   └── redis/           # Redis caching
├── services/            # Service layer (legacy)
├── utils/               # Utility functions
└── visualization/       # Visualization components
```

Total: 35 directories, 137 source files

### Original Test Structure (`tests/`) - BEFORE
```
tests/
├── unit/                # 99 test files (93 in root, 6 in subdirs)
│   ├── application/     # 1 test file
│   ├── data/           # 2 test files
│   ├── domain/         # 2 test files
│   └── ml/             # 1 test file
├── api/                # 8 test files (integration-style)
├── integration/        # 9 test files
├── smoke/              # 6 test files
├── behavior/           # 1 test file
├── benchmarks/         # 4 test files
├── archive/            # Deprecated tests
└── fixtures/           # Test fixtures and data
```

Total BEFORE: 134 test files

### Current Test Structure (AFTER Reorganization)
```
tests/
├── unit/                # 89 test files (0 in root! All organized)
│   ├── api/            # 10 files + routers/
│   ├── application/    # 5 files (config, jobs, pipeline)
│   ├── cli/            # 3 files
│   ├── domain/         # 25 files (organized in subdirs)
│   ├── infra/          # 40 files (ml_models, data, redis, etc.)
│   ├── models/         # 2 files (legacy)
│   ├── presentation/   # 2 files
│   └── utils/          # 3 files
├── api/                # 8 test files (unchanged, integration-style)
├── integration/        # 15 test files (6 moved from unit)
├── quality/            # 1 test file (code_quality)
├── smoke/              # 6 test files (unchanged)
├── behavior/           # 1 test file (unchanged)
├── benchmarks/         # 4 test files (unchanged)
└── fixtures/           # Test fixtures and data
```

Total AFTER: 124 test files (10 deleted: 3 coverage hacks, 7 from archive)

## Problems Identified

### 1. **Structural Misalignment**
- **CRITICAL**: Unit tests don't mirror source structure
- 93 unit tests dumped in `tests/unit/` root directory
- Only 6 tests properly organized in subdirectories
- Source has clean layered architecture, tests are flat chaos

### 2. **Test Location Confusion**
- API tests split between `tests/api/` and `tests/unit/test_api_*.py`
- Domain tests mostly in unit root instead of `tests/unit/domain/`
- Infrastructure tests scattered everywhere

### 3. **Naming Inconsistencies**
- Some tests prefix with category: `test_api_*.py`, `test_eegpt_*.py`
- Others use generic names: `test_config.py`, `test_cli.py`
- No clear naming convention

### 4. **Coverage Gaps**
- `application/` layer: Only 1 test file for entire layer
- `services/` layer: No dedicated test directory
- `visualization/` layer: Tests exist but scattered in unit/
- `cli/` tests: 3 files in unit root, no subdirectory
- `presentation/` tests: 2 report tests in unit root
- `utils/logging/` tests: 2 files in unit root

### 5. **Test Categorization Issues**
- Unclear boundary between unit and integration tests
- Some "unit" tests require external resources
- Smoke tests overlap with integration tests

### 6. **Maintenance Debt**
- `archive/` directory with deprecated tests still present
- Multiple test files for same component (e.g., 3 serialization test files)
- "boost" tests added for coverage but not properly organized

## Professional Test Suite Best Practices

### 1. **Mirror Source Structure**
Tests should mirror the source code structure exactly:
```
tests/
├── unit/
│   ├── api/
│   │   ├── routers/
│   │   └── test_*.py
│   ├── application/
│   │   ├── config/
│   │   ├── factories/
│   │   └── ...
│   ├── domain/
│   └── infra/
```

### 2. **Clear Test Categories**
- **unit/**: Pure unit tests, no external dependencies
- **integration/**: Tests crossing boundaries
- **e2e/**: End-to-end user scenarios
- **performance/**: Performance benchmarks
- **contract/**: API contract tests

### 3. **Naming Conventions**
- Test file: `test_{module_name}.py`
- Test class: `Test{ClassName}`
- Test method: `test_{method_name}_{scenario}_{expected_outcome}`

### 4. **Test Organization Principles**
- One test file per source file
- Tests live next to what they test (in parallel structure)
- Shared fixtures in `conftest.py` at appropriate level
- Test data in `fixtures/` at appropriate level

## Specific Issues Found

### Orphaned Unit Tests (need reorganization)
```
tests/unit/
├── test_abnormality_*.py (5 files)    → should be in unit/domain/abnormal/
├── test_api_*.py (10 files)           → should be in unit/api/
├── test_eegpt_*.py (15 files)         → should be in unit/infra/ml_models/
├── test_preprocessing_*.py (3 files)   → should be in unit/domain/preprocessing/
├── test_redis_*.py (2 files)          → should be in unit/infra/redis/
├── test_serialization_*.py (4 files)  → should be in unit/infra/
├── test_sleep_*.py (4 files)          → should be in unit/domain/sleep/
├── test_yasa_*.py (3 files)           → should be in unit/infra/external/
├── test_cli*.py (3 files)             → should be in unit/cli/
├── test_*report.py (2 files)          → should be in unit/presentation/
├── test_log*.py (2 files)             → should be in unit/utils/logging/
```

### Heavy Tests to Reclassify
```
tests/unit/
├── test_*_real.py (4 files)           → should be in integration/
├── test_train_*.py (1 file)           → should be in integration/
```

### Duplicate/Redundant Tests
- `test_coverage_boost_minimal.py` and `test_coverage_boost_refactored.py`
- `test_eegpt_compat_coverage.py` and `test_eegpt_compat_strict.py`
- Multiple serialization test files that could be consolidated

### Test-to-Test Import Issues
- `test_abnormality_accuracy.py` imports from `test_accuracy_metrics.py`
- This will break when tests move to different directories
- Solution: Extract shared helpers to `tests/_test_utils.py`

### Missing Test Coverage
- No tests for `application/factories/`
- No tests for `application/pipeline/`
- Limited tests for `domain/ports/`
- No tests for `services/` layer

## Recommendations

### Phase 1: Reorganize Existing Tests
1. Create proper directory structure mirroring source
2. Move tests to appropriate locations
3. Consolidate duplicate tests
4. Update imports

### Phase 2: Fill Coverage Gaps
1. Add missing tests for application layer
2. Add tests for service layer
3. Increase domain layer coverage

### Phase 3: Improve Test Quality
1. Separate true unit tests from integration tests
2. Add proper mocking for external dependencies
3. Improve test naming consistency
4. Add docstrings explaining test purpose

### Phase 4: Testing Infrastructure
1. Improve fixture organization
2. Add test utilities module
3. Create test data factories
4. Add performance baseline tests

## Next Steps

1. Create `TEST_SUITE_REORGANIZATION_PLAN.md` with detailed migration plan
2. Create migration script to automate file moves
3. Update all imports after reorganization
4. Run tests to ensure nothing breaks
5. Update CI/CD configuration if needed
