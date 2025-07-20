# Security and Linting Suppression Justifications

This document centralizes the rationale for all security (`# nosec`) and linting (`# noqa`) suppressions in the codebase.

## Security Suppressions (Bandit)

### B104: Binding to all interfaces
**Location**: `api/main.py:11`
```python
uvicorn.run(app, host="0.0.0.0", port=8000)  # nosec B104
```
**Justification**: Required for Docker deployment. The container's network isolation provides the security boundary.

### B301: Pickle usage
**Location**: `infra/redis/connection_pool.py:multiple`
```python
pickle.loads(value)  # nosec B301
```
**Justification**: Redis storage requires serialization. We control both ends of the pickle operation and only pickle trusted data from our own application.

## Linting Suppressions (Ruff)

### E402: Import after code
**Location**: Various example scripts
```python
sys.path.insert(0, str(project_root))
from brain_go_brrr.models import ...  # noqa: E402
```
**Justification**: Example scripts need to modify sys.path to import the main package when run directly. This will be removed once proper packaging is implemented.

### PLR0913: Too many arguments
**Location**: `core/quality.py:run_full_qc_pipeline`
```python
def run_full_qc_pipeline(self, raw, ...):  # noqa: PLR0913
```
**Justification**: EEG processing requires many configuration parameters. Refactoring to use a config object is planned for future sprints.

## Type Checking Suppressions (mypy)

### ignore[literal-required]
**Location**: `core/jobs/store.py:72`
```python
job[key] = value  # type: ignore[literal-required]
```
**Justification**: TypedDict doesn't support dynamic key access, but we need it for the update operation. Keys are validated at runtime.

## Migration Plan

1. **Example scripts**: Will be fixed by proper packaging (tracked in todo)
2. **Pickle usage**: Consider moving to JSON for Redis storage in v2
3. **Function arguments**: Refactor to use config objects in next major version
