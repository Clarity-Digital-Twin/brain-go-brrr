# Security and Linting Suppression Justifications

This document centralizes the rationale for all security (`# nosec`) and linting (`# noqa`) suppressions in the codebase.

## Security Suppressions (Bandit)

### B104: Binding to all interfaces
**Location**: `api/main.py:11`
```python
uvicorn.run(app, host="0.0.0.0", port=8000)  # nosec B104
```
**Justification**: Required for Docker deployment. The container's network isolation provides the security boundary.

### B614: Loading PyTorch models without weights_only
**Location**:
- `src/brain_go_brrr/models/eegpt_architecture.py:458`
- `scripts/validate_data_pipeline.py:138`
- `scripts/inspect_eegpt_checkpoint.py:21`

```python
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)  # nosec B614
```
**Justification**: Loading pretrained EEGPT model weights from trusted sources. The model files contain legitimate PyTorch objects beyond just tensors (optimizer states, config, etc.) that require full deserialization.

## Linting Suppressions (Ruff)

### E402: Import after code
**Location**: Various example scripts
```python
sys.path.insert(0, str(project_root))
from brain_go_brrr.models import ...  # noqa: E402
```
**Justification**: Example scripts need to modify sys.path to import the main package when run directly. This will be removed once proper packaging is implemented.

### ARG002: Unused function arguments
**Location**:
- `core/snippets/maker.py:364`
- `api/cache.py:22`
- `core/quality/controller.py:468`
- `src/brain_go_brrr/visualization/markdown_report.py:94`
- `api/routers/cache.py:71`

```python
def analyze_snippet_with_eegpt(self, snippet: dict, model_path: Path | None = None) -> dict:  # noqa: ARG002
```
**Justification**: These methods accept arguments for API compatibility or future extensibility. The unused parameters maintain consistent interfaces across the codebase.

## Type Checking Suppressions (mypy)

### ignore[literal-required]
**Location**:
- `core/jobs/store.py:71`
- `core/jobs/store.py:102`

```python
job[key] = value  # type: ignore[literal-required]
```
**Justification**: TypedDict doesn't support dynamic key access, but we need it for the update/patch operations. Keys are validated at runtime.

### ignore[unreachable]
**Location**: `src/brain_go_brrr/models/eegpt_model.py:232`
```python
return np.zeros(...)  # type: ignore[unreachable]
```
**Justification**: Defensive programming - provides a fallback return even after error logging.

### ignore[assignment, misc, no-any-return]
**Location**:
- `core/quality/controller.py:28` - AutoReject optional import
- `core/quality/controller.py:225` - Return type inference
- `infra/redis/pool.py:249-250` - Redis info() method typing
- `api/routers/resources.py:10` - GPUtil optional import

**Justification**: Third-party library typing inconsistencies and optional imports.

## Migration Plan

1. **Example scripts**: Will be fixed by proper packaging (tracked in todo)
2. **Pickle usage**: Consider moving to JSON for Redis storage in v2
3. **Function arguments**: Refactor to use config objects in next major version
