# Technical Debt Implementation Plan

## Executive Summary

This document outlines the TDD-based approach to eliminate remaining technical debt in the Brain-Go-Brrr codebase. Each item includes test specifications, implementation approach, and acceptance criteria.

## Priority Matrix

| Priority | Item | Impact | Effort | Risk |
|----------|------|--------|--------|------|
| 🔴 P0 | TUAB Collate Workaround | High | Low | Medium |
| 🟡 P1 | Channel Routing in API | Medium | Medium | Low |
| 🟡 P2 | EEGPT Model Consolidation | Low | Low | Low |
| 🟢 P3 | Experiment Docs Cleanup | Low | Low | None |
| 🟢 P4 | TestClient File Upload | Low | High | None |

## 🔴 P0: TUAB Collate Workaround Investigation

### Current State
- Collate function contains 20→19 channel truncation code
- Originally for 304 contaminated windows from old cache
- Current logs show NO 20-channel errors
- Workaround may be obsolete

### TDD Test Specification

```python
# tests/unit/test_collate_validation.py

def test_tuab_collate_rejects_20_channels():
    """Collate should reject 20-channel input after workaround removal."""
    batch = [(torch.randn(20, 1024), 0)]
    with pytest.raises(RuntimeError, match="Expected exactly 19 channels"):
        collate_tuab_batch(batch)

def test_tuab_collate_accepts_19_channels():
    """Collate should accept exactly 19 channels."""
    batch = [(torch.randn(19, 1024), 0)]
    data, labels = collate_tuab_batch(batch)
    assert data.shape == (1, 19, 1024)

def test_cache_channel_consistency():
    """Verify no 20-channel windows exist in current cache."""
    cache_dir = Path("data/cache/tuab/mne-ar-v3")
    index_file = cache_dir / "index.json"
    
    with open(index_file) as f:
        index = json.load(f)
    
    twenty_channel_count = 0
    for window_info in index["windows"]:
        window_file = cache_dir / window_info["file"]
        data = torch.load(window_file, weights_only=False)  # nosec:weights_only
        if data["x"].shape[0] == 20:
            twenty_channel_count += 1
    
    assert twenty_channel_count == 0, f"Found {twenty_channel_count} windows with 20 channels"
```

### Implementation Steps

1. **Investigation Phase** (30 min)
   ```bash
   # Script to scan cache for 20-channel windows
   uv run python scripts/scan_cache_channels.py
   ```

2. **If No 20-Channel Windows Found**:
   - Remove workaround code from `collate_tuab.py`
   - Add strict assertion for 19 channels
   - Update tests to verify strict enforcement

3. **If 20-Channel Windows Still Exist**:
   - Document which files contain them
   - Create cache repair script
   - Schedule cache rebuild

### Acceptance Criteria
- [ ] Cache scan completed and documented
- [ ] Tests written and failing (TDD)
- [ ] Workaround removed OR justified with documentation
- [ ] All tests passing
- [ ] No training crashes

## 🟡 P1: Intelligent Channel Routing

### Current State
- API rejects <19 channels with 400 error
- YASA can work with ANY channel count
- Users get poor experience with valid data

### TDD Test Specification

```python
# tests/unit/test_channel_routing.py

@pytest.mark.asyncio
async def test_route_to_yasa_with_few_channels():
    """Should route to YASA when <19 channels."""
    mock_raw = create_mock_raw(n_channels=2)
    
    result = await analyze_eeg(mock_raw)
    
    assert result["pathway"] == "yasa"
    assert "sleep_stages" in result
    assert result["error"] is None

@pytest.mark.asyncio
async def test_route_to_both_with_full_channels():
    """Should offer both pathways with 19+ channels."""
    mock_raw = create_mock_raw(n_channels=19)
    
    result = await analyze_eeg(mock_raw)
    
    assert "eegpt" in result["available_pathways"]
    assert "yasa" in result["available_pathways"]

@pytest.mark.asyncio
async def test_graceful_degradation():
    """Should fall back to YASA if EEGPT fails."""
    mock_raw = create_mock_raw(n_channels=19)
    
    with mock.patch("eegpt_analyze", side_effect=Exception):
        result = await analyze_eeg(mock_raw, prefer="eegpt")
    
    assert result["pathway"] == "yasa"
    assert result["fallback_reason"] == "EEGPT analysis failed"
```

### Implementation Steps

1. **Create Router Service** (2 hours)
   ```python
   # src/brain_go_brrr/services/channel_router.py
   class ChannelRouter:
       def route(self, n_channels: int) -> list[str]:
           if n_channels < 19:
               return ["yasa"]
           return ["eegpt", "yasa"]
   ```

2. **Update API Endpoints** (1 hour)
   - Modify `/api/v1/eeg/analyze` to use router
   - Add pathway selection parameter
   - Return available pathways in response

3. **Add Fallback Logic** (1 hour)
   - Try primary pathway first
   - Fall back on failure
   - Log fallback reasons

### Acceptance Criteria
- [ ] Tests written and failing
- [ ] Router service implemented
- [ ] API endpoints updated
- [ ] Fallback logic working
- [ ] Documentation updated
- [ ] All tests passing

## 🟡 P2: EEGPT Model File Consolidation

### Current State
- 6 files in `infra/ml_models/`
- `eegpt_probe_unified.py` vs `linear_probe.py` overlap
- Training uses `linear_probe.py::TwoLayerProbe`

### TDD Test Specification

```python
# tests/unit/test_probe_compatibility.py

def test_unified_probe_matches_two_layer():
    """Unified probe should produce same results as TwoLayerProbe."""
    input_dim = 2048
    hidden_dim = 256
    output_dim = 2
    
    probe1 = TwoLayerProbe(input_dim, hidden_dim, output_dim)
    probe2 = UnifiedProbe(input_dim, hidden_dim, output_dim, mode="two_layer")
    
    # Copy weights
    probe2.load_state_dict(probe1.state_dict())
    
    x = torch.randn(32, input_dim)
    out1 = probe1(x)
    out2 = probe2(x)
    
    torch.testing.assert_close(out1, out2)

def test_unified_probe_supports_all_modes():
    """Unified probe should support linear and two-layer modes."""
    probe_linear = UnifiedProbe(2048, 256, 2, mode="linear")
    probe_two_layer = UnifiedProbe(2048, 256, 2, mode="two_layer")
    
    assert len(list(probe_linear.parameters())) == 2  # Just W and b
    assert len(list(probe_two_layer.parameters())) == 4  # W1, b1, W2, b2
```

### Implementation Steps

1. **Analysis Phase** (30 min)
   - Compare implementations
   - Check for feature differences
   - Identify dependencies

2. **If Redundant**:
   - Mark one as deprecated
   - Add compatibility layer
   - Update imports

3. **If Different Features**:
   - Document the differences
   - Rename for clarity
   - Add module docstrings

### Acceptance Criteria
- [ ] Analysis documented
- [ ] Tests written
- [ ] Decision made and implemented
- [ ] No breaking changes
- [ ] Documentation updated

## 🟢 P3: Experiment Documentation Cleanup

### Current State
- 3 docs in `experiments/eegpt_linear_probe/docs/`
- May be redundant with root docs
- Need decision on keep/remove

### Implementation Steps

1. **Content Audit** (15 min)
   - Compare with root documentation
   - Identify unique content
   - Check for references

2. **Decision Tree**:
   - If redundant → Delete
   - If unique but outdated → Archive
   - If actively useful → Keep and update header

### Acceptance Criteria
- [ ] Audit completed
- [ ] Decision documented
- [ ] Files handled appropriately
- [ ] No broken references

## 🟢 P4: TestClient File Upload Fix

### Current State
- FastAPI TestClient doesn't handle dependency overrides with file uploads
- Test skipped with `pytest.skip()`
- Low priority as unit tests cover the logic

### TDD Test Specification

```python
# tests/integration/test_file_upload_integration.py

@pytest.mark.integration
async def test_file_upload_with_httpx():
    """Test file upload using httpx instead of TestClient."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        files = {"file": ("test.edf", edf_bytes, "application/octet-stream")}
        response = await client.post("/upload", files=files)
        assert response.status_code == 200
```

### Implementation Options

1. **Option A: Use httpx** (Recommended)
   - Replace TestClient with httpx.AsyncClient
   - More realistic testing
   - Supports dependency overrides

2. **Option B: Service-Level Testing**
   - Test services directly
   - Mock only external dependencies
   - Skip HTTP layer

3. **Option C: Keep Skipped**
   - Document why skipped
   - Ensure unit tests cover logic
   - Revisit if FastAPI fixes issue

### Acceptance Criteria
- [ ] Decision documented
- [ ] Implementation complete (if chosen)
- [ ] Tests passing or properly skipped
- [ ] Documentation updated

## Implementation Schedule

### Week 1 (High Priority)
- **Day 1-2**: P0 - TUAB Collate Investigation
  - Morning: Write tests, scan cache
  - Afternoon: Implement fix or document justification
  
- **Day 3-4**: P1 - Channel Routing
  - Day 3: Write tests, implement router service
  - Day 4: Update API, add fallback logic

- **Day 5**: P2 - Model Consolidation
  - Morning: Analysis and decision
  - Afternoon: Implementation

### Week 2 (Low Priority)
- **Day 1**: P3 - Documentation cleanup
- **Day 2-3**: P4 - TestClient fix (if needed)
- **Day 4-5**: Integration testing and documentation

## Success Metrics

1. **Code Quality**
   - [ ] Zero hardcoded workarounds without justification
   - [ ] All tests passing
   - [ ] No duplicate code

2. **Performance**
   - [ ] No performance degradation
   - [ ] Training stability maintained
   - [ ] API response times unchanged

3. **Developer Experience**
   - [ ] Clear documentation
   - [ ] Intuitive API behavior
   - [ ] Reduced confusion

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Breaking training | Run tests on staging first |
| Cache corruption | Backup cache before modifications |
| API breaking changes | Version endpoints if needed |
| Performance regression | Benchmark before/after |

## Review Checklist

Before marking any item complete:
- [ ] Tests written first (TDD)
- [ ] Tests passing
- [ ] Code reviewed
- [ ] Documentation updated
- [ ] No regression in other areas
- [ ] Changelog updated

## Notes for Senior Auditor

1. **Priority Justification**:
   - P0 is critical as it's blocking code with unknown state
   - P1 improves user experience significantly
   - P2-P4 are nice-to-have cleanups

2. **Testing Philosophy**:
   - TDD for all changes
   - Integration tests for API changes
   - Performance benchmarks for critical paths

3. **Rollback Plan**:
   - Each change in separate PR
   - Feature flags for API changes if needed
   - Cache backup before modifications

---

**Created**: September 2, 2025
**Author**: Technical Debt Taskforce
**Status**: AWAITING SENIOR REVIEW