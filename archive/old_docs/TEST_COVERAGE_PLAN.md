# Test Coverage Improvement Plan

## Current Status
- **Coverage**: 55.59% (up from 52.95%)
- **Total Tests**: 537 passing, 29 skipped
- **Recent Improvements**: 
  - api/cache.py: 95%
  - api/routers/eegpt.py: 89.82%
  - api/app.py: 100%

## Auditor Feedback Addressed

### ✅ Fixed Issues
1. **Deterministic tests** - Removed all RNG, using fixed values
2. **Makefile coverage** - Added PYTEST_WITH_COV for explicit plugin loading
3. **TC004 cleanup** - Per-file ignores in pyproject.toml
4. **Benchmark artifact** - bench-local.json created

### ⚠️ To Address
1. **Private globals reset** - Need clean API for test state reset
2. **Full benchmark run** - Partial artifact only, need complete run on ext4
3. **Health endpoint** - Remove coupling to optional keys
4. **Spec'd mocks** - Consider lighter spec_set instead of full spec

## Priority Targets for Clean Coverage

### Tier 1: Quick Wins (Can reach 60% total)
1. **api/routers/resources.py** (12 lines) - Simple endpoints
2. **api/routers/cache.py** (21 lines) - Cache management 
3. **api/routers/queue.py** (24 lines) - Queue status

### Tier 2: Critical Path (Can reach 65% total)
1. **cli.py** (43 lines) - Argument parsing only, no model loads
2. **api/routers/jobs.py** (47 lines) - Job management
3. **core/jobs/store.py** (83 lines) - Job storage logic

### Tier 3: Medical Critical (Must have >80%)
1. **core/abnormal/detector.py** (156 lines) - CRITICAL for patient safety
2. **core/features/extractor.py** (92 lines) - Feature extraction
3. **core/pipeline/parallel.py** (66 lines) - Processing pipeline

## Testing Principles

### DO
- Test real logic, not mocks
- Use fixtures for common test data
- Keep tests fast (<1s each)
- Test error paths and edge cases
- Use deterministic values

### DON'T
- Import heavy models in unit tests
- Use RNG without seeding
- Create files without cleanup
- Test implementation details
- Mock everything

## Implementation Strategy

### Phase 1: Quick Wins (Today)
```bash
# Add simple router tests
make test-unit-cov  # Target: 58%
```

### Phase 2: CLI Tests (Tomorrow)
```bash
# Test CLI argument parsing only
uv run pytest tests/unit/test_cli.py --cov=brain_go_brrr.cli
```

### Phase 3: Critical Modules (This Week)
```bash
# Focus on medical-critical paths
make test-all-cov  # Target: 65%
```

## Coverage Gates

### Immediate
- Block PRs that decrease coverage
- Warning at <55%

### Next Sprint
- Fail CI at <60%
- Target 65% by end of sprint

### Q1 2025
- Target 75% overall
- 90% for medical-critical modules

## Test Categories

### Unit Tests (Fast)
- No external dependencies
- Mock I/O operations
- <1s execution time
- Run on every commit

### Integration Tests (Medium)
- Test module interactions
- Use test databases
- <10s execution time
- Run on PR

### E2E Tests (Slow)
- Full pipeline tests
- Real model loading
- >10s execution time
- Run nightly

## Monitoring

```bash
# Check coverage trends
make cov  # Quick check
make test-all-cov  # Full report
coverage html  # Detailed HTML report
```

## Next Actions

1. [ ] Create reset_state_for_tests() helper
2. [ ] Add CLI smoke tests (--help only)
3. [ ] Test api/routers/resources.py
4. [ ] Test api/routers/cache.py
5. [ ] Add coverage gate to CI
6. [ ] Run full benchmark suite on ext4
7. [ ] Document coverage requirements

## Success Metrics

- **Week 1**: 60% coverage
- **Week 2**: 65% coverage
- **Month 1**: 70% coverage
- **Quarter**: 75% coverage

## Risk Mitigation

- **Test hangs**: Use pytest-timeout
- **Flaky tests**: Fix or mark as flaky
- **Slow tests**: Parallelize with pytest-xdist
- **Heavy imports**: Mock at module boundary