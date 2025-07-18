# GitHub Actions Workflow Analysis & Recommendations

## Current Workflows

### 1. `claude.yml` - Claude PR Action
- Triggers on issue comments and PR comments containing `@claude`
- Uses anthropics/claude-code-action for automated code assistance
- Good for AI-assisted development

### 2. `ci.yml` - Main CI/CD Pipeline
Comprehensive pipeline with:
- ✅ Code quality checks (ruff, mypy, bandit)
- ✅ Multi-OS and multi-Python version testing
- ✅ Security scanning with Trivy
- ✅ Documentation building
- ✅ Docker build testing
- ✅ Automated releases

### 3. `auto-doc.yml` - Documentation generation
- Appears to exist but wasn't reviewed

## Recommendations for Improvement

### 1. Add Integration Test Job
```yaml
integration-tests:
  name: Integration Tests
  runs-on: ubuntu-latest
  needs: test
  services:
    redis:
      image: redis:7-alpine
      options: >-
        --health-cmd "redis-cli ping"
        --health-interval 10s
        --health-timeout 5s
        --health-retries 5
      ports:
        - 6379:6379
  steps:
    - uses: actions/checkout@v4
    - name: Install uv
      uses: astral-sh/setup-uv@v2
    - name: Set up Python
      run: uv python install ${{ env.PYTHON_VERSION }}
    - name: Install dependencies
      run: uv sync
    - name: Run integration tests
      run: uv run pytest tests/integration -v --maxfail=3
      env:
        REDIS_HOST: localhost
        REDIS_PORT: 6379
```

### 2. Add Benchmark Performance Tests
```yaml
benchmarks:
  name: Performance Benchmarks
  runs-on: ubuntu-latest
  needs: test
  if: github.event_name == 'pull_request'
  steps:
    - uses: actions/checkout@v4
    - name: Install uv
      uses: astral-sh/setup-uv@v2
    - name: Set up Python
      run: uv python install ${{ env.PYTHON_VERSION }}
    - name: Install dependencies
      run: uv sync
    - name: Run benchmarks
      run: |
        uv run pytest tests/benchmarks --benchmark-only --benchmark-json=benchmark.json
    - name: Store benchmark result
      uses: benchmark-action/github-action-benchmark@v1
      with:
        tool: 'pytest'
        output-file-path: benchmark.json
        github-token: ${{ secrets.GITHUB_TOKEN }}
        auto-push: false
        comment-on-alert: true
        alert-threshold: '150%'
        fail-on-alert: true
```

### 3. Add EEG Data Validation
```yaml
- name: Validate EEG test data
  run: |
    # Check if Sleep-EDF data exists
    if [ -d "data/datasets/external/sleep-edf" ]; then
      echo "Sleep-EDF data found"
      ls -la data/datasets/external/sleep-edf/sleep-cassette/*.edf | head -5
    else
      echo "::warning::Sleep-EDF data not found - some tests may be skipped"
    fi
```

### 4. Cache Model Weights
```yaml
- name: Cache EEGPT model
  uses: actions/cache@v3
  with:
    path: data/models/pretrained
    key: eegpt-model-${{ hashFiles('**/model_requirements.txt') }}
    restore-keys: |
      eegpt-model-
```

### 5. Add GPU Testing (Optional)
```yaml
gpu-test:
  name: GPU Tests
  runs-on: [self-hosted, gpu]  # Requires self-hosted runner with GPU
  needs: test
  if: github.event_name == 'push' && contains(github.event.head_commit.message, '[gpu]')
  steps:
    # ... setup steps
    - name: Run GPU tests
      run: |
        uv run pytest tests -v -m gpu --gpu-memory-fraction 0.5
      env:
        CUDA_VISIBLE_DEVICES: 0
```

### 6. Improve Test Result Reporting
```yaml
- name: Test Report
  uses: dorny/test-reporter@v1
  if: always()
  with:
    name: Test Results
    path: 'test-results.xml'
    reporter: java-junit
    fail-on-error: true
```

### 7. Add Dependency Update Checks
Create `.github/workflows/dependency-update.yml`:
```yaml
name: Dependency Update Check
on:
  schedule:
    - cron: '0 0 * * MON'  # Weekly on Monday
  workflow_dispatch:

jobs:
  update-dependencies:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        uses: astral-sh/setup-uv@v2
      - name: Update dependencies
        run: |
          uv lock --upgrade
          uv sync
      - name: Run tests
        run: uv run pytest tests -v
      - name: Create PR
        uses: peter-evans/create-pull-request@v5
        with:
          commit-message: 'chore: update dependencies'
          title: 'Weekly dependency updates'
          body: |
            ## Dependency Updates

            This PR contains the latest dependency updates.
            All tests have been run to ensure compatibility.
          branch: deps/weekly-update
```

### 8. Add Memory Leak Detection
```yaml
- name: Check for memory leaks
  run: |
    uv run pytest tests/benchmarks/test_eegpt_performance.py::TestMemoryBenchmarks -v
    uv run python -m memory_profiler scripts/check_memory_usage.py
```

## Quick Wins

1. **Update UV version**: The workflow uses UV 0.1.0, but newer versions are available
2. **Add workflow concurrency** to cancel old runs:
```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
```

3. **Add test splitting** for faster CI:
```yaml
- name: Run tests in parallel
  run: |
    uv run pytest tests -v --splits 4 --group ${{ matrix.group }}
  strategy:
    matrix:
      group: [1, 2, 3, 4]
```

4. **Add status badges** to README:
```markdown
[![CI/CD](https://github.com/Clarity-Digital-Twin/brain-go-brrr/actions/workflows/ci.yml/badge.svg)](https://github.com/Clarity-Digital-Twin/brain-go-brrr/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Clarity-Digital-Twin/brain-go-brrr/branch/main/graph/badge.svg)](https://codecov.io/gh/Clarity-Digital-Twin/brain-go-brrr)
```

## Security Improvements

1. **Add SAST scanning**:
```yaml
- name: Run Semgrep
  uses: returntocorp/semgrep-action@v1
  with:
    config: >-
      p/security-audit
      p/python
```

2. **Add secret scanning**:
```yaml
- name: Scan for secrets
  uses: trufflesecurity/trufflehog@main
  with:
    path: ./
    base: ${{ github.event.repository.default_branch }}
```

## Next Steps

1. Implement integration test job with Redis service
2. Add benchmark performance tracking
3. Set up dependency update automation
4. Configure workflow concurrency
5. Add GPU testing when self-hosted runners available
