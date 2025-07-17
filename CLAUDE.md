# CLAUDE.md - Brain-Go-Brrr Project

## Critical Context
This is a medical-adjacent EEG analysis system using the EEGPT foundation model. While not FDA-approved, code quality matters - bugs could impact clinical decisions. Always prioritize safety and accuracy over speed.

## Project Overview
- **Purpose**: Python wrapper around EEGPT for EEG analysis (QC, abnormality detection, sleep staging)
- **Model**: EEGPT 10M parameters at `/data/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt`
- **Key Docs**: Read `/docs/literature-master-reference.md` for technical specs
- **Architecture**: Service-oriented (not microservices yet), FastAPI + PyTorch + MNE

## Development Commands
```bash
# Environment
uv sync                    # Install/update dependencies
uv run python             # Run Python in project env

# Quality Checks (run after changes)
uv run ruff check src/    # Linting
uv run mypy src/          # Type checking
uv run pytest tests/ -xvs # Run tests (stop on first failure)

# Development
make dev-setup            # Full dev environment setup
make test                 # Run all tests
make format               # Auto-format code
```

## Code Style Rules
- Python 3.11+ with type hints ALWAYS
- Use `pathlib.Path`, never string paths
- Async/await for all I/O operations
- SOLID principles - inject dependencies
- Never hardcode paths - use config
- Log errors, but NEVER log PHI/patient data

## EEG-Specific Guidelines
- Standard sampling rate: 256 Hz (EEGPT), 100 Hz (Sleep-EDF)
- Window size: 4 seconds for EEGPT
- Channel names: Use 10-20 system (Fp1, Fp2, etc.)
- Always validate EEG data before processing
- Handle missing channels gracefully

## File Structure
```
src/brain_go_brrr/    # Main package code
services/             # High-level service implementations
tests/                # Test files (mirror src structure)
data/models/          # Model weights (gitignored)
data/datasets/        # EEG data (gitignored)
docs/                 # Documentation (PRD, TRD, BDD)
```

## Testing Approach
1. Write test FIRST (TDD)
2. Use pytest fixtures for EEG data
3. Mock external services
4. Test error cases explicitly
5. Run `uv run pytest tests/test_specific.py::test_name` for single tests

## Common Tasks

### Adding New EEG Analysis Feature
```bash
think hard about the feature requirements
# 1. Create test file: tests/test_new_feature.py
# 2. Write failing tests with expected behavior
# 3. Implement in src/brain_go_brrr/
# 4. Ensure tests pass
# 5. Update documentation
```

### Processing Sleep-EDF Data
```python
# Data location: /data/datasets/external/sleep-edf/
# Use existing services:
from services.sleep_metrics import SleepAnalyzer
analyzer = SleepAnalyzer()
results = analyzer.run_full_sleep_analysis(raw)
```

### Working with EEGPT Model
```python
# Model path is in config, never hardcode
# Check /services/qc_flagger.py for example
# Always handle GPU/CPU fallback
```

## Safety Requirements
- Validate all inputs rigorously
- Return confidence scores with predictions
- Flag low-confidence results for human review
- Never crash - return safe defaults
- Log all predictions for audit trail

## API Development
- Use FastAPI with Pydantic models
- Version all endpoints (/api/v1/)
- Return structured JSON responses
- Include correlation IDs for tracing
- Document with OpenAPI annotations

## Git Workflow
- Feature branches: `feature/add-sleep-staging`
- Commit format: `type(scope): description`
- Always run tests before committing
- Update CHANGELOG.md for features

## Performance Targets
- Process 20-min EEG in <2 minutes
- Support 50 concurrent analyses
- Use chunking for large files
- Cache preprocessed data when possible

## When Stuck
1. Check `/docs/literature-master-reference.md` for technical details
2. Look at existing implementations in `/services/`
3. Review `/reference_repos/` for original code
4. Read the actual research papers in `/literature/markdown/`

## Do NOT
- Log patient identifiers
- Hardcode file paths
- Skip input validation
- Ignore error handling
- Make synchronous I/O calls
- Modify data in-place
- Create files without user request

## Thinking Triggers
Use these phrases for complex problems:
- "think" - Standard analysis
- "think hard" - Deeper evaluation
- "think harder" - Extensive analysis
- "ultrathink" - Maximum depth

## Quick Debug Commands
```bash
# Check GPU availability
uv run python -c "import torch; print(torch.cuda.is_available())"

# Test model loading
uv run python scripts/test_sleep_analysis.py

# Verify environment
uv run python -c "import mne, torch, numpy; print('Environment OK')"
```

Remember: This handles brain data. Accuracy matters. Test everything.