# Project Status: brain-go-brrr

## Current State (January 2025)

### What This Project Is
A Python framework wrapping the EEGPT foundation model for EEG analysis, focusing on:
- Quality control and artifact detection
- Abnormality classification
- Sleep stage analysis
- Event detection (spikes, seizures)

### What's Actually Implemented
✅ **Completed:**
- Project structure and configuration
- Virtual environment with all dependencies
- Literature review and technical documentation
- Model directory structure
- Reference implementations in `/reference_repos/`
- Service wrappers for QC, sleep, and snippet analysis

⚠️ **In Progress:**
- Core API implementation
- Model integration
- Testing framework

❌ **Not Started:**
- Web frontend
- Database integration
- EMR/FHIR integration
- Deployment infrastructure

### Key Decisions Made

1. **Technology Stack**: Python 3.11+, PyTorch, MNE-Python, FastAPI (planned)
2. **Architecture**: Service-oriented, not full microservices initially
3. **Model**: Using EEGPT large (10M parameters) as primary model
4. **License**: Apache 2.0 (not MIT)

### Directory Structure

```
brain-go-brrr/
├── data/
│   ├── models/
│   │   ├── pretrained/           # Downloaded models (EEGPT here)
│   │   ├── fine_tuned/          # Models we train
│   │   └── external/            # pip-managed models
│   └── datasets/
│       ├── eegpt/               # TUAB, TUEV datasets
│       └── external/            # Other datasets
├── docs/                        # Comprehensive documentation
├── reference_repos/             # Cloned research repos
├── services/                    # Core service implementations
├── examples/                    # Usage examples
└── src/brain_go_brrr/          # Main package code
```

### Model Locations
- **EEGPT Pretrained**: `/data/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt`
- **Config**: Model expects 256Hz, 4-second windows, up to 58 channels

### Next Steps

1. **Immediate**:
   - Download TUAB/TUEV datasets
   - Test model loading and inference
   - Implement basic CLI interface

2. **Short-term**:
   - Build FastAPI endpoints
   - Create Docker container
   - Add comprehensive tests

3. **Medium-term**:
   - Fine-tune for specific use cases
   - Add web interface
   - Deploy MVP

### How to Get Started

```bash
# Activate environment
source .venv/bin/activate  # or use 'uv' commands

# Run tests
uv run pytest

# Check code quality
uv run ruff check .
uv run mypy src/

# Run example pipeline
uv run python examples/end_to_end_pipeline.py
```

### Important Notes

- This is currently a **research prototype**, not a medical device
- Documentation describes future vision, not current state
- Focus on getting core inference working before adding complexity
- EEGPT model requires GPU for reasonable performance

### Contact
For questions about the current implementation vs. documentation, see `/docs/ROUGH_DRAFT.md` for pragmatic approach.
