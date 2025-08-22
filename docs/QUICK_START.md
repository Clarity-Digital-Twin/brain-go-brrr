# Quick Start Guide

Get Brain-Go-Brrr running in 5 minutes.

## Prerequisites

- Python 3.11 or 3.12
- CUDA GPU (optional, for training)
- 16GB RAM minimum
- WSL2 (for Windows users)

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/Clarity-Digital-Twin/brain-go-brrr.git
cd brain-go-brrr
```

### 2. Install UV (Package Manager)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 3. Install Dependencies

```bash
uv sync
```

### 4. Download EEGPT Model

```bash
# Create directories
mkdir -p data/models/pretrained

# Download checkpoint (58MB)
# Place at: data/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt
```

## Running the API

### Start FastAPI Server

```bash
uv run uvicorn brain_go_brrr.api.main:app --reload
```

### Test API

```bash
# Health check
curl http://localhost:8000/health

# API docs
open http://localhost:8000/docs
```

## Running Analysis

### Understanding the Dual Pathways

Brain-Go-Brrr has two PARALLEL processing pathways:

1. **YASA Pathway** (for Sleep-EDF data, 2 channels)
   - Direct sleep staging without EEGPT
   - Optimized for overnight PSG recordings

2. **EEGPT Pathway** (for full EEG, 19+ channels)
   - Extract 512-dim embeddings
   - Feed to task-specific linear probes

### Python Example - Sleep Analysis (YASA)

```python
from brain_go_brrr.services import YASASleepStager
from pathlib import Path
import mne

# Load Sleep-EDF data (2 channels)
edf_path = Path("data/sample.edf")
raw = mne.io.read_raw_edf(edf_path, preload=True)

# Run YASA sleep analysis (no EEGPT needed)
stager = YASASleepStager()
results = stager.stage_sleep(raw)

print(f"Sleep efficiency: {results['efficiency']:.1f}%")
print(f"Sleep stages: {results['stages']}")
```

### Python Example - Parallel Processing

```python
from brain_go_brrr.application.pipeline.parallel import ParallelEEGPipeline

# Run both pathways in parallel
pipeline = ParallelEEGPipeline()
results = pipeline.process(raw)

# EEGPT results (if 19+ channels)
if results["eegpt"]["status"] == "success":
    print(f"EEGPT embeddings: {results['eegpt']['embedding_shape']}")

# YASA results (works with any channel count)
if results["yasa"]["status"] == "success":
    print(f"Sleep stages: {results['yasa']['hypnogram']}")
```

### CLI Usage

```bash
# Run quality control
uv run python -m brain_go_brrr qc analyze data/sample.edf

# Run sleep staging
uv run python -m brain_go_brrr sleep analyze data/sample.edf
```

## Running Tests

```bash
# Quick test (unit tests only)
uv run pytest tests/unit -q

# Full test suite
make test

# With coverage
make coverage
```

## Training Models

### Quick Training (TUAB)

```bash
cd experiments/eegpt_linear_probe
./scripts/launch_tuab.sh

# Monitor progress
tmux attach -t tuab_training
```

## Docker Setup (Optional)

```bash
# Build image
docker build -t brain-go-brrr .

# Run container
docker run -p 8000:8000 brain-go-brrr
```

## Sample Data

Download Sleep-EDF dataset:

```bash
python scripts/download_sleep_edf.py
# Files saved to: data/datasets/external/sleep-edf/
```

## Environment Variables

Create `.env` file:

```bash
# Redis (optional)
REDIS_URL=redis://localhost:6379

# Data paths
BGB_DATA_ROOT=/path/to/data

# GPU settings
CUDA_VISIBLE_DEVICES=0
```

## Common Issues

### WSL Memory Issues

```bash
# Create .wslconfig in Windows home directory
[wsl2]
memory=8GB
swap=0
```

### GPU Not Detected

```bash
# Check CUDA
nvidia-smi

# Force CPU mode
export CUDA_VISIBLE_DEVICES=""
```

### Import Errors

```bash
# Reinstall dependencies
uv sync --refresh
```

## Next Steps

1. Read [ARCHITECTURE.md](ARCHITECTURE.md) for system overview
2. Check [API.md](API.md) for endpoint documentation
3. See [TRAINING.md](TRAINING.md) for model training
4. Review [TESTING.md](TESTING.md) for test guidelines

## Getting Help

- GitHub Issues: [Report bugs](https://github.com/Clarity-Digital-Twin/brain-go-brrr/issues)
- Documentation: [docs/](.)
- Tests: [tests/](../tests)

## License

Apache 2.0 - See [LICENSE](../LICENSE) for details.
