# Literature Collection

This directory contains research papers and documentation related to EEG analysis and foundation models.

## Directory Structure

```
literature/
├── README.md                 # This file
├── pdfs/                    # Original PDF papers
│   ├── ALFEE.pdf           # Adaptive Large Foundation Model for EEG
│   ├── NeuroLM.pdf         # Universal Multi-Task Foundation Model
│   └── EEGPT-summary.md    # EEGPT paper summary
├── markdown/               # Markdown versions for easy reading
│   ├── README.md          # Index of all papers
│   ├── ALFEE/            # ALFEE paper documentation
│   │   ├── README.md     # Clean summary
│   │   └── ALFEE.md      # Auto-converted content
│   └── NeuroLM/          # NeuroLM paper documentation
│       ├── README.md     # Clean summary
│       └── NeuroLM.md    # Auto-converted content
└── convert_pdf_to_md.py    # PDF to markdown converter script
```

## Quick Access

### Foundation Models
- **[ALFEE](markdown/ALFEE/README.md)**: Hybrid attention architecture for robust EEG representation
- **[NeuroLM](markdown/NeuroLM/README.md)**: LLM-based approach treating EEG as language
- **[EEGPT](EEGPT-summary.md)**: Vision transformer baseline for EEG analysis

### Key Insights

1. **Model Scale**: Current state-of-the-art ranges from 10M (EEGPT) to 1.7B (NeuroLM) parameters
2. **Pretraining Data**: 25,000+ hours of EEG recordings becoming standard
3. **Architecture Trends**: Evolution from CNN → Transformer → LLM integration
4. **Performance**: Multi-task models achieving 70-80% accuracy across diverse tasks

## Adding New Papers

To add a new paper to the collection:

1. Place PDF in `pdfs/` directory
2. Run converter: `uv run python convert_pdf_to_md.py`
3. Create clean summary in `markdown/<paper_name>/README.md`
4. Update index in `markdown/README.md`

## Usage in Project

These papers inform our implementation choices:

- **EEGPT Architecture**: Used in `src/brain_go_brrr/infra/ml_models/eegpt_compat.py`
- **Feature Dimensions**: 4 summary tokens × 512 dims = 2,048 total (flattened for linear probes)
- **Training Strategy**: See `experiments/eegpt_linear_probe/` for our implementation

## Related Documentation

- [Project README](../README.md)
- [Architecture Documentation](../docs/ARCHITECTURE.md)
- [Training Guide](../docs/TRAINING.md)
- [CLAUDE.md](../CLAUDE.md) - AI assistant context