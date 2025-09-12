# 📚 Literature Collection

This directory contains research papers and documentation related to EEG analysis and foundation models.

## 🚀 PDF to Markdown Converter (2025 Edition)

**NEW!** We now use `PyMuPDF4LLM` - the best PDF to markdown converter for scientific papers.

### Quick Convert

```bash
# Convert single PDF
uv run python pdf_to_markdown.py pdfs/EEGPT.pdf

# Convert all PDFs
uv run python pdf_to_markdown.py --all

# Force reconvert (overwrite existing)
uv run python pdf_to_markdown.py --all --overwrite

# Custom settings (e.g., higher DPI for figures)
uv run python pdf_to_markdown.py pdfs/paper.pdf --config '{"dpi": 300}'

# Skip image extraction
uv run python pdf_to_markdown.py pdfs/paper.pdf --no-images
```

### Features
- ✅ Preserves equations, tables, and formatting
- ✅ Extracts images with proper naming
- ✅ Optimized for LLM consumption
- ✅ Batch processing support
- ✅ Metadata tracking (conversion time, settings, etc.)

## 📁 Directory Structure

```
literature/
├── README.md                   # This file
├── pdf_to_markdown.py         # 🚀 Universal converter (use this!)
├── _archive/                  # Old conversion scripts (deprecated)
├── pdfs/                      # Original PDF papers
│   ├── ALFEE.pdf             # Adaptive Large Foundation Model for EEG
│   ├── AUTOREJECT.pdf        # Automated artifact rejection
│   ├── EEGPT.pdf             # EEG Pre-trained Transformer
│   ├── MNE-Python.pdf        # MNE-Python documentation
│   ├── MNE-SOFTWARE.pdf      # MNE software paper
│   ├── NeuroLM.pdf           # Universal Multi-Task Foundation Model
│   ├── seizure_preprocessing.pdf  # Seizure detection preprocessing
│   └── SEIZURE_TRANSFORMER.pdf    # SeizureTransformer (Wu et al. 2025)
└── markdown/                  # Converted markdown versions
    ├── ALFEE/                # Each PDF gets its own directory
    │   ├── ALFEE.md         # Converted markdown
    │   ├── *.png            # Extracted images
    │   └── metadata.json    # Conversion metadata
    ├── EEGPT/
    ├── seizure_preprocessing/
    ├── seizure_transformer/
    └── ...
```

## 📖 Quick Access to Papers

### Foundation Models
- **[EEGPT](markdown/EEGPT/)**: Vision transformer for EEG (10M params) - Our primary model
- **[ALFEE](markdown/ALFEE/)**: Hybrid attention architecture for robust EEG representation
- **[NeuroLM](markdown/NeuroLM/)**: LLM-based approach treating EEG as language (1.7B params)
- **[SeizureTransformer](markdown/seizure_transformer/)**: State-of-the-art seizure detection (2025)

### EEG Processing & Tools
- **[MNE-Python](markdown/MNE-Python/)**: Core EEG processing library
- **[Autoreject](markdown/autoreject/)**: Automated artifact rejection
- **[Seizure Preprocessing](markdown/seizure_preprocessing/)**: Preprocessing pipeline for seizure detection

### Key Papers for Our Implementation

1. **EEGPT** - Core architecture we're using
   - 4 summary tokens × 512 dims = 2,048 feature dimensions
   - Pretrained on 25,000+ hours of EEG
   - See: `src/brain_go_brrr/infra/ml_models/eegpt_compat.py`

2. **SeizureTransformer** - Latest seizure detection
   - AUROC: 0.876 on TUSZ test set
   - Uses 60-second windows, 19 channels
   - See: `CURRENT_SEIZURE_TRANSFORMER_DATAFLOW.md`

3. **Autoreject** - Quality control
   - Bad channel detection
   - Artifact rejection
   - See: `src/brain_go_brrr/infra/preprocessing/autoreject_adapter.py`

## 🔧 Adding New Papers

1. **Add PDF**: Place in `pdfs/` directory
2. **Convert**: `uv run python pdf_to_markdown.py pdfs/new_paper.pdf`
3. **Review**: Check `markdown/new_paper/` for quality
4. **Document**: Update this README with key insights

## 📊 Key Insights from Literature

### Model Evolution
- **2020**: CNN-based (EEGNet, ~50k params)
- **2023**: Transformer-based (EEGPT, 10M params)
- **2024**: LLM-based (NeuroLM, 1.7B params)
- **2025**: Hybrid approaches (SeizureTransformer)

### Performance Benchmarks
- **Sleep Staging**: 87% accuracy (YASA, EEGPT)
- **Abnormality Detection**: 87% AUROC (EEGPT on TUAB)
- **Seizure Detection**: 87.6% AUROC (SeizureTransformer on TUSZ)
- **Artifact Rejection**: 87.5% expert agreement (Autoreject)

### Preprocessing Standards
- **Sampling Rate**: 256 Hz (standard for transformers)
- **Filtering**: 0.5-50 Hz bandpass typical
- **Normalization**: Z-score per channel
- **Window Size**: 4s (EEGPT), 60s (SeizureTransformer)

## 🔗 Related Documentation

- [Project README](../README.md) - Main project overview
- [CLAUDE.md](../CLAUDE.md) - AI assistant context & rules
- [Architecture](../docs/ARCHITECTURE.md) - System design
- [Training Guide](../docs/TRAINING.md) - Model training
- [API Documentation](../docs/API.md) - REST endpoints

## 💡 Pro Tips

1. **Batch Convert**: Use `--all` flag to convert everything at once
2. **High Quality**: Use `--config '{"dpi": 300}'` for papers with detailed figures
3. **Fast Mode**: Use `--no-images` if you only need text
4. **Check Metadata**: Each conversion creates `metadata.json` with stats

---
*Last updated: December 2025 | Using PyMuPDF4LLM v0.0.27*