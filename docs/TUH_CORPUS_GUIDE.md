# TUH EEG Corpus Download & Setup Guide

## 🏥 Temple University Hospital EEG Corpus Overview

The TUH EEG Corpus is the world's largest publicly available EEG dataset with 26,846 clinical recordings. This guide covers downloading, organizing, and preparing these datasets for training.

## 📊 Available Datasets & Versions

**CRITICAL: Use these EXACT versions to avoid training failures!**

| Dataset | Version | Size | Description | Our Usage |
|---------|---------|------|-------------|-----------|
| **TUAB** | v3.0.1 | ~120GB | Abnormal EEG detection (normal/abnormal) | ✅ Linear probe training |
| **TUEV** | v2.0.1 | ~60GB | Event detection (SPSW, GPED, PLED, etc.) | ✅ Event classification |
| **TUSZ** | v2.0.3 | ~40GB | Seizure detection with annotations | 🔄 Future work |
| **TUEP** | v2.0.1 | ~30GB | Epilepsy classification (100 with/100 without) | 🔄 Future work |
| **TUAR** | v3.0.1 | ~25GB | Artifact detection (5 artifact types) | 🔄 QC pipeline |
| **TUSL** | v2.0.1 | ~20GB | Slowing event detection | 🔄 Future work |
| **TUEG** | v2.0.1 | ~1.5TB | Full corpus (all 26,846 recordings) | 💾 Archive only |

## 🔐 Setup Credentials

1. **Register for access**: https://www.isip.piconepress.com/projects/nedc/html/tuh_eeg/
   - Fill out the form correctly (address must be accurate)
   - Email to: help@nedcdata.org
   - Subject: "Download The TUH EEG Corpus"
   - Wait ~24 hours for credentials

2. **Store credentials securely**:
```bash
# Add to .env file (NEVER commit this!)
echo "TUH_USERNAME=nedc-tuh-eeg" >> .env
echo "TUH_PASSWORD=your_password_here" >> .env
```

## 📥 Download Commands

### Quick Download (Recommended)

```bash
# Download TUAB for abnormality detection
uv run python scripts/data/download_datasets.py TUAB

# Download TUEV for event detection
uv run python scripts/data/download_datasets.py TUEV

# Download everything we use
uv run python scripts/data/download_datasets.py ALL
```

### Manual rsync (If script fails)

```bash
# Test connection first
rsync -auxvL nedc-tuh-eeg@www.isip.piconepress.com:data/tuh_eeg/TEST .

# Download specific corpus
rsync -auxvL nedc-tuh-eeg@www.isip.piconepress.com:data/tuh_eeg/tuh_eeg_abnormal/v3.0.1/ data/datasets/tuab/
rsync -auxvL nedc-tuh-eeg@www.isip.piconepress.com:data/tuh_eeg/tuh_eeg_events/v2.0.1/ data/datasets/tuev/
```

## 📁 Expected Directory Structure

```
data/datasets/
├── tuab/                          # Temple University Abnormal
│   ├── edf/
│   │   ├── train/
│   │   │   ├── normal/
│   │   │   │   └── 01_tcp_ar/    # ~1400 files
│   │   │   └── abnormal/
│   │   │       └── 01_tcp_ar/    # ~1350 files
│   │   └── eval/
│   │       ├── normal/
│   │       │   └── 01_tcp_ar/    # ~150 files
│   │       └── abnormal/
│   │           └── 01_tcp_ar/    # ~130 files
│   └── .download_complete         # Marker file
│
├── tuev/                          # Temple University Events
│   ├── edf/
│   │   ├── train/
│   │   └── dev_test/
│   └── .download_complete
│
└── [other_datasets]/              # Future additions
```

## 🔧 Post-Download Setup

### 1. Build MNE Cache (Preprocessed data)

```bash
# Build TUAB cache with MNE preprocessing
./experiments/eegpt_linear_probe/scripts/build_mne_cache.sh

# Build TUEV cache
./experiments/eegpt_linear_probe/scripts/build_tuev_mne_cache.sh
```

### 2. Verify Dataset Integrity

```bash
# Check TUAB files
find data/datasets/tuab -name "*.edf" | wc -l
# Expected: ~3025 files

# Check TUEV structure
uv run python scripts/data/verify_tuev_dataset.py
```

### 3. Launch Training

```bash
# Train TUAB linear probe
./experiments/eegpt_linear_probe/scripts/launch_tuab_mne.sh

# Train TUEV event classifier
./experiments/eegpt_linear_probe/scripts/launch_tuev_mne.sh
```

## ⚠️ Common Issues & Solutions

### Wrong Version Downloaded
- **Problem**: Downloaded v3.0.0 instead of v3.0.1
- **Solution**: Check version in rsync path, use exact versions above

### Password Authentication Fails
- **Problem**: Special characters in password
- **Solution**: Use raw string in .env: `TUH_PASSWORD=r"your!pass@word"`

### Partial Downloads
- **Problem**: Download interrupted
- **Solution**: rsync resumes automatically, just re-run the command

### Disk Space
- **Problem**: Not enough space for full corpus
- **Solution**: Download only what you need (TUAB + TUEV = ~180GB)

## 📊 Dataset Details

### TUAB (Abnormal EEG)
- **Task**: Binary classification (normal/abnormal)
- **Files**: ~3025 EDF files
- **Duration**: Variable (10min - 1hr recordings)
- **Channels**: 20-30 channels (varies)
- **Sampling**: 250-256 Hz
- **Paper AUROC**: 0.869

### TUEV (EEG Events)
- **Task**: 6-class event detection
- **Classes**: SPSW, GPED, PLED, EYEM, ARTF, BCKG
- **Files**: ~3000 EDF files with annotations
- **Window**: 1-second segments
- **Paper F1**: 0.435

### Channel Mapping (CRITICAL!)
```python
# Old naming (in TUAB files) → Modern naming (EEGPT expects)
CHANNEL_MAPPING = {
    'T3': 'T7',
    'T4': 'T8',
    'T5': 'P7',
    'T6': 'P8'
}
```

## 🚀 Full Corpus Download (Advanced)

If you need the entire TUH EEG Corpus (1.5TB):

```bash
# Download EVERYTHING (requires 2TB free space)
rsync -auxvL nedc-tuh-eeg@www.isip.piconepress.com:data/tuh_eeg/ data/full_corpus/

# Or use physical media (8TB USB drive)
# Ship to: Joseph Picone, 1610 Rhawn Street, Philadelphia, PA 19111
```

## 📚 References

- [TUH EEG Corpus Paper](https://www.sciencedirect.com/science/article/pii/S0165027013003471)
- [TUAB Description (Lopez MS Thesis)](https://www.isip.piconepress.com/publications/ms_theses/)
- [Annotation Guidelines](https://www.isip.piconepress.com/projects/nedc/html/tuh_eeg/)

## 🔒 Security Notes

- **NEVER** commit credentials to git
- **NEVER** hardcode passwords in scripts
- **ALWAYS** use .env file for credentials
- **ALWAYS** use the secure download script

## 📞 Support

- Email: help@nedcdata.org
- Subject must include: "TUH EEG Corpus"
- Response time: ~24-48 hours

---

Last Updated: August 2024
Next Dataset Release Check: January 2025
