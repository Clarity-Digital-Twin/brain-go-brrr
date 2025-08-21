# Data Transfer Setup Guide

**Purpose**: Instructions for setting up Brain-Go-Brrr with pre-downloaded data package

## 📦 What's in the transfer package

The `brain-go-brrr-DATA-ONLY-*.tar.gz` contains:

```
📦 Package Contents
├── data/
│   ├── models/
│   │   └── eegpt/
│   │       └── pretrained/
│   │           └── eegpt_mcae_58chs_4s_large4E.ckpt (973MB)
│   ├── cache/
│   │   └── tuab_preprocessed/ (1.1GB - preprocessed windows)
│   └── datasets/
│       └── external/
│           └── tuh_eeg_abnormal/ (79GB - full TUAB dataset)
│               └── v3.0.1/
│                   └── edf/
│                       ├── train/
│                       ├── eval/
│                       └── test/
├── experiments/
│   └── eegpt_linear_probe/
│       └── logs/ (training logs/checkpoints)
├── .env (environment variables)
└── reference_repos_list.txt (clone instructions)
```

## 🎯 Where everything goes

After cloning the repo and extracting the package:

```bash
brain-go-brrr/                    # Repository root
├── data/                         # ← Extract data/ here
│   ├── models/                   # Model weights
│   ├── cache/                    # Preprocessed data
│   └── datasets/                 # Raw datasets
├── experiments/                  # ← Logs go here
├── reference_repos/              # ← Clone repos here (empty, needs cloning)
├── src/                          # Source code (from git)
├── tests/                        # Tests (from git)
├── docs/                         # Documentation (from git)
└── .env                          # ← Place at root
```

## 📋 Step-by-step setup

```bash
# 1. Clone repository
git clone https://github.com/Clarity-Digital-Twin/brain-go-brrr.git
cd brain-go-brrr
git checkout development

# 2. Extract data package (preserves directory structure)
tar -xzf /path/to/brain-go-brrr-DATA-ONLY-*.tar.gz

# 3. Verify key files exist
ls -la data/models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt
ls -la data/cache/tuab_preprocessed/ | head
ls -la data/datasets/external/tuh_eeg_abnormal/v3.0.1/

# 4. Clone reference repos
mkdir -p reference_repos
cd reference_repos
git clone https://github.com/ncclab/EEGPT.git
git clone https://github.com/mne-tools/mne-python.git
git clone https://github.com/raphaelvallat/yasa.git
git clone https://github.com/autoreject/autoreject.git
cd ..

# 5. Set up Python environment
conda create -n eegpt python=3.11
conda activate eegpt
pip install -r requirements.txt

# 6. Run training
cd experiments/eegpt_linear_probe
python train_final_proper.py
```

## ✅ Verification checklist

After extraction, verify these critical paths exist:
- [ ] `data/models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt`
- [ ] `data/cache/tuab_preprocessed/*.pkl` (thousands of files)
- [ ] `data/datasets/external/tuh_eeg_abnormal/v3.0.1/edf/train/`
- [ ] `.env` (at repository root)
- [ ] `reference_repos/EEGPT/` (after cloning)

## 🚀 Ready to train!

The PC should now be an exact replica of the Mac setup, ready for GPU training.
