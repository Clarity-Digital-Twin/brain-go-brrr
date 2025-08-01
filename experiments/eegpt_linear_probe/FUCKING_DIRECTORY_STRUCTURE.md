# TUAB DATASET DIRECTORY STRUCTURE - THE ACTUAL FUCKING TRUTH

## CORRECT PATHS - USE THESE OR DIE

```
/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data/datasets/external/tuh_eeg_abnormal/v3.0.1/edf/
├── train/
│   ├── normal/     (2,742 EDF files)
│   └── abnormal/   (2,692 EDF files)
├── eval/
│   ├── normal/     (300 EDF files)
│   └── abnormal/   (252 EDF files)
└── test/
    └── (empty or not used)
```

## THE FUCKING PROBLEM

The TUABDataset expects:
- root_dir: `/path/to/tuh_eeg_abnormal/v3.0.1`
- But the actual structure has an extra `edf/` directory!

## CORRECT ROOT DIRECTORY

```python
# WRONG - WILL CRASH
root_dir = data_root / "datasets/external/tuh_eeg_abnormal/v3.0.1"

# CORRECT - USE THIS
root_dir = data_root / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf"
```

## FILE COUNTS

- Train Normal: 2,742 files
- Train Abnormal: 2,692 files  
- Eval Normal: 300 files
- Eval Abnormal: 252 files
- **TOTAL: 5,986 EDF files**

## EXAMPLE EDF PATH

```
/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data/datasets/external/tuh_eeg_abnormal/v3.0.1/edf/train/normal/01_tcp_ar/aaaaaaaq_s005_t001.edf
```

## DON'T FUCKING FORGET

1. The dataset is under `v3.0.1/edf/` NOT just `v3.0.1/`
2. Each split (train/eval) has normal/abnormal subdirectories
3. Files are further organized by protocol (01_tcp_ar, etc.)