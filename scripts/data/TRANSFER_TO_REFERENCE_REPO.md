# 📦 TUEV Download Setup for Reference EEGPT Repository

## Files to Transfer

Transfer these files to `reference_repos/EEGPT/`:

1. **`.env`** - Contains TUH credentials
   ```bash
   cp .env reference_repos/EEGPT/.env
   ```

2. **Download script** (choose one):
   - **Python version** (recommended - handles password automatically):
     ```bash
     cp scripts/data/download_tuev_for_reference.py reference_repos/EEGPT/
     ```
   - **Bash version** (requires manual password entry):
     ```bash
     cp scripts/data/download_tuev_for_reference.sh reference_repos/EEGPT/
     ```

## Setup in Reference Repo

```bash
# Navigate to reference repo
cd reference_repos/EEGPT/

# Copy files from main repo
cp ../../.env .
cp ../../scripts/data/download_tuev_for_reference.py .

# Install dependencies (if using Python script)
pip install pexpect python-dotenv

# Run download
python download_tuev_for_reference.py
# OR
bash download_tuev_for_reference.sh
```

## What Gets Downloaded

- **Dataset**: TUEV v2.0.1 (Event detection)
- **Location**: `data/datasets/tuev/`
- **Size**: ~15GB
- **Structure**:
  ```
  data/datasets/tuev/
  └── v2.0.1/
      └── edf/
          ├── train/   # Training set
          ├── dev/     # Development/validation set
          └── test/    # Test set
  ```

## Credentials Required

The `.env` file must contain:
```bash
TUH_USERNAME=nedc-tuh-eeg
TUH_PASSWORD=K4!Tf#6Y$8qLpNcR
```

## Running Reference Implementation

After download completes:

1. **Update data paths** in reference config:
   ```python
   # In reference EEGPT config
   data_dir = "data/datasets/tuev"
   ```

2. **Run their exact implementation**:
   ```bash
   python train_tuev.py  # Or whatever their training script is
   ```

3. **Compare results** with our implementation

## Troubleshooting

- **Connection refused**: Check internet/firewall
- **Permission denied**: Verify credentials in .env
- **Disk space**: Need ~15GB free
- **Resume interrupted download**: Just run script again

## Expected Outcome

With the exact same dataset and reference implementation, we can verify:
1. Whether they achieve 62.32% BAC
2. How they handle class imbalance
3. Any hidden preprocessing steps
4. Actual training configuration