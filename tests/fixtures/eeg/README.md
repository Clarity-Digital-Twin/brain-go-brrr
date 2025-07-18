# EEG Test Fixtures

Synthetic EEG test fixtures for unit testing the abnormality detection system.

## Files

- `tuab_001_norm_30s.fif` - Normal EEG, 30 seconds
- `tuab_002_norm_30s.fif` - Normal EEG, 30 seconds
- `tuab_003_abnorm_30s.fif` - Abnormal EEG with slowing and spikes
- `tuab_004_abnorm_30s.fif` - Abnormal EEG with rhythmic discharges
- `tuab_005_short_10s.fif` - Short normal EEG for edge case testing

Each 30s fixture also has a 5s version for faster tests.

## Note

These are synthetic fixtures for testing only. Use real TUAB data for validation.
