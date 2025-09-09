# Archived TUEV Configs

These configs are outdated and superseded by `tuev_paper_parity.yaml`:

## tuev.yaml
- **Why archived**: Uses 20-channel preprocessing (wrong approach)
- **Problem**: Drops A1/A2/T1/T2, synthesizes Fpz
- **Result**: ~22% BAC (far below 62% target)

## tuev_smoke_test.yaml  
- **Why archived**: Based on old 20-channel approach
- **Problem**: Wrong channel count, outdated paths
- **Superseded by**: tuev_paper_parity.yaml for all testing

## Active Config
Use `../tuev_paper_parity.yaml` which implements:
- 23 input channels (keeps A1/A2/T1/T2)
- Learnable Conv2d(23→20) mapper
- Exact EEGPT paper hyperparameters
- Target: 62.32% BAC