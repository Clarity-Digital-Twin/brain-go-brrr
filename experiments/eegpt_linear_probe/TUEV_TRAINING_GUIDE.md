# TUEV Paper Parity Training Guide

## Overview
This guide helps OSS contributors reproduce the TUEV paper parity results (62.32% balanced accuracy) using the EEGPT model with a learnable 23→20 channel mapper.

## Prerequisites

1. **Environment Setup**
   ```bash
   # Install dependencies
   uv sync
   
   # Set environment variables
   export BGB_DATA_ROOT=/path/to/your/data
   export BGB_CACHE_DIR=$BGB_DATA_ROOT/cache
   ```

2. **Data Requirements**
   - TUEV dataset in `$BGB_DATA_ROOT/datasets/tuev/`
   - EEGPT checkpoint at `$BGB_DATA_ROOT/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt`

## Building the Cache

Before training, build the 23-channel cache for paper parity:

```bash
cd experiments/eegpt_linear_probe
./scripts/build_tuev_cache.sh
```

This creates cached windows at `$BGB_CACHE_DIR/tuev_23ch_paper_parity/` with:
- 23 channels (including A1, A2, T1, T2)
- 4-second windows at 256Hz
- Data in Volts (SI units)

## Training

### Quick Start
```bash
# Launch training in tmux (recommended)
./scripts/launch_tuev_paper_parity.sh
```

### Manual Training
```bash
python train_tuev_mne.py \
    --config configs/tuev_paper_parity.yaml \
    --cache-dir $BGB_CACHE_DIR \
    --output-dir output/tuev_paper_parity_$(date +%Y%m%d_%H%M%S)
```

### Resume from Checkpoint
```bash
python train_tuev_mne.py \
    --config configs/tuev_paper_parity.yaml \
    --cache-dir $BGB_CACHE_DIR \
    --output-dir output/existing_run \
    --resume output/existing_run/checkpoint_epoch_10.pt
```

## Key Configuration

The `configs/tuev_paper_parity.yaml` file contains critical settings:

```yaml
data:
  use_paper_parity: true  # Enables 23-channel mode
  n_channels: 23          # Keep ALL channels
  
model:
  use_channel_mapper: true   # 23→20 learnable mapping
  mapper_dropout: 0.8        # From EEGPT reference
  
training:
  learning_rate: 5.0e-4      # Paper value
  weight_decay: 0.05         # Paper value (not 0.01!)
  label_smoothing: 0.1       # Paper value
  gradient_clip: 1.0         # Stabilizes training
```

## Architecture Details

### 23→20 Channel Mapping
The paper parity mode uses a learnable linear mapper to transform TUEV's 23 channels to EEGPT's expected 20:

```python
# Maps TUEV channels (with A1/A2/T1/T2) to EEGPT interface
channel_mapper = nn.Linear(23, 20)
```

### Class Distribution
TUEV is heavily imbalanced (~99.5% background class):
- Class 0 (spsw): ~0.04%
- Class 1 (gped): ~0.10%
- Class 2 (pled): ~0.08%
- Class 3 (eyem): ~0.07%
- Class 4 (artf): ~0.13%
- Class 5 (bckg): ~99.58%

## Monitoring Training

### View Training Progress
```bash
# Attach to tmux session
tmux attach -t tuev_paper_parity

# Detach without stopping: Ctrl+B, then D
```

### Check Logs
```bash
# Latest log file
tail -f logs/tuev_paper_parity_*.log

# Monitor metrics
grep "balanced_accuracy" logs/tuev_paper_parity_*.log
```

## Target Metrics

From the EEGPT paper (Table 13):
- **Balanced Accuracy**: 62.32%
- **Weighted F1**: 81.87% (misleading due to imbalance)
- **Cohen's Kappa**: 0.635

## Troubleshooting

### Common Issues

1. **"Cache not found" error**
   - Run `./scripts/build_tuev_cache.sh` first
   - Check `$BGB_CACHE_DIR` is set correctly

2. **"Argument --cache_dir not recognized"**
   - Use `--cache-dir` (with hyphen) not `--cache_dir`
   - Launch script has been fixed but check manual commands

3. **torch.load TypeError**
   - Older PyTorch versions (<2.4) don't support `weights_only`
   - Code auto-detects and handles this

4. **Training hangs at epoch boundary**
   - OneCycleLR scheduler issue - code now handles this
   - Uses internal step counting to prevent overrun

### Performance Tips

- Use GPU: Training is ~100x faster on GPU
- Batch size: 64 works well for 16GB GPUs
- Multi-GPU: Not yet supported
- Mixed precision: Can reduce memory by 50%

## Code Quality Checks

Before submitting PRs, run:

```bash
make format      # Auto-format code
make lint        # Check for issues
make typecheck   # Type validation
make test        # Run tests
```

## Advanced Usage

### Custom Configurations
Create your own config by copying and modifying:
```bash
cp configs/tuev_paper_parity.yaml configs/my_experiment.yaml
# Edit hyperparameters
python train_tuev_mne.py --config configs/my_experiment.yaml
```

### Evaluation Only
```bash
python train_tuev_mne.py \
    --config configs/tuev_paper_parity.yaml \
    --eval-only \
    --checkpoint output/run/best_model.pt
```

## Citations

If using this code, please cite:

```bibtex
@article{eegpt2025,
  title={Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI},
  author={Wei-Bang Jiang and Li-Ming Zhao and Bao-Liang Lu},
  journal={ICLR},
  year={2025}
}
```

## Support

- GitHub Issues: Report bugs or request features
- Documentation: See `/docs/TRAINING.md` for general training info
- Architecture: See `CLAUDE.md` for system design principles