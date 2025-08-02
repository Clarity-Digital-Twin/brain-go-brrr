# Extraction Plan for Complete Dataset

## Pre-extraction Checklist
- [ ] Download complete tar.gz from Mac (~40-50GB)
- [ ] Verify file integrity: `md5sum brain-go-brrr-COMPLETE-DATA-*.tar.gz`
- [ ] Ensure 100GB+ free space available

## Clean Extraction Steps

```bash
# 1. Navigate to project root
cd /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr

# 2. Clean incomplete data (keep model weights)
rm -rf data/datasets/external/tuh_eeg_abnormal/v3.0.1/edf/eval/*
rm -rf data/datasets/external/tuh_eeg_abnormal/v3.0.1/edf/train/*

# 3. Extract in tmux session
tmux new -s extract -d \
  "tar -xzf brain-go-brrr-COMPLETE-DATA-*.tar.gz \
   --checkpoint=1000 \
   --checkpoint-action=echo='Extracted %u files' \
   2>&1 | tee extraction_$(date +%Y%m%d_%H%M%S).log"

# 4. Monitor extraction
tmux attach -t extract  # Ctrl+B, D to detach

# 5. Verify extraction
find data/datasets/external/tuh_eeg_abnormal -name "*.edf" | wc -l
# Should show ~3000 files (2717 train + 276 eval + test)
```

## Post-extraction Verification

```bash
# Check file counts
echo "Train:" && find data/datasets/external/tuh_eeg_abnormal/v3.0.1/edf/train -name "*.edf" | wc -l
echo "Eval:" && find data/datasets/external/tuh_eeg_abnormal/v3.0.1/edf/eval -name "*.edf" | wc -l

# Verify model weights
ls -lh data/models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt

# Clean Mac artifacts if needed
find data -name "._*" -type f -delete
```

## Start Training

```bash
cd experiments/eegpt_linear_probe
tmux new -s gpu_train -d \
  "source ../../.venv/bin/activate && \
   python train_gpu_optimized.py 2>&1 | tee ../../training_gpu.log"

# Monitor
tmux attach -t gpu_train
nvidia-smi -l 1  # In another terminal
```