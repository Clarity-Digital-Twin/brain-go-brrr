# TRAINING MONITOR SCHEDULE

## 🚨 MONITORING COMMANDS

### Quick check (every 30 min):
```bash
bash /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/MONITOR_TRAINING.sh
```

### Live view:
```bash
tmux attach -t eegpt_training
# Press Ctrl+B then D to detach
```

### Just peek at output:
```bash
tmux capture-pane -t eegpt_training -p | tail -50
```

### Check GPU:
```bash
nvidia-smi
```

### Tail logs:
```bash
tail -f logs/eegpt_training_20250803_202800/training.log
```

## 📅 MONITORING SCHEDULE

- ✅ 20:33 - Initial check (DONE) - Training starting up
- ⏰ 21:03 - Check 1 - Should see first epoch progress
- ⏰ 21:33 - Check 2 - Should see multiple epochs
- ⏰ 22:03 - Check 3 - Check loss is decreasing
- ⏰ 22:33 - Check 4 - Check AUROC improving
- ⏰ 23:03 - Check 5 - Mid-training health check

## 🔍 WHAT TO LOOK FOR

### ✅ GOOD SIGNS:
- GPU utilization > 90%
- Loss decreasing
- AUROC increasing (target > 0.90)
- No NaN/inf values
- Cache loaded in < 1 second
- NO file scanning messages

### ❌ BAD SIGNS:
- GPU utilization < 50%
- Loss = nan or inf
- "Scanning" or "Found X .edf files"
- Errors or exceptions
- Training stopped/crashed
- Memory errors

## 🚨 CRITICAL METRICS

From the paper, we expect:
- **AUROC**: 0.90-0.93 (target)
- **Balanced Accuracy**: ~80%
- **Training time**: ~2-3 hours for 50 epochs
- **GPU memory**: ~10-15 GB used
- **Batch processing**: ~100-200 batches/sec

## 💀 IF SOMETHING GOES WRONG

1. Check tmux session:
   ```bash
   tmux ls
   ```

2. Check full error:
   ```bash
   tail -100 logs/eegpt_training_*/training.log | grep -B10 -A10 "ERROR\|Exception"
   ```

3. Kill and restart:
   ```bash
   tmux kill-session -t eegpt_training
   ./experiments/eegpt_linear_probe/RUN_TRAINING_NOW.sh
   ```