# EEGPT Linear Probe Training - Documentation Index

## 🚀 Quick Start
- **[README.md](README.md)** - Project overview and current status
- **[RUN_TRAINING_8S.sh](RUN_TRAINING_8S.sh)** - Launch training immediately

## 📚 Setup & Configuration
- **[SETUP_COOKBOOK.md](SETUP_COOKBOOK.md)** - Complete setup guide with all fixes
- **[ISSUES_AND_FIXES.md](ISSUES_AND_FIXES.md)** - Problems encountered and solutions
- **[configs/](configs/)** - Training configuration files

## 💻 Code & Templates
- **[TRAINING_SCRIPT_TEMPLATE.py](TRAINING_SCRIPT_TEMPLATE.py)** - Professional template for new experiments
- **[train_paper_aligned.py](train_paper_aligned.py)** - Current active training script
- **[custom_collate_fixed.py](custom_collate_fixed.py)** - Handles variable channel counts

## 📊 Status & Progress
- **[TRAINING_STATUS.md](TRAINING_STATUS.md)** - Current training progress and metrics
- **[output/](output/)** - Training outputs and checkpoints

## 🏆 Best Practices
- **[PROFESSIONAL_PRACTICES.md](PROFESSIONAL_PRACTICES.md)** - What pro teams do and why

## 🗄️ Archive
- **[archive/](archive/)** - Old scripts and failed attempts (for reference)

---

## Common Tasks

### Check Training Progress
```bash
tmux attach -t eegpt_training
# or
tail -f output/training_8s_*/training.log | grep -E "AUROC|Epoch"
```

### Start New Experiment
1. Copy `TRAINING_SCRIPT_TEMPLATE.py`
2. Create new config in `configs/`
3. Follow setup in `SETUP_COOKBOOK.md`

### Debug Issues
1. Check `ISSUES_AND_FIXES.md` for known problems
2. Validate setup with smoke test
3. Use debugging checklist in `SETUP_COOKBOOK.md`

---

## Key Takeaways

1. **PyTorch Lightning doesn't work** with large cached datasets - use pure PyTorch
2. **Path resolution** needs manual handling for `${BGB_DATA_ROOT}`
3. **Channel counts vary** - always use custom collate function
4. **Documentation saves time** - keep it updated

---

Last Updated: August 4, 2025
Training Status: Active (8s windows, targeting >0.85 AUROC)