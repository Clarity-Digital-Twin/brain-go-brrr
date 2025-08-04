#!/bin/bash
# WORKING launch script that forces Lightning to skip dataloader counting

cd /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr

export BGB_DATA_ROOT=/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export EEGPT_CONFIG=configs/tuab_stable.yaml

# Force fast_dev_run for immediate results
echo "Launching with fast_dev_run to verify training works..."

cd experiments/eegpt_linear_probe
../../.venv/bin/python train_enhanced.py trainer.fast_dev_run=True trainer.max_epochs=1