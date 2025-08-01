#!/bin/bash
cd /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/experiments/eegpt_linear_probe
source ../../.venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"
python train_eval_only.py 2>&1 | tee training_live.log