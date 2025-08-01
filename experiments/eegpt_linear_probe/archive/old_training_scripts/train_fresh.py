#!/usr/bin/env python
"""Fresh training script with no Python caching."""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Disable Python bytecode caching
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Set environment
os.environ["BGB_DATA_ROOT"] = str(project_root / "data")

# Create log directory
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Generate log filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"training_{timestamp}.log"

# Build command with correct configuration
cmd = [
    sys.executable,
    "-B",  # Don't write bytecode
    "train_tuab_probe.py",
    # Paper-faithful settings from EEGPT
    "training.epochs=10",
    "training.batch_size=64",
    "training.learning_rate=5e-4",
    "training.num_workers=0",  # MPS compatibility
    # Early stopping on val_loss
    "training.monitor=val_loss",
    "training.mode=min",
    "training.patience=5",
    # Use full dataset
    "experiment.limit_train_batches=1.0",
    "experiment.limit_val_batches=1.0",
    # Disable pin memory for MPS
    "data.pin_memory=False",
    # Set log directory
    f"trainer.default_root_dir=logs/run_{timestamp}",
]

print("=" * 60)
print("EEGPT Linear Probe Training - Fresh Run (No Cache)")
print("=" * 60)
print(f"Log file: {log_file}")
print(f"Checkpoint dir: logs/run_{timestamp}")
print("Dataset: 20 channels × 2048 samples (8s @ 256Hz)")
print("Settings: batch=64, lr=5e-4, epochs=10")
print("Channel mapping: T3→T7, T4→T8, T5→P7, T6→P8")
print("=" * 60)

# Run training with proper logging

with open(log_file, "w") as f:
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in iter(process.stdout.readline, ""):
        print(line, end="")  # Print to console
        f.write(line)  # Write to log file
        f.flush()
    process.wait()

print(f"\nTraining complete! Check {log_file} for full output.")
print(f"Checkpoints saved to: logs/run_{timestamp}")
