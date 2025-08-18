#!/usr/bin/env python
"""Launch TUEV training with proper environment setup."""

import os
import subprocess
import sys
from pathlib import Path

# Set environment variables
os.environ['BGB_DATA_ROOT'] = '/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTHONUNBUFFERED'] = '1'

# Paths
PROJECT_ROOT = Path('/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr')
SCRIPT_DIR = PROJECT_ROOT / 'experiments' / 'eegpt_linear_probe'
CACHE_DIR = Path(os.environ['BGB_DATA_ROOT']) / 'cache' / 'tuev_table13'

print("==========================================")
print("🚀 TUEV 6-CLASS EVENT DETECTION TRAINING")
print("==========================================")
print("")
print("📊 ARCHITECTURE (Table 13):")
print("  - Input: 23 × 1000 (3.9 seconds @ 256Hz)")
print("  - Channel reduction: 23 → 20")
print("  - Temporal kernel: 55")
print("  - Dropout: 0.5")
print("  - Batch size: 500")
print("  - Learning rate: 5e-4 (constant)")
print("")
print("🎯 TARGET PERFORMANCE (Paper Table 3):")
print("  - Balanced Accuracy: 0.6232 ± 0.0114")
print("  - Weighted F1:       0.8187 ± 0.0063")
print("  - Cohen's Kappa:     0.6351 ± 0.0134")
print("")
print("==========================================")

# Change to script directory
os.chdir(SCRIPT_DIR)

# Step 1: Check if cache exists
train_cache = CACHE_DIR / 'tuev_train_cache'
if not train_cache.exists():
    print("⚠️  Cache not found. Building cache first...")
    print("")
    
    # Build cache
    cmd = [
        sys.executable, 'build_tuev_cache.py',
        '--config', 'configs/tuev_table13_aligned.yaml',
        '--output', str(CACHE_DIR)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ Cache building failed:")
        print(result.stderr)
        sys.exit(1)
    
    print("✅ Cache built successfully!")
    print("")
else:
    print(f"✅ Cache found at {CACHE_DIR}")
    print("")

# Step 2: Run training
print("==========================================")
print("📈 STARTING TRAINING")
print("==========================================")
print("")

# Create logs directory
logs_dir = SCRIPT_DIR / 'logs'
logs_dir.mkdir(exist_ok=True)

# Run training with first seed
SEED = 42
log_file = logs_dir / f'tuev_seed{SEED}_launch.log'

print(f"🎲 Starting training with seed {SEED}")
print(f"📝 Logging to: {log_file}")
print("")

cmd = [
    sys.executable, 'train_tuev_aligned.py',
    '--config', 'configs/tuev_table13_aligned.yaml',
    '--device', 'cuda',
    '--seed', str(SEED),
    '--use-cache'
]

# Run training
with open(log_file, 'w') as f:
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Stream output
    for line in iter(process.stdout.readline, ''):
        if line:
            print(line.rstrip())
            f.write(line)
            f.flush()
    
    process.wait()
    
if process.returncode == 0:
    print("")
    print("✅ Training completed successfully!")
else:
    print("")
    print("❌ Training failed. Check the log file for details.")
    sys.exit(1)