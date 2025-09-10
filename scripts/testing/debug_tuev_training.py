import sys
from pathlib import Path

import torch

print("1. Testing basic imports...")
sys.stdout.flush()

print("2. CUDA status:", torch.cuda.is_available())
sys.stdout.flush()

print("3. Loading dataset...")
from brain_go_brrr.infra.data.tuev_event_dataset import TUEVEventDataset

dataset = TUEVEventDataset(Path("data/datasets/tuev"), split="train")
print(f"   Loaded {len(dataset)} samples")
sys.stdout.flush()

print("4. Testing data loading...")
x, y = dataset[0]
print(f"   Sample shape: {x.shape}, label: {y}")
sys.stdout.flush()

print("5. Creating model...")
from brain_go_brrr.infra.ml_models.eegpt_wrapper import EEGPTWrapper

use_channels_names = [
    'FP1',
    'FPZ',
    'FP2',
    'F7',
    'F3',
    'FZ',
    'F4',
    'F8',
    'T7',
    'C3',
    'CZ',
    'C4',
    'T8',
    'P7',
    'P3',
    'PZ',
    'P4',
    'P8',
    'O1',
    'O2',
]
model_kwargs = {"n_channels": use_channels_names, "time_steps": 1024}
print("   About to create EEGPTWrapper...")
sys.stdout.flush()

try:
    wrapper = EEGPTWrapper(
        checkpoint_path="data/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt",
        model_kwargs=model_kwargs,
    )
    print("   Model created successfully!")
except Exception as e:
    print(f"   ERROR creating model: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("6. All tests passed!")
