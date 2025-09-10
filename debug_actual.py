import torch
from pathlib import Path
import sys
sys.path.insert(0, '/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/src')

from brain_go_brrr.infra.ml_models.eegpt_architecture import create_eegpt_model

# TUEV config
use_channels_names = [
    'FP1', 'FPZ', 'FP2',
    'F7', 'F3', 'FZ', 'F4', 'F8',
    'T7', 'C3', 'CZ', 'C4', 'T8',
    'P7', 'P3', 'PZ', 'P4', 'P8',
    'O1', 'O2'
]  # 20 channels

# Create model with 20 channels
checkpoint_path = "data/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"
model_kwargs = {"n_channels": use_channels_names, "time_steps": 1024}

print(f"Creating EEGPT with {len(use_channels_names)} channels")
model = create_eegpt_model(checkpoint_path, **model_kwargs)

# Test input
batch_size = 2
x = torch.randn(batch_size, 20, 1024)
chan_ids = torch.arange(20)

print(f"Input shape: {x.shape}")
print(f"Chan IDs: {chan_ids}")

# Try forward pass
model.eval()
with torch.no_grad():
    try:
        out = model(x, chan_ids, return_all_temporal=False)
        print(f"Success! Output shape: {out.shape}")
    except Exception as e:
        print(f"Error: {e}")
        
        # Debug patch embedding
        x_debug = x.unsqueeze(1)  # Add channel dim
        patches = model.patch_embed.proj(x_debug)
        print(f"After conv2d: {patches.shape}")
        patches = patches.transpose(1, 3)
        print(f"After transpose: {patches.shape}")
