import torch
import torch.nn as nn

# Simulate the issue
batch_size = 32
n_channels = 20
time_steps = 1024
patch_size = 64
embed_dim = 512

# Compute number of patches
num_patches = time_steps // patch_size  # 1024 / 64 = 16
print(f"Time patches: {num_patches}")

# After patch embedding, x should be (B, num_patches, n_channels, embed_dim)
x = torch.randn(batch_size, num_patches, n_channels, embed_dim)
print(f"x shape: {x.shape}")

# Channel IDs 
chan_ids = torch.arange(20)
print(f"chan_ids shape: {chan_ids.shape}")

# Channel embedding
chan_embed = nn.Embedding(62, embed_dim)
chan_embed_out = chan_embed(chan_ids).unsqueeze(0).unsqueeze(0)
print(f"chan_embed shape: {chan_embed_out.shape}")

# Try to add
try:
    result = x + chan_embed_out
    print(f"Success! Result shape: {result.shape}")
except RuntimeError as e:
    print(f"Error: {e}")
