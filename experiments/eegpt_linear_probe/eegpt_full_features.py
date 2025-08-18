"""Modified EEGPT wrapper that returns full patch features, not just summary tokens."""

import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.brain_go_brrr.infra.ml_models.eegpt_architecture import create_eegpt_model


class EEGPTFullFeatures(nn.Module):
    """EEGPT that returns all patch features for downstream tasks."""
    
    def __init__(self, checkpoint_path: str):
        super().__init__()
        self.model = create_eegpt_model(checkpoint_path)
        self.model.eval()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract ALL features, not just summary tokens.
        
        Args:
            x: Input (batch, channels, time)
            
        Returns:
            All patch features flattened
        """
        # Get shapes
        batch_size, n_channels, time_steps = x.shape
        
        # Patch embedding
        x = self.model.patch_embed(x)
        _, num_patches, num_channels, embed_dim = x.shape
        
        # Add channel embedding
        chan_ids = torch.arange(0, num_channels, device=x.device, dtype=torch.long)
        chan_embed = self.model.chan_embed(chan_ids).unsqueeze(0).unsqueeze(0)
        x = x + chan_embed
        
        # Reshape for transformer
        x = x.reshape(batch_size, num_patches * num_channels, embed_dim)
        
        # Add summary tokens
        summary_tokens = self.model.summary_token.repeat(batch_size, 1, 1)
        x = torch.cat([x, summary_tokens], dim=1)
        
        # Apply transformer blocks
        for block in self.model.blocks:
            x = block(x)
        
        # Return ALL tokens (not just summary)
        # Remove summary tokens and return patch features
        patch_features = x[:, :-self.model.embed_num, :]  # All except last 4 tokens
        
        # Apply final norm
        patch_features = self.model.norm(patch_features)
        
        # Flatten: (batch, num_patches*num_channels, embed_dim) -> (batch, flat_features)
        return patch_features.reshape(batch_size, -1)