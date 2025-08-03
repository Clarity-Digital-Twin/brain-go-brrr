# EEGPT Implementation Guide

## Overview

EEGPT is a 10-million-parameter pretrained transformer model designed for universal EEG feature extraction. This guide provides comprehensive implementation details extracted from the original paper.

## Core Architecture

### Model Specifications
- **Parameters**: 10 million
- **Architecture**: Vision Transformer adapted for EEG
- **Input**: EEG signals with M channels and T time points
- **Patch Size**: 64 samples (250ms at 256Hz)
- **Masking Strategy**: 50% time patches, 80% channel patches
- **Training Method**: Dual self-supervised learning

### Key Components

#### 1. Local Spatio-Temporal Embedding
```python
# Patch extraction
patch_size = 64  # samples (250ms at 256Hz)
patch_stride = 32  # 50% overlap

# Channel-wise patching
for channel in range(n_channels):
    patches = extract_patches(eeg_signal[channel], patch_size, patch_stride)
    tokens = embed_patches(patches)  # Project to d_model dimensions
```

#### 2. Dual Self-Supervised Learning
- **Spatio-temporal representation alignment**: Aligns masked predictions with momentum encoder output
- **Mask-based reconstruction**: Recovers original signal from masked patches

```python
# Loss function
loss = MSE(predicted_features, momentum_features) + MSE(reconstructed_signal, original_signal)
```

#### 3. Hierarchical Processing
- **Spatial Encoder**: Processes channels at each time point
- **Temporal Predictor**: Predicts complete temporal features
- **Momentum Encoder**: Provides stable target representations (τ = 0.01)

## Training Configuration

### Pretraining Dataset
- **PhysioMI**: Motor imagery data
- **HGD**: High gamma dataset
- **M3CV**: Multi-modal dataset
- **Total**: Mixed multi-task EEG dataset

### Hyperparameters
```yaml
learning_rate: 5e-4
batch_size: 256
epochs: 100
optimizer: AdamW
weight_decay: 0.05
warmup_epochs: 10
momentum_tau: 0.01
mask_ratio_time: 0.5
mask_ratio_channel: 0.8
```

### Data Preprocessing
1. **Resampling**: Standardize to 256Hz
2. **Filtering**: 0.5-50Hz bandpass
3. **Normalization**: Z-score per channel
4. **Windowing**: 4-second segments (1024 samples)
5. **Channel Mapping**: Handle different montages

## Implementation Details

### 1. Patch Embedding
```python
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=64, d_model=768):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size, d_model)
        
    def forward(self, x):
        # x: [B, C, T]
        B, C, T = x.shape
        n_patches = T // self.patch_size
        
        # Extract patches
        patches = x.unfold(2, self.patch_size, self.patch_size)
        # patches: [B, C, n_patches, patch_size]
        
        # Project to d_model
        tokens = self.projection(patches)
        # tokens: [B, C, n_patches, d_model]
        
        return tokens
```

### 2. Masking Strategy
```python
def create_mask(n_time_patches, n_channels, mask_ratio_time=0.5, mask_ratio_channel=0.8):
    # Time masking: random 50% of time patches
    time_mask = torch.rand(n_time_patches) < mask_ratio_time
    
    # Channel masking: random 80% of channels per time patch
    channel_mask = torch.rand(n_channels, n_time_patches) < mask_ratio_channel
    
    # Combined mask
    mask = time_mask.unsqueeze(0) | channel_mask
    return mask
```

### 3. Encoder Architecture
```python
class SpatialEncoder(nn.Module):
    def __init__(self, d_model=768, n_heads=12, n_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads)
            for _ in range(n_layers)
        ])
        self.summary_token = nn.Parameter(torch.randn(1, 1, d_model))
        
    def forward(self, tokens):
        # tokens: [B, C, n_patches, d_model]
        B, C, P, D = tokens.shape
        
        # Process each time patch across channels
        outputs = []
        for p in range(P):
            x = tokens[:, :, p, :]  # [B, C, D]
            
            # Add summary token
            x = torch.cat([self.summary_token.expand(B, -1, -1), x], dim=1)
            
            # Apply transformer layers
            for layer in self.layers:
                x = layer(x)
            
            outputs.append(x[:, 0, :])  # Extract summary token
            
        return torch.stack(outputs, dim=1)  # [B, P, D]
```

## Downstream Task Adaptation

### Linear Probing Protocol
```python
class EEGPTLinearProbe(nn.Module):
    def __init__(self, eegpt_checkpoint, n_classes):
        super().__init__()
        # Load pretrained EEGPT
        self.eegpt = load_eegpt(eegpt_checkpoint)
        self.eegpt.eval()  # Freeze backbone
        
        # Linear probe
        self.classifier = nn.Linear(768, n_classes)
        
    def forward(self, x):
        with torch.no_grad():
            features = self.eegpt.extract_features(x)
        
        # Global average pooling
        features = features.mean(dim=1)  # [B, D]
        
        return self.classifier(features)
```

### Task-Specific Configurations

#### Motor Imagery (PhysioNet)
- Window: 4 seconds
- Overlap: 50%
- Classes: 4 (left/right hand, feet, tongue)
- Accuracy: 87.5% (SOTA)

#### Sleep Staging (Sleep-EDF)
- Window: 30 seconds
- Overlap: 0%
- Classes: 5 (W, N1, N2, N3, REM)
- Accuracy: 85.0%

#### Abnormal Detection (TUAB)
- Window: 10 seconds
- Overlap: 50%
- Classes: 2 (normal/abnormal)
- AUROC: 0.93

## Performance Benchmarks

| Task | Dataset | Metric | EEGPT | Previous SOTA |
|------|---------|--------|-------|---------------|
| Motor Imagery | PhysioNet | Accuracy | 87.5% | 74.5% |
| Sleep Staging | Sleep-EDF | Accuracy | 85.0% | 82.8% |
| Abnormal Detection | TUAB | AUROC | 0.93 | 0.89 |
| Event Detection | CHB-MIT | F1 | 0.76 | 0.72 |

## Key Implementation Notes

1. **Channel Compatibility**: Use channel interpolation for different montages
2. **Sampling Rate**: Always resample to 256Hz for consistency
3. **Window Size**: Flexible, but maintain patch size of 64 samples
4. **GPU Memory**: ~8GB required for batch size 32
5. **Training Time**: ~48 hours on 4x V100 GPUs for full pretraining

## Common Pitfalls

1. **Incorrect Masking**: Ensure time and channel masks are applied correctly
2. **Position Encoding**: Use rotary position embeddings for better generalization
3. **Normalization**: Apply per-channel z-score normalization
4. **Momentum Update**: Update momentum encoder every iteration, not epoch
5. **Feature Extraction**: Use summary tokens, not raw patch embeddings

## Integration with Brain-Go-Brrr

```python
from brain_go_brrr.models.eegpt_wrapper import create_normalized_eegpt
from brain_go_brrr.models.eegpt_two_layer_probe import EEGPTTwoLayerProbe

# Load pretrained model
eegpt = create_normalized_eegpt(
    checkpoint_path="data/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"
)

# Create task-specific probe
probe = EEGPTTwoLayerProbe(
    backbone_dim=768,
    n_input_channels=20,
    n_adapted_channels=19,
    hidden_dim=16,
    n_classes=2,
    dropout=0.5
)

# Combine for downstream task
model = EEGPTWithProbe(eegpt, probe)
```

## References

- Original Paper: "EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals"
- Code Repository: https://github.com/BINE022/EEGPT
- Model Checkpoint: Available in data/models/pretrained/