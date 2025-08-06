# EEGPT Linear Probe Implementation Analysis

## Key Findings from EEGPT Repository

### 1. Linear Probe Architecture (from `linear_probe_EEGPT_BCIC2A.py`)

The EEGPT linear probe implementation consists of:

```python
# 1. Channel convolution (adaptive spatial filter)
self.chan_conv = nn.Conv1d(
    in_channels=self.chans_num,  # 19 for BCIC2A
    out_channels=58,  # EEGPT expects 58 channels
    kernel_size=1,
    groups=1,
    bias=True
)

# 2. Frozen EEGPT backbone
target_encoder = EEGTransformer(
    img_size=[19, 1024],
    patch_size=32*2,
    embed_num=4,  # 4 summary tokens
    embed_dim=512,
    depth=8,
    num_heads=8,
    mlp_ratio=4.0,
    drop_rate=0.,
    attn_drop_rate=0.,
    drop_path_rate=0.1,
    init_values=0.,
    qkv_bias=True,
    mode="pre-train",
    use_mean_pooling=False
)

# 3. Load pretrained weights and freeze
checkpoint = torch.load(load_path)
if 'model' in checkpoint:
    checkpoint = checkpoint['model']
msg = target_encoder.load_state_dict(checkpoint, strict=False)
for param in target_encoder.parameters():
    param.requires_grad = False

# 4. Linear classification head
self.linear_probe1 = LinearWithConstraint(embed_dim*4, embed_dim*4, max_norm=0.25)
self.linear_probe2 = LinearWithConstraint(embed_dim*4, 4, max_norm=0.25)  # 4 classes
```

### 2. Forward Pass

```python
def forward(self, x):
    # Input shape: [batch, channel, time]
    # Normalize input
    mean = x.mean(dim=0, keepdim=True)
    var = x.var(dim=0, keepdim=True, unbiased=False)
    x = (x - mean) / var.sqrt()

    # Adaptive channel mapping
    x = self.chan_conv(x)  # [batch, 58, 1024]

    # Extract features with frozen EEGPT
    with torch.no_grad():
        x_encoded = self.target_encoder(x.unsqueeze(1))  # [batch, 4*embed_dim]

    # Linear classification
    cls_token = x_encoded
    x = self.linear_probe1(cls_token)
    x = torch.nn.GELU()(x)
    x = self.linear_probe2(x)

    return x  # logits
```

### 3. Training Configuration

- **Optimizer**: AdamW with weight_decay=0.01
- **Learning Rate**: OneCycleLR with max_lr=5e-4
- **Loss**: CrossEntropyLoss
- **Batch Size**: 64
- **Training**: Only channel conv + linear probe parameters

### 4. Key Implementation Details

1. **Channel Alignment**: Uses 1x1 convolution to map from dataset channels to EEGPT's 58 channels
2. **Normalization**: Input is z-score normalized before processing
3. **Feature Extraction**: Uses 4 summary tokens from EEGPT (4 * 512 = 2048 dim features)
4. **Constraint**: Uses LinearWithConstraint with max_norm=0.25 for regularization
5. **Two-stage Linear**: First linear layer keeps dimension, second reduces to classes

## Updated Implementation Plan for Brain-Go-Brrr

### Core Linear Probe Module

```python
# src/brain_go_brrr/models/eegpt_linear_probe.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from pathlib import Path

from brain_go_brrr.models.eegpt_wrapper import NormalizedEEGPT
from brain_go_brrr.modules.constraints import LinearWithConstraint


class EEGPTLinearProbe(nn.Module):
    """EEGPT Linear Probe following the paper implementation.

    Architecture:
    1. Channel adaptation layer (1x1 conv)
    2. Frozen EEGPT encoder
    3. Two-layer linear classifier with GELU activation
    """

    def __init__(
        self,
        pretrained_path: Path,
        n_input_channels: int,
        n_classes: int,
        embed_dim: int = 512,
        n_summary_tokens: int = 4,
        max_norm: float = 0.25,
        freeze_backbone: bool = True
    ):
        super().__init__()

        # Load pretrained EEGPT
        self.backbone = NormalizedEEGPT.load_from_checkpoint(
            pretrained_path,
            map_location="cpu"
        )

        if freeze_backbone:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Channel adaptation layer
        self.channel_adapter = nn.Conv1d(
            in_channels=n_input_channels,
            out_channels=58,  # EEGPT expects 58 channels
            kernel_size=1,
            bias=True
        )

        # Classification head
        feature_dim = embed_dim * n_summary_tokens
        self.classifier = nn.Sequential(
            LinearWithConstraint(feature_dim, feature_dim, max_norm=max_norm),
            nn.GELU(),
            LinearWithConstraint(feature_dim, n_classes, max_norm=max_norm)
        )

        self.n_input_channels = n_input_channels
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input EEG data [batch, channels, time]

        Returns:
            Logits [batch, n_classes]
        """
        # Input normalization
        mean = x.mean(dim=0, keepdim=True)
        var = x.var(dim=0, keepdim=True, unbiased=False)
        x = (x - mean) / (var.sqrt() + 1e-6)

        # Adapt channels
        x = self.channel_adapter(x)  # [batch, 58, time]

        # Extract features with frozen backbone
        with torch.no_grad() if not self.backbone.training else torch.enable_grad():
            # EEGPT expects [batch, 1, channels, time]
            x = x.unsqueeze(1)
            features = self.backbone.extract_features(x)  # [batch, 4*embed_dim]

        # Classify
        logits = self.classifier(features)

        return logits
```

### Task-Specific Implementations

#### 1. Abnormality Detection (TUAB)

```python
# src/brain_go_brrr/tasks/abnormality_detection.py

class AbnormalityDetectionProbe(EEGPTLinearProbe):
    """TUAB abnormality detection probe.

    Binary classification: normal (0) vs abnormal (1)
    Input: 23 channels, 30s windows at 256Hz
    """

    def __init__(self, pretrained_path: Path):
        super().__init__(
            pretrained_path=pretrained_path,
            n_input_channels=23,  # TUAB uses 23 channels
            n_classes=2,  # normal/abnormal
            freeze_backbone=True
        )

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get abnormality probability."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)[:, 1]  # Abnormal probability
```

#### 2. Sleep Staging (Sleep-EDF)

```python
# src/brain_go_brrr/tasks/sleep_staging.py

class SleepStagingProbe(EEGPTLinearProbe):
    """Sleep-EDF sleep staging probe.

    5-class classification: W, N1, N2, N3, REM
    Input: 2 channels (Fpz-Cz, Pz-Oz), 30s windows at 256Hz
    """

    def __init__(self, pretrained_path: Path):
        super().__init__(
            pretrained_path=pretrained_path,
            n_input_channels=2,  # Sleep-EDF uses 2 channels
            n_classes=5,  # W, N1, N2, N3, REM
            freeze_backbone=True
        )

        # Additional context encoder for 30s windows (per paper)
        self.context_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=512*4,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1
            ),
            num_layers=4
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process 30s window as multiple 4s segments."""
        batch_size, channels, time = x.shape

        # Split into 4s windows (assuming 256Hz)
        window_size = 1024  # 4s at 256Hz
        n_windows = time // window_size

        # Process each window
        features = []
        for i in range(n_windows):
            window = x[:, :, i*window_size:(i+1)*window_size]
            feat = super().forward(window)
            features.append(feat)

        # Stack and encode context
        features = torch.stack(features, dim=1)  # [batch, n_windows, feat_dim]
        features = self.context_encoder(features)

        # Average pool and classify
        features = features.mean(dim=1)

        return features
```

### Training Script Template

```python
# experiments/linear_probe_tuab.py

import pytorch_lightning as pl
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from brain_go_brrr.tasks.abnormality_detection import AbnormalityDetectionProbe
from brain_go_brrr.data.tuab_dataset import TUABDataset


class LinearProbeTrainer(pl.LightningModule):
    def __init__(
        self,
        model: EEGPTLinearProbe,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.01
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_hyperparameters(ignore=['model'])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Metrics
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()

        self.log('train_loss', loss)
        self.log('train_acc', acc)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Store for epoch-end metrics
        probs = F.softmax(logits, dim=-1)[:, 1]  # Abnormal probability

        return {
            'loss': loss,
            'probs': probs,
            'labels': y
        }

    def validation_epoch_end(self, outputs):
        # Calculate AUROC
        all_probs = torch.cat([x['probs'] for x in outputs])
        all_labels = torch.cat([x['labels'] for x in outputs])

        from sklearn.metrics import roc_auc_score
        auroc = roc_auc_score(
            all_labels.cpu().numpy(),
            all_probs.cpu().numpy()
        )

        self.log('val_auroc', auroc)

    def configure_optimizers(self):
        # Only optimize non-frozen parameters
        params = [p for p in self.model.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.2
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }


def main():
    # Model
    model = AbnormalityDetectionProbe(
        pretrained_path=Path("data/models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt")
    )

    # Data
    train_dataset = TUABDataset(
        root_dir=Path("data/datasets/external/tuh_eeg_abnormal/v3.0.0"),
        split="train"
    )
    val_dataset = TUABDataset(
        root_dir=Path("data/datasets/external/tuh_eeg_abnormal/v3.0.0"),
        split="val"
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision=16,
        logger=pl.loggers.TensorBoardLogger("logs/", name="tuab_probe"),
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="val_auroc",
                mode="max",
                save_top_k=1
            ),
            pl.callbacks.EarlyStopping(
                monitor="val_auroc",
                patience=5,
                mode="max"
            )
        ]
    )

    # Train
    lightning_model = LinearProbeTrainer(model)
    trainer.fit(lightning_model, train_loader, val_loader)

    # Test
    test_dataset = TUABDataset(
        root_dir=Path("data/datasets/external/tuh_eeg_abnormal/v3.0.0"),
        split="test"
    )
    test_loader = DataLoader(test_dataset, batch_size=64)

    results = trainer.test(lightning_model, test_loader)
    print(f"Test AUROC: {results[0]['test_auroc']:.4f}")


if __name__ == "__main__":
    main()
```

## Next Steps

1. **Implement LinearWithConstraint module** (missing from our codebase)
2. **Create TUAB dataset loader** following EEGPT's preprocessing
3. **Verify channel mapping** for different datasets
4. **Test with small subset** before full training

## Key Differences from Initial Plan

1. Uses **two-stage linear classifier** with GELU activation (not single layer)
2. Applies **max_norm constraint** on linear layers (0.25)
3. **Input normalization** happens in forward pass (not in preprocessing)
4. **Channel adapter** uses Conv1d with bias (not just weights)
5. **OneCycleLR** with pct_start=0.2 (not constant LR)
