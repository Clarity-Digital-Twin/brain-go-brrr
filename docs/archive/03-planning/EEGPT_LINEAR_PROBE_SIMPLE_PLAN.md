# EEGPT Linear Probe - Simple Implementation Plan

## What We're Building

A linear probe for EEGPT that:
1. Takes EEG data (e.g., 23 channels for TUAB)
2. Adapts it to EEGPT's 58 channels
3. Extracts features using frozen EEGPT
4. Classifies using a simple 2-layer head

## Step 1: Add Missing Module (LinearWithConstraint)

```python
# src/brain_go_brrr/modules/constraints.py

import torch
import torch.nn as nn

class LinearWithConstraint(nn.Linear):
    """Linear layer with weight norm constraint.

    From EEGPT/downstream/Modules/Network/utils.py
    Applies max norm constraint to weights during forward pass.
    """
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)
```

## Step 2: Create Linear Probe Module

```python
# src/brain_go_brrr/models/eegpt_linear_probe.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from brain_go_brrr.models.eegpt_wrapper import NormalizedEEGPT
from brain_go_brrr.modules.constraints import LinearWithConstraint


class EEGPTLinearProbe(nn.Module):
    """Simple linear probe for EEGPT following paper implementation."""

    def __init__(
        self,
        checkpoint_path: Path,
        n_input_channels: int,
        n_classes: int,
        freeze_backbone: bool = True
    ):
        super().__init__()

        # Load pretrained EEGPT
        self.backbone = NormalizedEEGPT.load_from_checkpoint(
            checkpoint_path,
            map_location="cpu"
        )

        # Freeze backbone
        if freeze_backbone:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Channel adapter (maps input channels to 58)
        self.channel_adapter = nn.Conv1d(
            in_channels=n_input_channels,
            out_channels=58,
            kernel_size=1,
            bias=True
        )

        # Classification head (2048 = 512 embed_dim * 4 tokens)
        self.classifier = nn.Sequential(
            LinearWithConstraint(2048, 2048, max_norm=0.25),
            nn.GELU(),
            LinearWithConstraint(2048, n_classes, max_norm=0.25)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [batch, channels, time] EEG data

        Returns:
            logits: [batch, n_classes]
        """
        # Normalize input
        mean = x.mean(dim=0, keepdim=True)
        var = x.var(dim=0, keepdim=True, unbiased=False)
        x = (x - mean) / (var.sqrt() + 1e-6)

        # Adapt channels
        x = self.channel_adapter(x)  # [batch, 58, time]

        # Extract features (frozen)
        with torch.no_grad():
            x = x.unsqueeze(1)  # [batch, 1, 58, time]
            features = self.backbone.extract_features(x)  # [batch, 2048]

        # Classify
        logits = self.classifier(features)

        return logits
```

## Step 3: Create Simple Training Script

```python
# experiments/train_tuab_probe.py

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

from brain_go_brrr.models.eegpt_linear_probe import EEGPTLinearProbe
from brain_go_brrr.data.tuab_dataset import TUABDataset  # Need to implement


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, device):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = F.softmax(output, dim=1)[:, 1]  # Abnormal probability

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    auroc = roc_auc_score(all_labels, all_probs)
    return auroc


def main():
    # Config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 64
    epochs = 10
    lr = 5e-4

    # Model
    model = EEGPTLinearProbe(
        checkpoint_path=Path("data/models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"),
        n_input_channels=23,  # TUAB
        n_classes=2  # normal/abnormal
    ).to(device)

    # Data (placeholder - need to implement TUABDataset)
    # train_dataset = TUABDataset(split='train')
    # val_dataset = TUABDataset(split='val')
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Optimizer (only non-frozen params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

    # Training loop
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_auroc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val AUROC: {val_auroc:.4f}")

        # Save best model
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save(model.state_dict(), 'best_tuab_probe.pth')


if __name__ == "__main__":
    main()
```

## Summary

We've analyzed the reference repos and created a simple, focused plan:

1. **Only 4 reference repos needed**:
   - EEGPT (for implementation pattern) ✅
   - mne-python (for data loading) ✅
   - autoreject (for QC) ✅
   - yasa (for sleep metrics) ✅

2. **Skip the rest** to avoid over-complexity:
   - braindecode, mne-bids, pyEDFlib, tsfresh

3. **Simple implementation path**:
   - Add LinearWithConstraint module
   - Create EEGPTLinearProbe class
   - Build simple training loop
   - Focus on TUAB abnormality detection first

4. **Next immediate step**:
   - Implement the LinearWithConstraint module
   - Create TUAB dataset loader
   - Test with a small subset

This keeps things clean and focused on getting a working EEGPT linear probe without unnecessary complexity.
