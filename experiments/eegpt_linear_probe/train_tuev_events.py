#!/usr/bin/env python
"""Train TUEV event classifier for EEGPT paper parity.

REQUIRES EEGPT checkpoint - no MLP fallback!

Two modes:
- --use_parity: TRUE paper (1000 samples native, requires modified EEGPT)
- Default: Pad to 1024 (compatible with standard EEGPT checkpoints)

Expected: 62.32% ± 1.14% balanced accuracy
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from brain_go_brrr.infra.data.tuev_event_dataset import TUEVEventDataset
from brain_go_brrr.infra.ml_models.channel_mapper import TUEVChannelMapper
from brain_go_brrr.infra.ml_models.eegpt_architecture import CHANNEL_DICT
from brain_go_brrr.infra.ml_models.eegpt_wrapper import EEGPTWrapper


class TUEVClassifierHead(nn.Module):
    """EEGPT-based classifier head for TUEV event classification.

    ALWAYS uses EEGPT - no MLP fallback!
    """

    def __init__(
        self,
        eegpt_checkpoint: str,  # REQUIRED - no default
        num_classes: int = 6,
        use_parity: bool = False,  # TRUE paper parity (1000 samples native)
    ):
        super().__init__()
        self.use_parity = use_parity

        # ALWAYS use EEGPT (no MLP option)
        if eegpt_checkpoint:
            # CRITICAL: Configure EEGPT for TUEV's 20 channels
            # This is the EXACT order from the EEGPT reference
            use_channels_names = [
                'FP1', 'FPZ', 'FP2',
                'F7', 'F3', 'FZ', 'F4', 'F8',
                'T7', 'C3', 'CZ', 'C4', 'T8',
                'P7', 'P3', 'PZ', 'P4', 'P8',
                'O1', 'O2'
            ]  # 20 channels!
            
            # Configure model with 20 channels and correct time steps
            model_kwargs = {"n_channels": use_channels_names}
            if use_parity:
                # Configure EEGPT for native 1000 time steps with stride 64
                model_kwargs.update({"time_steps": 1000, "patch_stride": 64})
            self.eegpt = EEGPTWrapper(checkpoint_path=eegpt_checkpoint, model_kwargs=model_kwargs)

            # Convert channel names to IDs using EEGPT's channel dictionary
            chan_ids = []
            for ch in use_channels_names:
                if ch in CHANNEL_DICT:
                    chan_ids.append(CHANNEL_DICT[ch])
                else:
                    raise ValueError(f"Channel {ch} not found in CHANNEL_DICT")

            # Store as tensor for use in forward pass
            self.register_buffer('chan_ids', torch.tensor(chan_ids).long())

            # EEGPT outputs 4×512 features, we need to classify
            self.classifier = nn.Sequential(
                nn.Flatten(),  # (B, 4, 512) -> (B, 2048)
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes),
            )
        else:
            raise ValueError("EEGPT checkpoint is required for paper parity!")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, 20, 1000)

        Returns:
            Logits of shape (batch, 6)
        """
        # Handle padding based on parity mode
        if self.use_parity:
            # TRUE PARITY: Use 1000 samples natively (requires modified EEGPT)
            # No padding needed if EEGPT is configured for time_steps=1000
            pass
        else:
            # FALLBACK: Pad to 1024 for standard EEGPT
            if x.shape[-1] == 1000:
                x = F.pad(x, (0, 24), mode='constant', value=0)  # Pad to 1024

        # Prepare channel IDs for this batch
        batch_size = x.shape[0]
        chan_ids = self.chan_ids.unsqueeze(0).expand(batch_size, -1)

        # Extract EEGPT features WITH PROPER CHANNEL IDS
        features = self.eegpt.extract_features(x, chan_ids=chan_ids, summary=False)  # (B, 4, 512)

        # Classify
        logits = self.classifier(features)
        return logits


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute label smoothing cross entropy.

        Args:
            pred: Predictions of shape (batch, num_classes)
            target: Target labels of shape (batch,)

        Returns:
            Scalar loss
        """
        n_classes = pred.size(-1)
        log_probs = F.log_softmax(pred, dim=-1)

        # Create smoothed target distribution
        smooth_target = torch.zeros_like(log_probs)
        smooth_target.fill_(self.smoothing / (n_classes - 1))
        smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        # Compute loss
        loss = -(smooth_target * log_probs).sum(dim=-1).mean()
        return loss


def create_optimizer_with_layer_decay(
    model: nn.Module, lr: float = 5e-4, weight_decay: float = 0.05, layer_decay: float = 0.65
) -> torch.optim.AdamW:
    """Create AdamW optimizer with layer-wise learning rate decay.

    Applies higher LR to shallower layers and heads, decays deeper transformer blocks.
    """
    import re

    param_groups: list[dict] = []
    no_decay_tokens = ("bias", "norm", "LayerNorm", "layernorm", "bn", "BatchNorm")

    # Determine maximum transformer block depth from parameter names like '...blocks.12...'
    block_idx_pattern = re.compile(r"\.blocks\.(\d+)\.")
    max_block = -1
    for name, _ in model.named_parameters():
        m = block_idx_pattern.search(name)
        if m:
            idx = int(m.group(1))
            if idx > max_block:
                max_block = idx

    # Group params by depth and decay
    grouped: dict[tuple[int, bool], dict] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # Assign depth: transformer blocks use their index; classifier and mapper act as heads; others as shallow
        m = block_idx_pattern.search(name)
        if m:
            depth = int(m.group(1))
        elif name.startswith("classifier"):
            depth = max_block + 2  # highest LR
        elif name.startswith("mapper"):
            depth = max_block + 1
        else:
            depth = 0  # patch_embed, chan_embed, summary_token, norms

        # Compute decayed LR relative to deepest block (blocks get decayed; heads use base LR)
        decay_depth = min(depth, max_block if max_block >= 0 else 0)
        layer_lr = lr * (layer_decay ** (max_block - decay_depth)) if max_block >= 0 else lr

        apply_wd = not any(tok in name for tok in no_decay_tokens)
        key = (depth, apply_wd)
        if key not in grouped:
            grouped[key] = {
                "params": [],
                "lr": layer_lr,
                "weight_decay": weight_decay if apply_wd else 0.0,
                "name": f"depth_{depth}_{'decay' if apply_wd else 'no_decay'}",
            }
        grouped[key]["params"].append(p)

    param_groups = [g for g in grouped.values() if g["params"]]
    if not param_groups:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return torch.optim.AdamW(param_groups)


def warmup_scheduler(
    optimizer: torch.optim.Optimizer, warmup_epochs: int, total_epochs: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create warmup scheduler with cosine annealing.

    Args:
        optimizer: Optimizer to schedule
        warmup_epochs: Number of warmup epochs
        total_epochs: Total number of epochs

    Returns:
        LambdaLR scheduler with warmup + cosine decay
    """

    def lr_lambda(epoch: int) -> float:
        """Calculate learning rate multiplier.

        - Linear warmup from 0 to 1 over warmup_epochs
        - Cosine annealing from 1 to 0 over remaining epochs
        """
        if epoch < warmup_epochs:
            # Linear warmup
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing after warmup
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device
) -> dict[str, float]:
    """Evaluate model on dataset.

    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation
        criterion: Loss criterion
        device: Device to run on

    Returns:
        Dictionary of metrics
    """
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)

            # Forward pass
            logits = model(x)
            loss = criterion(logits, y)

            # Store predictions
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            total_loss += loss.item()

    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = {
        'loss': total_loss / len(dataloader),
        'balanced_accuracy': balanced_accuracy_score(all_labels, all_preds),
        'weighted_f1': f1_score(all_labels, all_preds, average='weighted'),
        'cohen_kappa': cohen_kappa_score(all_labels, all_preds),
    }

    return metrics


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    accumulate_steps: int = 1,
) -> dict[str, float]:
    """Train for one epoch.

    Args:
        model: Model to train
        dataloader: Training DataLoader
        criterion: Loss criterion
        optimizer: Optimizer
        device: Device to run on
        accumulate_steps: Gradient accumulation steps

    Returns:
        Dictionary of training metrics
    """
    model.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    optimizer.zero_grad()

    for i, (x, y) in enumerate(tqdm(dataloader, desc="Training")):
        x, y = x.to(device), y.to(device)

        # Forward pass
        logits = model(x)
        loss = criterion(logits, y)

        # Scale loss for gradient accumulation
        loss = loss / accumulate_steps
        loss.backward()

        # Update weights
        if (i + 1) % accumulate_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Store metrics
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        total_loss += loss.item() * accumulate_steps

    # Final optimizer step if needed
    if len(dataloader) % accumulate_steps != 0:
        optimizer.step()

    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = {
        'loss': total_loss / len(dataloader),
        'balanced_accuracy': balanced_accuracy_score(all_labels, all_preds),
    }

    return metrics


def main(args):
    """Main training loop."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create datasets
    print("Loading datasets...")
    train_dataset = TUEVEventDataset(
        root_dir=Path(args.data_dir),
        split='train',
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        force_rebuild=args.force_rebuild,
    )

    eval_dataset = TUEVEventDataset(
        root_dir=Path(args.data_dir),
        split='eval',
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        force_rebuild=False,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Create model
    print("Creating model...")

    # EEGPT checkpoint is REQUIRED
    if not args.eegpt_checkpoint:
        raise ValueError("--eegpt_checkpoint is REQUIRED for paper parity!")

    print(f"Using EEGPT backbone from {args.eegpt_checkpoint}")
    if args.use_parity:
        print("TRUE PARITY MODE: Using native 1000 samples (no padding)")
    else:
        print("FALLBACK MODE: Padding to 1024 samples")

    channel_mapper = TUEVChannelMapper(dropout=0.8)

    classifier = TUEVClassifierHead(
        eegpt_checkpoint=args.eegpt_checkpoint,
        num_classes=6,
        use_parity=args.use_parity,
    )

    # Combine into single model
    class TUEVModel(nn.Module):
        def __init__(self, mapper, classifier):
            super().__init__()
            self.mapper = mapper
            self.classifier = classifier

        def forward(self, x):
            # x: (batch, 23, 1000)
            x = x.unsqueeze(2)  # Add spatial dim: (batch, 23, 1, 1000)
            x = self.mapper(x)  # Map to 20 channels: (batch, 20, 1, 1000) - keeps 4D!
            x = x.squeeze(2)  # Remove spatial dim: (batch, 20, 1000)
            x = self.classifier(x)  # Classify: (batch, 6)
            return x

    model = TUEVModel(channel_mapper, classifier).to(device)

    # Create optimizer and scheduler
    optimizer = create_optimizer_with_layer_decay(
        model, lr=args.lr, weight_decay=args.weight_decay, layer_decay=args.layer_decay
    )

    scheduler = warmup_scheduler(
        optimizer, warmup_epochs=args.warmup_epochs, total_epochs=args.epochs
    )

    # Create loss function
    criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)

    # Calculate gradient accumulation steps
    effective_batch_size = 400  # Paper uses this
    accumulate_steps = max(1, effective_batch_size // args.batch_size)
    print(f"Gradient accumulation steps: {accumulate_steps}")

    # Training loop
    best_bac = 0.0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, accumulate_steps
        )
        print(
            f"Train - Loss: {train_metrics['loss']:.4f}, BAC: {train_metrics['balanced_accuracy']:.4f}"
        )

        # Evaluate
        eval_metrics = evaluate(model, eval_loader, criterion, device)
        print(
            f"Eval - Loss: {eval_metrics['loss']:.4f}, BAC: {eval_metrics['balanced_accuracy']:.4f}"
        )
        print(
            f"       F1: {eval_metrics['weighted_f1']:.4f}, Kappa: {eval_metrics['cohen_kappa']:.4f}"
        )

        # Update scheduler
        scheduler.step()

        # Save best model
        if eval_metrics['balanced_accuracy'] > best_bac:
            best_bac = eval_metrics['balanced_accuracy']

            if args.save_dir:
                save_path = Path(args.save_dir) / 'best_model.pt'
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'metrics': eval_metrics,
                    },
                    save_path,
                )
                print(f"Saved best model with BAC: {best_bac:.4f}")

        # Early stopping check
        if epoch >= 5 and best_bac < 0.30:
            print("WARNING: BAC still < 0.30 after 5 epochs. Check data pipeline!")

    print(f"\nTraining complete! Best BAC: {best_bac:.4f}")
    print("Target BAC: 0.6232 ± 0.0114")

    if best_bac >= 0.60:
        print("✓ Achieved paper parity!")
    else:
        print("✗ Did not achieve paper parity. Check implementation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TUEV event classifier")

    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True, help='Path to TUEV data directory')
    parser.add_argument('--cache_dir', type=str, default=None, help='Path to cache directory')
    parser.add_argument('--force_rebuild', action='store_true', help='Force rebuild cache')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs (default: 30)')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size (default: 64, paper uses 400 distributed)',
    )
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate (default: 5e-4)')
    parser.add_argument(
        '--weight_decay', type=float, default=0.05, help='Weight decay (default: 0.05)'
    )
    parser.add_argument(
        '--layer_decay', type=float, default=0.65, help='Layer decay (default: 0.65)'
    )
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup epochs (default: 5)')
    parser.add_argument(
        '--label_smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)'
    )

    # Model arguments
    parser.add_argument(
        '--eegpt_checkpoint',
        type=str,
        required=True,
        help='Path to EEGPT checkpoint (REQUIRED for paper parity)',
    )
    parser.add_argument(
        '--use_parity',
        action='store_true',
        help='Use TRUE paper parity with 1000 samples natively (requires modified EEGPT)',
    )

    # Other arguments
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save checkpoints')

    args = parser.parse_args()
    main(args)
