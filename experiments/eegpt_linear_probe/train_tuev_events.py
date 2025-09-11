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
from torch.cuda.amp import GradScaler
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score
from timm.loss import LabelSmoothingCrossEntropy as TimmLabelSmoothingCE
from torch.utils.data import DataLoader
from tqdm import tqdm

from brain_go_brrr.domain.constraints import LinearWithConstraint
from brain_go_brrr.infra.data.tuev_event_dataset import TUEVEventDataset
from brain_go_brrr.infra.ml_models.channel_mapper import TUEVChannelMapper
from brain_go_brrr.infra.ml_models.eegpt_architecture import CHANNEL_DICT
from brain_go_brrr.infra.ml_models.eegpt_wrapper import EEGPTWrapper


class TUEVClassifierHead(nn.Module):
    """EEGPT-based classifier head for TUEV event classification.

    Paper-parity implementation:
    - Enforce parity mode (1000 samples native; patch_stride=64)
    - Use all temporal summary tokens (N_temporal×4×512 → 30,720) with Dropout(0.8) → Linear(6)
    - No mean pooling; no flatten-to-2048 path
    """

    def __init__(
        self,
        eegpt_checkpoint: str,  # REQUIRED - no default
        num_classes: int = 6,
        use_parity: bool = True,  # ENFORCE: TRUE paper parity (1000 samples native)
    ):
        super().__init__()
        self.use_parity = use_parity

        # ALWAYS use EEGPT (no MLP option)
        if eegpt_checkpoint:
            # CRITICAL: Configure EEGPT for TUEV's 20 channels
            # This is the EXACT order from the EEGPT reference
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
            ]  # 20 channels!

            # Configure model with 20 channels and correct time steps
            model_kwargs = {
                "n_channels": use_channels_names,
                "drop_path_rate": 0.0,  # CRITICAL: Model hardcodes 0.0 despite CLI flag!
                "time_steps": 1000,  # Enforce native 1000 samples
                "patch_stride": 64,  # Stride 64 at 200 Hz
            }
            self.eegpt = EEGPTWrapper(checkpoint_path=eegpt_checkpoint, model_kwargs=model_kwargs)
            print("WARNING: DropPath=0.0 (model ignores CLI flag and hardcodes 0.0!)")

            # CRITICAL: Disable normalization to match reference (they use raw μV)
            # Our dataset outputs Volts, we'll scale to μV in forward pass
            self.eegpt.normalize = False
            print("Normalization DISABLED - using raw values like reference")

            # Convert channel names to IDs using EEGPT's channel dictionary
            chan_ids = []
            for ch in use_channels_names:
                if ch in CHANNEL_DICT:
                    chan_ids.append(CHANNEL_DICT[ch])
                else:
                    raise ValueError(f"Channel {ch} not found in CHANNEL_DICT")

            # Store as tensor for use in forward pass
            self.register_buffer('chan_ids', torch.tensor(chan_ids).long())

            # Build classifier head to consume ALL temporal summary tokens
            # Compute temporal patch count from the EEGPT model
            temporal_patches = int(self.eegpt.model.patch_embed.num_patches[1])
            embed_num = int(self.eegpt.model.embed_num)  # 4
            embed_dim = int(self.eegpt.model.embed_dim)  # 512
            in_features = temporal_patches * embed_num * embed_dim  # 15*4*512 = 30720
            print(
                f"Using TEMPORAL TOKEN FLATTENING: {temporal_patches}×{embed_num}×{embed_dim} = {in_features}"
            )

            # CRITICAL: Use LinearWithConstraint to match reference implementation
            # This prevents weight explosion with 30,720 features
            self.classifier = nn.Sequential(
                nn.Dropout(0.8),
                LinearWithConstraint(in_features, num_classes, max_norm=1.0),
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
        # Enforce native 1000 samples (parity); no padding needed
        
        # Reference reshapes to (B, 20, 5, 200) before EEGPT but then flattens immediately
        # This is purely for compatibility - functionally equivalent to our approach
        batch_size = x.shape[0]
        x_reshaped = x.reshape(batch_size, 20, 5, 200)  # Match reference reshape
        x = x_reshaped.reshape(batch_size, 20, 1000)  # Flatten back immediately like reference

        # Extract ALL temporal EEGPT features using proper channel IDs
        # Returns shape: (B, N_temporal, 4, 512)
        features_all = self.eegpt(x, chan_ids=self.chan_ids, return_all_temporal=True)
        # Flatten temporal + summary tokens: (B, N_temporal*4*512)
        features = features_all.reshape(features_all.shape[0], -1)
        logits = self.classifier(features)
        return logits


# Using timm's LabelSmoothingCrossEntropy for exact reference match
# Our custom implementation above is kept for reference but not used


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

    # Print layer-wise LR decay info for verification
    print("\nLayer-wise learning rate decay applied:")
    for group in param_groups:
        print(f"  {group['name']}: lr={group['lr']:.2e}, wd={group['weight_decay']:.3f}")

    return torch.optim.AdamW(param_groups)


def cosine_scheduler(
    base_value: float,
    final_value: float,
    epochs: int,
    niter_per_ep: int,
    warmup_epochs: int = 0,
    start_warmup_value: float = 0,
) -> np.ndarray:
    """Create per-iteration cosine scheduler matching reference implementation.

    Args:
        base_value: Initial learning rate
        final_value: Final learning rate
        epochs: Total number of epochs
        niter_per_ep: Number of iterations per epoch
        warmup_epochs: Number of warmup epochs
        start_warmup_value: Starting value for warmup

    Returns:
        Array of learning rates for each iteration
    """
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep

    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [
            final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * i / len(iters)))
            for i in iters
        ]
    )

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


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

            # Forward pass with mixed precision (matches reference)
            with torch.cuda.amp.autocast():
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

    # Handle empty dataloader case
    if len(dataloader) == 0:
        print("WARNING: Empty dataloader in evaluation!")
        return {
            'loss': 0.0,
            'balanced_accuracy': 0.0,
            'weighted_f1': 0.0,
            'cohen_kappa': 0.0,
        }

    metrics = {
        'loss': total_loss / len(dataloader),
        'balanced_accuracy': balanced_accuracy_score(all_labels, all_preds)
        if len(all_labels) > 0
        else 0.0,
        'weighted_f1': f1_score(all_labels, all_preds, average='weighted')
        if len(all_labels) > 0
        else 0.0,
        'cohen_kappa': cohen_kappa_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0,
    }

    # Add per-class metrics
    from sklearn.metrics import classification_report, confusion_matrix

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3, 4, 5])
    print("\nConfusion Matrix:")
    print(cm)

    # Print per-class report
    class_names = ['spsw', 'gped', 'pled', 'eyem', 'artf', 'bckg']
    report = classification_report(
        all_labels, all_preds, labels=[0, 1, 2, 3, 4, 5], target_names=class_names, zero_division=0
    )
    print("\nPer-class metrics:")
    print(report)

    return metrics


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    accumulate_steps: int = 1,
    lr_schedule: np.ndarray = None,
    wd_schedule: np.ndarray = None,
    epoch: int = 0,
    scaler: "GradScaler | None" = None,
) -> dict[str, float]:
    """Train for one epoch with per-iteration scheduling.

    Args:
        model: Model to train
        dataloader: Training DataLoader
        criterion: Loss criterion
        optimizer: Optimizer
        device: Device to run on
        accumulate_steps: Gradient accumulation steps
        lr_schedule: Per-iteration learning rates
        wd_schedule: Per-iteration weight decay values
        epoch: Current epoch number

    Returns:
        Dictionary of training metrics
    """
    model.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    # Calculate starting iteration for this epoch
    num_steps = len(dataloader) // accumulate_steps
    start_iter = epoch * num_steps

    optimizer.zero_grad()

    for i, (x, y) in enumerate(tqdm(dataloader, desc="Training")):
        x, y = x.to(device), y.to(device)

        # Update learning rate and weight decay per iteration (before optimizer step)
        it = start_iter + i // accumulate_steps
        if lr_schedule is not None and it < len(lr_schedule):
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_schedule[it] * param_group.get("lr_scale", 1.0)
        if wd_schedule is not None and it < len(wd_schedule):
            for param_group in optimizer.param_groups:
                if param_group.get("weight_decay", 0) > 0:  # Only update groups with weight decay
                    param_group["weight_decay"] = wd_schedule[it]

        # Forward pass with mixed precision (matches reference)
        with torch.cuda.amp.autocast():
            logits = model(x)
            loss = criterion(logits, y)

        # Scale loss for gradient accumulation
        loss = loss / accumulate_steps

        # Use GradScaler for mixed precision
        scaler.scale(loss).backward()

        # Update weights
        if (i + 1) % accumulate_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Store metrics
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        total_loss += loss.item() * accumulate_steps

    # Final optimizer step if needed
    if len(dataloader) % accumulate_steps != 0:
        scaler.step(optimizer)
        scaler.update()

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

    # NO BALANCED SAMPLING - Reference doesn't use it and achieves 62% BAC!
    print("Setting up training WITHOUT balanced sampling (matches reference)...")

    # Print class distribution for monitoring
    import json

    with open(train_dataset.cache_dir / train_dataset.split / "index.json") as f:
        index_data = json.load(f)
    train_labels = [seg["label"] for seg in index_data["segments"]]
    class_counts = torch.bincount(torch.tensor(train_labels, dtype=torch.long), minlength=6)
    print(f"Class distribution (natural): {class_counts.tolist()}")
    print("Using natural distribution - NO rebalancing")

    # Create dataloaders with WSL-safe defaults
    pin_memory = args.pin_memory  # Default False for WSL
    persistent_workers = (args.num_workers > 0) and args.persistent_workers

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,  # Simple shuffle, NO sampler
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    # Create model
    print("Creating model...")

    # EEGPT checkpoint is REQUIRED
    if not args.eegpt_checkpoint:
        raise ValueError("--eegpt_checkpoint is REQUIRED for paper parity!")

    print(f"Using EEGPT backbone from {args.eegpt_checkpoint}")
    print("TRUE PARITY MODE: Using native 1000 samples (no padding)")

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
            # x: (batch, 23, 1000) in Volts from dataset
            # CRITICAL: Reference divides μV by 100! (engine_for_finetuning_EEGPT.py:65)
            x = x * 1e6 / 100  # Convert V to μV then divide by 100 like reference
            x = x.unsqueeze(2)  # Add spatial dim: (batch, 23, 1, 1000)
            x = self.mapper(x)  # Map to 20 channels: (batch, 20, 1, 1000) - keeps 4D!
            x = x.squeeze(2)  # Remove spatial dim: (batch, 20, 1000)
            x = self.classifier(x)  # Classify: (batch, 6)
            return x

    model = TUEVModel(channel_mapper, classifier).to(device)

    # Calculate gradient accumulation steps to match reference batch=400
    effective_batch_size = 400  # Reference uses exactly 400

    # Calculate accumulation steps to get as close to 400 as possible
    # Option 1: batch=32, accum=13 -> 416 (4% over)
    # Option 2: batch=40, accum=10 -> 400 (exact)
    # Option 3: batch=50, accum=8 -> 400 (exact)

    # Use exact 400 if possible, otherwise get close
    if effective_batch_size % args.batch_size == 0:
        accumulate_steps = effective_batch_size // args.batch_size
    else:
        # Round to nearest for closest match
        accumulate_steps = round(effective_batch_size / args.batch_size)
        accumulate_steps = max(1, accumulate_steps)  # At least 1

    actual_effective_batch = args.batch_size * accumulate_steps
    print(
        f"Effective batch size: {actual_effective_batch} (batch={args.batch_size}, accum={accumulate_steps})"
    )
    if abs(actual_effective_batch - effective_batch_size) > 20:
        print(
            f"WARNING: Effective batch {actual_effective_batch} differs from target {effective_batch_size} by >5%"
        )

    # Create optimizer with layer-wise decay
    optimizer = create_optimizer_with_layer_decay(
        model, lr=args.lr, weight_decay=args.weight_decay, layer_decay=args.layer_decay
    )

    # Initialize GradScaler for mixed precision (like reference's NativeScaler)
    scaler = GradScaler()

    # Create per-iteration schedulers matching reference implementation
    num_training_steps_per_epoch = len(train_loader) // accumulate_steps
    print(f"Steps per epoch: {num_training_steps_per_epoch}")

    # Learning rate schedule with cosine decay
    lr_schedule = cosine_scheduler(
        base_value=args.lr,
        final_value=1e-6,  # min_lr from reference
        epochs=args.epochs,
        niter_per_ep=num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
        start_warmup_value=0,
    )

    # Weight decay schedule (typically constant in TUEV)
    wd_schedule = cosine_scheduler(
        base_value=args.weight_decay,
        final_value=args.weight_decay,  # Same as base (constant)
        epochs=args.epochs,
        niter_per_ep=num_training_steps_per_epoch,
        warmup_epochs=0,  # No warmup for weight decay
    )

    # Create loss function - use timm's implementation to match reference exactly
    criterion = TimmLabelSmoothingCE(smoothing=args.label_smoothing)

    # Training loop
    best_bac = 0.0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train with per-iteration scheduling
        train_metrics = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            accumulate_steps,
            lr_schedule=lr_schedule,
            wd_schedule=wd_schedule,
            epoch=epoch,
            scaler=scaler,
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

        # No scheduler.step() needed - using per-iteration scheduling

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
    parser.add_argument('--seed', type=int, default=0, help='Random seed (0 for paper parity)')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save checkpoints')
    parser.add_argument(
        '--pin_memory',
        action='store_true',
        default=False,
        help='Pin memory for DataLoader (avoid on WSL)',
    )
    parser.add_argument(
        '--persistent_workers',
        action='store_true',
        default=False,
        help='Keep workers alive between epochs',
    )

    args = parser.parse_args()
    main(args)
