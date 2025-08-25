#!/usr/bin/env python3
"""
Debug script to investigate loss=0 issue in TUAB training.
Runs for 200 batches with comprehensive logging.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import logging
import os
import sys

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.eegpt_linear_probe.datasets.tuab_dataset import TUABMemoryMappedDataset
from src.brain_go_brrr.models.eegpt_wrapper import EEGPTWrapper
from torch.utils.data import DataLoader
from experiments.eegpt_linear_probe.utils.custom_collate_fixed import collate_eeg_batch_fixed
from omegaconf import OmegaConf

# Copy LinearProbe class from train_tuab.py
class LinearProbe(nn.Module):
    """Two-layer linear probe with channel adapter."""

    def __init__(self, config):
        super().__init__()

        # Channel adapter (1x1 conv)
        if config.get("use_channel_adapter", False):
            self.channel_adapter = nn.Conv1d(
                config["channel_adapter_in"],
                config["channel_adapter_out"],
                kernel_size=1,
            )
        else:
            self.channel_adapter = None

        # Two-layer probe using LazyLinear to infer input dimension
        self.probe = nn.Sequential(
            nn.LazyLinear(config["hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["hidden_dim"], config["n_classes"]),
        )

    def forward(self, features):
        """Forward pass through probe."""
        batch_size = features.shape[0]
        
        # Handle both (B, 4, 512) and (B, 512) shapes
        if features.ndim == 3:
            # (B, 4, 512) -> flatten to (B, 2048)
            x = features.reshape(batch_size, -1)
        else:
            # (B, 512) - this is wrong, but handle gracefully
            x = features
        
        return self.probe(x)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def investigate_loss():
    """Run 200 batches with detailed debugging."""
    
    # Load config
    config_path = Path("configs/tuab.yaml")
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create dataset and loader
    data_root = os.environ.get("BGB_DATA_ROOT", 
                              "/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data")
    
    dataset = TUABMemoryMappedDataset(
        root_dir=f"{data_root}/datasets/external/tuab",
        cache_dir=f"{data_root}/cache/tuab_4s_final",
        split="train",
        n_channels=20,
        sampling_rate=256,
        window_duration=4.0,
        window_stride=2.0
    )
    
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,  # Deterministic for debugging
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_eeg_batch_fixed
    )
    
    # Load models
    checkpoint = f"{data_root}/models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"
    backbone = EEGPTWrapper(checkpoint_path=checkpoint)
    backbone.to(device)
    backbone.eval()
    
    probe_config = {
        "n_classes": 2,
        "hidden_dim": 128,
        "dropout": 0.1,
        "use_channel_adapter": False
    }
    probe = LinearProbe(probe_config)
    probe.to(device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(probe.parameters(), lr=0.001, weight_decay=0.0001)
    
    # Debug tracking
    all_losses = []
    all_labels = []
    all_logits_stats = []
    all_grads = []
    
    logger.info("="*80)
    logger.info("STARTING LOSS INVESTIGATION - 200 BATCHES")
    logger.info("="*80)
    
    for batch_idx, (data, labels) in enumerate(loader):
        if batch_idx >= 200:
            break
            
        data = data.to(device)
        labels = labels.to(device)
        batch_size = data.size(0)
        
        # Extract features
        with torch.no_grad():
            features = backbone.extract_features(data, summary=False)
            
            # DEBUG: Feature statistics
            if batch_idx < 10:
                feat_norm = features.float().norm(dim=-1).mean().item()
                feat_mean = features.float().mean().item()
                feat_std = features.float().std().item()
                logger.info(f"[Batch {batch_idx}] Features: norm={feat_norm:.6e}, "
                          f"mean={feat_mean:.6e}, std={feat_std:.6e}")
        
        # Forward through probe
        logits = probe(features)
        
        # DEBUG: Label distribution
        unique_labels, counts = torch.unique(labels, return_counts=True)
        label_dist = {int(l): int(c) for l, c in zip(unique_labels, counts)}
        all_labels.append(label_dist)
        
        # DEBUG: Logits statistics
        logits_detached = logits.detach()
        logits_stats = {
            "mean": logits_detached.mean().item(),
            "std": logits_detached.std().item(),
            "min": logits_detached.min().item(),
            "max": logits_detached.max().item()
        }
        all_logits_stats.append(logits_stats)
        
        # Compute loss (try both weighted and unweighted)
        loss_unweighted = F.cross_entropy(logits, labels)
        
        # Weighted loss
        n_classes = logits.size(1)
        class_counts = torch.bincount(labels, minlength=n_classes)
        class_weights = 1.0 / (class_counts.float() + 1e-5)
        class_weights = class_weights / class_weights.sum()
        loss_weighted = F.cross_entropy(logits, labels, weight=class_weights.to(device))
        
        # Manual BCE calculation for binary case
        if n_classes == 2:
            probs = torch.softmax(logits, dim=1)[:, 1]
            labels_float = labels.float()
            manual_bce = -(labels_float * torch.log(probs + 1e-12) + 
                          (1 - labels_float) * torch.log(1 - probs + 1e-12)).mean()
        else:
            manual_bce = torch.tensor(float('nan'))
        
        # Backward pass
        optimizer.zero_grad()
        loss_weighted.backward()
        
        # DEBUG: Check gradients
        grad_norms = []
        for name, param in probe.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.abs().mean().item()
                grad_norms.append(grad_norm)
                if batch_idx < 10:
                    logger.info(f"[Batch {batch_idx}] Grad {name}: {grad_norm:.6e}")
        
        mean_grad = np.mean(grad_norms) if grad_norms else 0
        all_grads.append(mean_grad)
        
        optimizer.step()
        
        # Store loss
        all_losses.append({
            "unweighted": loss_unweighted.item(),
            "weighted": loss_weighted.item(),
            "manual_bce": manual_bce.item() if not torch.isnan(manual_bce) else None
        })
        
        # Detailed logging for first 10 and every 20th batch
        if batch_idx < 10 or batch_idx % 20 == 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"BATCH {batch_idx}")
            logger.info(f"Labels: {label_dist}")
            logger.info(f"Logits: mean={logits_stats['mean']:.6e}, std={logits_stats['std']:.6e}, "
                      f"range=[{logits_stats['min']:.6e}, {logits_stats['max']:.6e}]")
            logger.info(f"Loss (unweighted): {loss_unweighted.item():.8e}")
            logger.info(f"Loss (weighted): {loss_weighted.item():.8e}")
            if manual_bce is not None and not torch.isnan(manual_bce):
                logger.info(f"Loss (manual BCE): {manual_bce.item():.8e}")
            logger.info(f"Mean gradient: {mean_grad:.8e}")
            
            # Check if loss is truly zero
            if loss_weighted.item() == 0.0:
                logger.warning("⚠️ LOSS IS EXACTLY ZERO!")
            elif loss_weighted.item() < 1e-4:
                logger.info(f"✓ Loss is small but non-zero: {loss_weighted.item():.12e}")
    
    # Summary statistics
    logger.info("\n" + "="*80)
    logger.info("SUMMARY AFTER 200 BATCHES")
    logger.info("="*80)
    
    # Loss statistics
    losses_array = np.array([l["weighted"] for l in all_losses])
    logger.info(f"\nLOSS STATISTICS:")
    logger.info(f"  Mean: {losses_array.mean():.8e}")
    logger.info(f"  Std: {losses_array.std():.8e}")
    logger.info(f"  Min: {losses_array.min():.8e}")
    logger.info(f"  Max: {losses_array.max():.8e}")
    logger.info(f"  Zeros: {np.sum(losses_array == 0)} / {len(losses_array)}")
    logger.info(f"  < 1e-4: {np.sum(losses_array < 1e-4)} / {len(losses_array)}")
    logger.info(f"  < 1e-6: {np.sum(losses_array < 1e-6)} / {len(losses_array)}")
    
    # Label distribution
    total_label_counts = {0: 0, 1: 0}
    for batch_labels in all_labels:
        for label, count in batch_labels.items():
            total_label_counts[label] = total_label_counts.get(label, 0) + count
    
    total_samples = sum(total_label_counts.values())
    logger.info(f"\nLABEL DISTRIBUTION:")
    for label, count in total_label_counts.items():
        pct = 100 * count / total_samples
        logger.info(f"  Class {label}: {count} ({pct:.1f}%)")
    
    # Gradient flow
    grads_array = np.array(all_grads)
    logger.info(f"\nGRADIENT STATISTICS:")
    logger.info(f"  Mean: {grads_array.mean():.8e}")
    logger.info(f"  Std: {grads_array.std():.8e}")
    logger.info(f"  Zeros: {np.sum(grads_array == 0)} / {len(grads_array)}")
    
    # Logits evolution
    logits_means = [s["mean"] for s in all_logits_stats]
    logger.info(f"\nLOGITS EVOLUTION:")
    logger.info(f"  First 10: {logits_means[:10]}")
    logger.info(f"  Last 10: {logits_means[-10:]}")
    
    # Recommendations
    logger.info("\n" + "="*80)
    logger.info("DIAGNOSIS AND RECOMMENDATIONS")
    logger.info("="*80)
    
    if np.all(losses_array == 0):
        logger.error("❌ ALL LOSSES ARE EXACTLY ZERO - CRITICAL BUG!")
        logger.error("   Check: dtype issues, loss function, no_grad context")
    elif np.mean(losses_array) < 1e-6:
        logger.warning("⚠️ Losses are extremely small - likely display rounding")
        logger.warning("   Fix: Change tqdm format from .4f to .6e")
    elif np.std(losses_array) < 1e-8:
        logger.warning("⚠️ Loss variance too low - model may be stuck")
        logger.warning("   Check: learning rate, initialization, feature extraction")
    else:
        logger.info("✅ Loss values appear normal - just a display issue")
        logger.info("   Fix: Update progress bar format to show scientific notation")
    
    if np.mean(grads_array) == 0:
        logger.error("❌ NO GRADIENTS FLOWING - training is broken!")
    elif np.mean(grads_array) < 1e-8:
        logger.warning("⚠️ Gradients very small - may need higher LR")
    else:
        logger.info(f"✅ Gradients flowing normally: {np.mean(grads_array):.6e}")
    
    class_imbalance = max(total_label_counts.values()) / min(total_label_counts.values())
    if class_imbalance > 5:
        logger.warning(f"⚠️ High class imbalance: {class_imbalance:.1f}:1")
        logger.warning("   Consider: focal loss, SMOTE, or stronger class weights")

if __name__ == "__main__":
    investigate_loss()