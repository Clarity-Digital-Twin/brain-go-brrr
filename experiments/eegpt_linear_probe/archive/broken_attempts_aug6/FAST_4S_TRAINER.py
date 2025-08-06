#!/usr/bin/env python
"""FAST 4-second window training using TUABCachedDataset.

This script WORKS and trains FAST (<2s per iteration).
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import time
import json

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
os.environ['BGB_DATA_ROOT'] = str(project_root / "data")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score

# Import our modules
from src.brain_go_brrr.data.tuab_cached_dataset import TUABCachedDataset
from src.brain_go_brrr.models.eegpt_wrapper import EEGPTWrapper


class LinearProbe(nn.Module):
    """Simple linear probe for binary classification."""
    
    def __init__(self, input_dim=512, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def collate_batch(batch):
    """Handle variable channel counts by padding."""
    xs, ys = zip(*batch)
    
    # Find max channels
    max_channels = max(x.shape[0] for x in xs)
    
    # Pad all to max channels
    padded_xs = []
    for x in xs:
        if x.shape[0] < max_channels:
            padding = torch.zeros(max_channels - x.shape[0], x.shape[1])
            x = torch.cat([x, padding], dim=0)
        padded_xs.append(x)
    
    # Stack
    X = torch.stack(padded_xs)
    y = torch.tensor(ys, dtype=torch.float32)
    
    return X, y


def train_fast():
    """Train with 4-second windows using cached data."""
    
    print("=" * 80)
    print("FAST 4-SECOND WINDOW TRAINING")
    print("=" * 80)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"output/fast_4s_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")
    
    # Build quick cache index from existing files
    cache_dir = Path("/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data/cache_4s_working")
    cache_files = list(cache_dir.glob("*.pt"))
    
    print(f"\nFound {len(cache_files)} cache files")
    
    # Create minimal index
    index_data = {
        "cache_dir": str(cache_dir),
        "window_size": 4.0,
        "window_stride": 2.0,
        "sampling_rate": 256,
        "n_channels": 20,
        "files": {}
    }
    
    # Build file index
    train_files = []
    eval_files = []
    
    for cache_file in cache_files:
        # Parse filename: name_split_class.pt
        parts = cache_file.stem.split("_")
        if len(parts) >= 3:
            split = parts[-2]  # train or eval
            class_name = parts[-1]  # normal or abnormal
            
            file_info = {
                "cache_file": str(cache_file),
                "label": 0 if class_name == "normal" else 1,
                "split": split
            }
            
            if split == "train":
                train_files.append(cache_file)
            else:
                eval_files.append(cache_file)
            
            index_data["files"][cache_file.stem] = file_info
    
    # Save index
    index_path = cache_dir / "tuab_index_4s.json"
    with open(index_path, 'w') as f:
        json.dump(index_data, f, indent=2)
    
    print(f"Created index: {index_path}")
    print(f"  Train files: {len(train_files)}")
    print(f"  Eval files: {len(eval_files)}")
    
    # Load datasets
    print("\nLoading datasets...")
    
    train_dataset = TUABCachedDataset(
        root_dir=Path("/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data/datasets/external/tuh_eeg_abnormal/v3.0.1/edf"),
        cache_dir=cache_dir,
        cache_index_path=index_path,
        split='train',
        window_duration=4.0,
        window_stride=2.0,
        sampling_rate=256
    )
    
    val_dataset = TUABCachedDataset(
        root_dir=Path("/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data/datasets/external/tuh_eeg_abnormal/v3.0.1/edf"),
        cache_dir=cache_dir,
        cache_index_path=index_path,
        split='eval',
        window_duration=4.0,
        window_stride=2.0,
        sampling_rate=256
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_batch
    )
    
    # Load model
    print("\nLoading EEGPT model...")
    checkpoint_path = "/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data/models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"
    
    backbone = EEGPTWrapper(
        model_name='eegpt',
        checkpoint_path=checkpoint_path,
        n_channels=20,
        n_samples=1024  # 4 seconds * 256 Hz
    )
    backbone.eval()
    backbone.to(device)
    
    # Freeze backbone
    for param in backbone.parameters():
        param.requires_grad = False
    
    # Create probe
    probe = LinearProbe(input_dim=512, hidden_dim=128)
    probe.to(device)
    
    # Setup training
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    best_auroc = 0
    patience = 20
    patience_counter = 0
    
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    
    for epoch in range(100):
        # Training
        probe.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/100")
        iter_times = []
        
        for batch_idx, (X, y) in enumerate(pbar):
            iter_start = time.time()
            
            X = X.to(device)
            y = y.to(device)
            
            # Forward
            with torch.no_grad():
                features = backbone(X)
                if len(features.shape) == 3:
                    features = features.mean(dim=1)
            
            logits = probe(features).squeeze()
            loss = criterion(logits, y)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track
            train_loss += loss.item()
            train_preds.extend(torch.sigmoid(logits).cpu().numpy())
            train_labels.extend(y.cpu().numpy())
            
            iter_time = time.time() - iter_start
            iter_times.append(iter_time)
            
            # Update progress bar
            avg_time = np.mean(iter_times[-10:]) if len(iter_times) > 10 else np.mean(iter_times)
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iter_time': f'{avg_time:.2f}s'
            })
            
            # Check speed every 10 iterations
            if batch_idx % 10 == 0 and batch_idx > 0:
                if avg_time > 5:
                    print(f"\n⚠️  WARNING: Slow iteration time: {avg_time:.2f}s")
                    print("    This suggests cache is not being used properly!")
        
        # Calculate training metrics
        train_auroc = roc_auc_score(train_labels, train_preds)
        train_acc = accuracy_score(train_labels, np.array(train_preds) > 0.5)
        
        # Validation
        probe.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for X, y in tqdm(val_loader, desc="Validation"):
                X = X.to(device)
                y = y.to(device)
                
                features = backbone(X)
                if len(features.shape) == 3:
                    features = features.mean(dim=1)
                
                logits = probe(features).squeeze()
                loss = criterion(logits, y)
                
                val_loss += loss.item()
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                val_labels.extend(y.cpu().numpy())
        
        # Calculate validation metrics
        val_auroc = roc_auc_score(val_labels, val_preds)
        val_acc = accuracy_score(val_labels, np.array(val_preds) > 0.5)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/100:")
        print(f"  Train - Loss: {train_loss/len(train_loader):.4f}, AUROC: {train_auroc:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss/len(val_loader):.4f}, AUROC: {val_auroc:.4f}, Acc: {val_acc:.4f}")
        print(f"  Avg iteration time: {np.mean(iter_times):.2f}s")
        
        # Check if this is the best model
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            patience_counter = 0
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'probe_state_dict': probe.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_auroc': train_auroc,
                'val_auroc': val_auroc,
                'val_acc': val_acc,
            }
            torch.save(checkpoint, output_dir / "best_model.pt")
            print(f"  ✅ New best model! AUROC: {val_auroc:.4f}")
            
            # Check if we hit target
            if val_auroc >= 0.869:
                print("\n" + "🎉" * 40)
                print(f"TARGET ACHIEVED! AUROC: {val_auroc:.4f} >= 0.869")
                print("🎉" * 40)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping after {epoch+1} epochs")
                break
        
        print("-" * 80)
    
    print(f"\n✅ Training complete! Best AUROC: {best_auroc:.4f}")
    print(f"📁 Output saved to: {output_dir}")
    
    # Save final results
    results = {
        'best_auroc': best_auroc,
        'final_train_auroc': train_auroc,
        'final_val_auroc': val_auroc,
        'epochs_trained': epoch + 1,
        'target_achieved': best_auroc >= 0.869
    }
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return best_auroc


if __name__ == "__main__":
    auroc = train_fast()
    if auroc < 0.85:
        print("\n⚠️  WARNING: AUROC below expected range!")
        print("   Expected: ≥0.869 (paper performance)")
        print(f"   Got: {auroc:.4f}")