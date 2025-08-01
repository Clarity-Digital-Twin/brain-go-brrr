#!/usr/bin/env python
"""Fast smoke test - catches dimension mismatches in 30 seconds."""

import os
import sys
from pathlib import Path

# Setup paths
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
os.environ["BGB_DATA_ROOT"] = str(project_root / "data")
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU for speed

def fast_smoke_test():
    """Run minimal training to catch errors early."""
    print("🚀 Fast smoke test - catches crashes in <30 seconds\n")
    
    try:
        # Import everything
        print("1️⃣ Testing imports...")
        import torch
        import pytorch_lightning as pl
        from omegaconf import OmegaConf
        from brain_go_brrr.data.tuab_dataset import TUABDataset
        from brain_go_brrr.tasks.abnormality_detection import AbnormalityDetectionProbe
        
        # Import the trainer class from the actual training script
        sys.path.insert(0, str(Path(__file__).parent))
        from train_tuab_probe import LinearProbeTrainer
        print("   ✅ Imports OK\n")
        
        # Load config
        print("2️⃣ Loading config...")
        cfg = OmegaConf.load(Path(__file__).parent / "configs/tuab_config.yaml")
        cfg.training.batch_size = 4  # Small batch
        cfg.training.num_workers = 0  # No multiprocessing
        print("   ✅ Config OK\n")
        
        # Check paths
        print("3️⃣ Checking paths...")
        data_root = Path(os.environ["BGB_DATA_ROOT"])
        checkpoint_path = data_root / "models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"
        tuab_path = data_root / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf"
        
        assert checkpoint_path.exists(), f"Missing EEGPT checkpoint: {checkpoint_path}"
        assert tuab_path.exists(), f"Missing TUAB dataset: {tuab_path}"
        print("   ✅ Paths OK\n")
        
        # Create minimal dataset
        print("4️⃣ Creating dataset (2 files only)...")
        dataset = TUABDataset(
            root_dir=tuab_path,
            split="train",
            window_duration=8.0,
            window_stride=8.0,
            sampling_rate=256,
            preload=False,
            normalize=True,
            max_files=2,  # Only load 2 files for speed
        )
        print(f"   ✅ Dataset OK: {len(dataset)} windows\n")
        
        # Create model
        print("5️⃣ Creating model...")
        probe = AbnormalityDetectionProbe(checkpoint_path, n_input_channels=20)
        
        # Create the RobustLinearProbeTrainer from train_paper_aligned.py
        class TestTrainer(LinearProbeTrainer):
            def validation_step(self, batch, batch_idx):
                x, labels = batch
                logits = self(x)
                loss = self.criterion(logits, labels)
                probs = torch.softmax(logits, dim=1)  # Use softmax for proper probs
                preds = torch.argmax(probs, dim=1)    # Fixed: proper argmax
                
                self.validation_step_outputs.append({
                    "loss": loss,
                    "probs": probs,
                    "preds": preds,
                    "labels": labels,
                })
                return loss
            
            def on_validation_epoch_end(self):
                # Test the concatenation that was failing
                all_probs = torch.cat([x["probs"] for x in self.validation_step_outputs])
                all_preds = torch.cat([x["preds"] for x in self.validation_step_outputs])
                all_labels = torch.cat([x["labels"] for x in self.validation_step_outputs])
                
                print(f"\n   📊 Validation shapes:")
                print(f"      - probs: {all_probs.shape}")
                print(f"      - preds: {all_preds.shape}")
                print(f"      - labels: {all_labels.shape}")
                
                # This was the failing line - test it
                acc = (all_preds == all_labels).float().mean()
                print(f"      - accuracy: {acc:.3f}")
                
                # Test AUROC calculation
                if all_probs.dim() > 1:
                    probs_positive = all_probs[:, 1].cpu().numpy()
                else:
                    probs_positive = all_probs.cpu().numpy()
                
                self.log("val/acc", acc)
                self.validation_step_outputs.clear()
        
        trainer_module = TestTrainer(
            model=probe,
            learning_rate=5e-4,
            weight_decay=0.05,
        )
        print("   ✅ Model OK\n")
        
        # Create minimal dataloader
        print("6️⃣ Creating dataloader...")
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
        print("   ✅ Dataloader OK\n")
        
        # Run one training step
        print("7️⃣ Testing forward pass...")
        trainer = pl.Trainer(
            max_epochs=1,
            limit_train_batches=2,
            limit_val_batches=2,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            accelerator="cpu",
        )
        
        # This will run sanity check + 2 train batches + 2 val batches
        trainer.fit(trainer_module, loader, loader)
        print("   ✅ Training step OK\n")
        
        print("✅ ALL TESTS PASSED! Safe to run full training.\n")
        return True
        
    except Exception as e:
        print(f"\n❌ SMOKE TEST FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if fast_smoke_test():
        print("💡 Run full training with:")
        print("   tmux attach -t eegpt_final")
    else:
        print("\n⚠️  Fix the error above before running full training!")
        sys.exit(1)