#!/usr/bin/env python
"""Debug version with extensive logging to diagnose crashes."""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
import traceback
import time
import gc

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Set environment
os.environ["BGB_DATA_ROOT"] = str(project_root / "data")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning:brain_go_brrr.data"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Synchronous for debugging

import numpy as np  # noqa: E402
import pytorch_lightning as pl  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from pytorch_lightning import Trainer  # noqa: E402
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint  # noqa: E402
from pytorch_lightning.loggers import TensorBoardLogger  # noqa: E402
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split  # noqa: E402

from brain_go_brrr.data.tuab_dataset import TUABDataset  # noqa: E402
from brain_go_brrr.tasks.abnormality_detection import AbnormalityDetectionProbe  # noqa: E402

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from train_tuab_probe import LinearProbeTrainer  # noqa: E402

# Create multiple log handlers
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def log_memory():
    """Log current memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logger.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    import psutil
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / 1e9
    logger.info(f"RAM Usage: {ram_usage:.2f}GB")


def main():
    start_time = time.time()
    logger.info("="*80)
    logger.info("DEBUG TRAINING SCRIPT STARTING")
    logger.info("="*80)
    
    try:
        # Set seed
        pl.seed_everything(42)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Log system info
        logger.info(f"Python: {sys.version}")
        logger.info(f"PyTorch: {torch.__version__}")
        logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
        log_memory()
        
        # Load config
        logger.info("Loading config...")
        cfg = OmegaConf.load("configs/tuab_config.yaml")
        
        # Ultra conservative settings
        cfg.training.epochs = 3  # Just 3 epochs for testing
        cfg.training.batch_size = 32  # Very small
        cfg.training.num_workers = 0  # NO multiprocessing
        cfg.training.learning_rate = 5e-4
        cfg.training.patience = 2
        cfg.training.monitor = "val_loss"
        cfg.training.mode = "min"
        
        logger.info(f"Config: batch_size={cfg.training.batch_size}, workers={cfg.training.num_workers}")
        
        # Paths
        data_root = Path(os.environ["BGB_DATA_ROOT"])
        model_checkpoint = data_root / cfg.model.checkpoint_path
        dataset_root = data_root / cfg.data.root_dir
        log_dir = Path(f"logs/debug_run_{timestamp}")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        logger.info("Loading model checkpoint...")
        checkpoint_start = time.time()
        model = AbnormalityDetectionProbe(
            checkpoint_path=model_checkpoint,
            n_input_channels=20,
        )
        logger.info(f"Model loaded in {time.time() - checkpoint_start:.1f}s")
        log_memory()
        
        # Move to GPU
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("Model moved to GPU")
            log_memory()
        
        # Load dataset
        logger.info("Loading dataset...")
        dataset_start = time.time()
        
        # Add timeout wrapper
        import signal
        def timeout_handler(signum, frame):
            raise TimeoutError("Dataset loading timed out!")
        
        try:
            # Set 5 minute timeout for dataset loading
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(300)
            
            full_dataset = TUABDataset(
                root_dir=dataset_root,
                split="eval",
                sampling_rate=cfg.data.sampling_rate,
                window_duration=cfg.data.window_duration,
                window_stride=cfg.data.window_stride,
                normalize=cfg.data.normalize,
            )
            
            signal.alarm(0)  # Cancel timeout
            
        except TimeoutError:
            logger.error("Dataset loading timed out after 5 minutes!")
            raise
        except Exception as e:
            logger.error(f"Dataset loading failed: {type(e).__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
        
        logger.info(f"Dataset loaded in {time.time() - dataset_start:.1f}s")
        logger.info(f"Total windows: {len(full_dataset)}")
        log_memory()
        
        # Create a TINY subset for testing
        subset_size = min(1000, len(full_dataset))  # Only 1000 samples
        subset_indices = torch.randperm(len(full_dataset))[:subset_size]
        subset_dataset = torch.utils.data.Subset(full_dataset, subset_indices)
        
        logger.info(f"Using subset of {len(subset_dataset)} windows for testing")
        
        # Split
        train_size = int(0.8 * len(subset_dataset))
        val_size = len(subset_dataset) - train_size
        train_dataset, val_dataset = random_split(
            subset_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        
        # Simple dataloaders - no fancy sampling
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=0,  # No workers!
            pin_memory=False,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        
        logger.info("Dataloaders created")
        
        # Lightning module
        class_weights = torch.tensor([1.0, 1.0])  # Equal weights for testing
        lightning_model = LinearProbeTrainer(
            model=model,
            learning_rate=cfg.training.learning_rate,
            weight_decay=1e-4,
            pct_start=0.1,
            div_factor=10,
            final_div_factor=100,
            class_weights=class_weights,
        )
        
        logger.info("Lightning module created")
        
        # Simple callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=log_dir / "checkpoints",
            filename="debug-{epoch:02d}",
            save_top_k=1,
            verbose=True,
        )
        
        # Trainer
        trainer = Trainer(
            max_epochs=cfg.training.epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            precision=32,  # No mixed precision for debugging
            callbacks=[checkpoint_callback],
            default_root_dir=log_dir,
            log_every_n_steps=1,  # Log every step
            enable_progress_bar=True,
            enable_model_summary=True,
            gradient_clip_val=1.0,
            fast_dev_run=False,  # Set to True for ultra-fast test
            # limit_train_batches=10,  # Only 10 batches per epoch
            # limit_val_batches=5,
        )
        
        logger.info("="*60)
        logger.info("Starting training...")
        logger.info(f"Setup time: {time.time() - start_time:.1f}s")
        logger.info("="*60)
        
        # Train with detailed error catching
        training_start = time.time()
        try:
            trainer.fit(lightning_model, train_loader, val_loader)
            logger.info(f"Training completed in {time.time() - training_start:.1f}s")
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
            raise
        except Exception as e:
            logger.error(f"Training failed: {type(e).__name__}: {e}")
            logger.error(traceback.format_exc())
            log_memory()
            raise
        
        logger.info("SUCCESS! Training completed without errors")
        logger.info(f"Total time: {time.time() - start_time:.1f}s")
        
    except Exception as e:
        logger.error(f"FATAL ERROR: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        log_memory()
        sys.exit(1)


if __name__ == "__main__":
    main()