#!/usr/bin/env python
"""Script to train sleep stage linear probe on Sleep-EDF dataset.

Example usage:
    uv run python scripts/train_sleep_probe.py \
        --data-dir data/datasets/external/sleep-edf \
        --epochs 5 \
        --batch-size 16
"""

import argparse
import logging
from pathlib import Path

from brain_go_brrr.models.eegpt_model import EEGPTModel
from brain_go_brrr.training.sleep_probe_trainer import train_sleep_probe

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train EEGPT sleep stage linear probe")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/datasets/external/sleep-edf"),
        help="Path to Sleep-EDF dataset directory",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/sleep_probe"),
        help="Directory to save checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam learning rate",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=10,
        help="Maximum number of EDF files to load (for quick testing)",
    )

    args = parser.parse_args()

    # Validate data directory
    if not args.data_dir.exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        return 1

    # Create checkpoint directory
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Initialize EEGPT model
    logger.info("Loading EEGPT model...")
    eegpt_model = EEGPTModel()

    if not eegpt_model.is_loaded:
        logger.error("Failed to load EEGPT model")
        return 1

    # Train the probe
    logger.info(f"Training sleep probe on data from {args.data_dir}")
    logger.info(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")

    try:
        probe, metrics = train_sleep_probe(
            data_dir=args.data_dir,
            eegpt_model=eegpt_model,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            checkpoint_dir=args.checkpoint_dir,
        )

        # Print final metrics
        logger.info("\n=== Training Complete ===")
        logger.info(f"Final train accuracy: {metrics['train_accuracy']:.4f}")
        logger.info(f"Final val accuracy: {metrics['val_accuracy']:.4f}")

        # Print confusion matrix
        logger.info("\nValidation Confusion Matrix:")
        logger.info("(rows: true labels, cols: predicted labels)")
        logger.info("Labels: [W, N1, N2, N3, REM]")
        cm = metrics["val_confusion_matrix"]
        for i, row in enumerate(cm):
            logger.info(f"  {['W', 'N1', 'N2', 'N3', 'REM'][i]}: {row}")

        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
