"""Training script for EEGPT sleep stage linear probe.

This script implements a simple PyTorch training loop to fit a linear probe
on top of frozen EEGPT features for sleep stage classification.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from brain_go_brrr.core.edf_loader import load_edf_safe
from brain_go_brrr.models.eegpt_model import EEGPTModel
from brain_go_brrr.models.linear_probe import SleepStageProbe

logger = logging.getLogger(__name__)


class SleepDataset(Dataset):
    """PyTorch dataset for sleep EEG windows."""

    def __init__(
        self,
        windows: list[np.ndarray],
        labels: list[int],
        augment: bool = False,
    ):
        """Initialize sleep dataset.

        Args:
            windows: List of EEG windows (n_channels, n_samples)
            labels: List of sleep stage labels (0-4)
            augment: Whether to apply data augmentation
        """
        self.windows = windows
        self.labels = labels
        self.augment = augment

    def __len__(self) -> int:
        """Return number of windows."""
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get window and label at index."""
        window = self.windows[idx].copy()
        label = self.labels[idx]

        # Apply augmentation if enabled
        if self.augment:
            # Add small Gaussian noise
            noise = np.random.normal(0, 0.1, window.shape)
            window = window + noise

        return torch.FloatTensor(window), torch.LongTensor([label])


class SleepProbeTrainer:
    """Trainer for sleep stage linear probe."""

    def __init__(
        self,
        probe: SleepStageProbe,
        eegpt_model: EEGPTModel,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        device: str = "cpu",
    ):
        """Initialize trainer.

        Args:
            probe: Sleep stage probe to train
            eegpt_model: Pretrained EEGPT model for features
            learning_rate: Learning rate for Adam optimizer
            batch_size: Batch size for training
            device: Device to train on (cpu/cuda)
        """
        self.probe = probe.to(device)
        self.eegpt_model = eegpt_model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device

        # Only train the probe, not EEGPT
        self.optimizer = Adam(self.probe.parameters(), lr=learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train_step(self, batch_windows: torch.Tensor, batch_labels: torch.Tensor) -> float:
        """Perform single training step.

        Args:
            batch_windows: Batch of EEG windows
            batch_labels: Batch of labels

        Returns:
            Loss value
        """
        self.probe.train()
        self.optimizer.zero_grad()

        # Extract EEGPT features for batch
        batch_features = []
        for window in batch_windows:
            window_np = window.numpy()
            # Assume standard channel names for now
            channel_names = [f"Ch{i}" for i in range(window_np.shape[0])]
            features = self.eegpt_model.extract_features(window_np, channel_names)
            batch_features.append(features.flatten())

        # Stack features
        features_tensor = torch.FloatTensor(np.stack(batch_features)).to(self.device)
        batch_labels = batch_labels.squeeze().to(self.device)

        # Forward pass through probe
        logits = self.probe(features_tensor)
        loss = self.criterion(logits, batch_labels)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def train_epoch(self, dataset: SleepDataset) -> float:
        """Train for one epoch.

        Args:
            dataset: Sleep dataset

        Returns:
            Average loss for epoch
        """
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        total_loss = 0.0
        n_batches = 0

        for batch_windows, batch_labels in dataloader:
            loss = self.train_step(batch_windows, batch_labels)
            total_loss += loss
            n_batches += 1

        return total_loss / n_batches if n_batches > 0 else 0.0

    def save_checkpoint(self, path: Path, epoch: int, accuracy: float) -> None:
        """Save training checkpoint.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            accuracy: Current accuracy
        """
        checkpoint = {
            "model_state_dict": self.probe.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "accuracy": accuracy,
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")


def evaluate_probe(
    probe: SleepStageProbe,
    eegpt_model: EEGPTModel,
    dataset: SleepDataset,
    batch_size: int = 32,
) -> tuple[float, np.ndarray]:
    """Evaluate probe on dataset.

    Args:
        probe: Trained probe
        eegpt_model: EEGPT model for features
        dataset: Dataset to evaluate on
        batch_size: Batch size

    Returns:
        Accuracy and confusion matrix
    """
    probe.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch_windows, batch_labels in dataloader:
            # Extract features
            batch_features = []
            for window in batch_windows:
                window_np = window.numpy()
                channel_names = [f"Ch{i}" for i in range(window_np.shape[0])]
                features = eegpt_model.extract_features(window_np, channel_names)
                batch_features.append(features.flatten())

            features_tensor = torch.FloatTensor(np.stack(batch_features))

            # Get predictions
            logits = probe(features_tensor)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.squeeze().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=range(5))

    return accuracy, conf_matrix


def load_sleep_edf_data(
    data_dir: Path, max_files: int | None = None
) -> tuple[list[np.ndarray], list[int]]:
    """Load Sleep-EDF dataset.

    Args:
        data_dir: Directory containing Sleep-EDF files
        max_files: Maximum number of files to load

    Returns:
        Windows and labels
    """
    windows = []
    labels = []

    # Find all EDF files
    edf_files = list(data_dir.glob("**/*.edf"))
    if max_files:
        edf_files = edf_files[:max_files]

    logger.info(f"Loading {len(edf_files)} EDF files from {data_dir}")

    for edf_file in edf_files:
        try:
            # Load EDF
            raw = load_edf_safe(edf_file, preload=True, verbose=False)

            # Extract windows
            file_windows = extract_windows_from_raw(raw)

            # For now, use random labels (would load from annotations)
            file_labels = [i % 5 for i in range(len(file_windows))]

            windows.extend(file_windows)
            labels.extend(file_labels)

        except Exception as e:
            logger.warning(f"Failed to load {edf_file}: {e}")
            continue

    logger.info(f"Loaded {len(windows)} windows from {len(edf_files)} files")
    return windows, labels


def extract_windows_from_raw(raw: Any, window_duration: float = 4.0) -> list[np.ndarray]:
    """Extract fixed-duration windows from Raw object.

    Args:
        raw: MNE Raw object
        window_duration: Window duration in seconds

    Returns:
        List of windows
    """
    data = raw.get_data()
    sfreq = int(raw.info["sfreq"])
    window_samples = int(window_duration * sfreq)

    windows = []
    for start in range(0, data.shape[1] - window_samples, window_samples):
        window = data[:, start : start + window_samples]
        windows.append(window)

    return windows


def split_train_val(
    windows: list[np.ndarray],
    labels: list[int],
    val_split: float = 0.2,
    random_seed: int = 42,
) -> tuple[tuple[list, list], tuple[list, list]]:
    """Split data into train and validation sets.

    Args:
        windows: List of EEG windows
        labels: List of labels
        val_split: Validation split ratio
        random_seed: Random seed

    Returns:
        (train_windows, train_labels), (val_windows, val_labels)
    """
    # Set random seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Calculate split
    n_samples = len(windows)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val

    # Random indices
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    # Split data
    train_windows = [windows[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_windows = [windows[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    return (train_windows, train_labels), (val_windows, val_labels)


def train_sleep_probe(
    data_dir: Path,
    eegpt_model: EEGPTModel,
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    validation_split: float = 0.2,
    checkpoint_dir: Path | None = None,
) -> tuple[SleepStageProbe, dict[str, Any]]:
    """Train sleep stage probe on Sleep-EDF data.

    Args:
        data_dir: Directory containing Sleep-EDF files
        eegpt_model: Pretrained EEGPT model
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        validation_split: Validation split ratio
        checkpoint_dir: Directory to save checkpoints

    Returns:
        Trained probe and metrics dictionary
    """
    # Load data
    windows, labels = load_sleep_edf_data(data_dir, max_files=10)

    # Split train/val
    (train_windows, train_labels), (val_windows, val_labels) = split_train_val(
        windows, labels, validation_split
    )

    # Create datasets
    train_dataset = SleepDataset(train_windows, train_labels, augment=True)
    val_dataset = SleepDataset(val_windows, val_labels, augment=False)

    # Initialize probe and trainer
    probe = SleepStageProbe()
    trainer = SleepProbeTrainer(probe, eegpt_model, learning_rate, batch_size)

    # Training metrics
    train_losses = []
    val_accuracies = []

    # Training loop
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        # Train
        avg_loss = trainer.train_epoch(train_dataset)
        train_losses.append(avg_loss)

        # Evaluate
        val_acc, _ = evaluate_probe(probe, eegpt_model, val_dataset)
        val_accuracies.append(val_acc)

        logger.info(
            f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if checkpoint_dir:
                checkpoint_path = checkpoint_dir / "best_sleep_probe.pt"
                trainer.save_checkpoint(checkpoint_path, epoch, val_acc)

    # Final evaluation
    train_acc, train_cm = evaluate_probe(probe, eegpt_model, train_dataset)
    val_acc, val_cm = evaluate_probe(probe, eegpt_model, val_dataset)

    metrics = {
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "train_losses": train_losses,
        "val_accuracies": val_accuracies,
        "train_confusion_matrix": train_cm,
        "val_confusion_matrix": val_cm,
    }

    return probe, metrics


def load_sleep_annotations(annotation_file: Path) -> dict[int, str]:
    """Load sleep stage annotations from file.

    Args:
        annotation_file: Path to annotation file

    Returns:
        Dictionary mapping epoch index to stage
    """
    return parse_sleep_annotations(annotation_file)


def parse_sleep_annotations(annotation_file: Path) -> dict[int, str]:  # noqa: ARG001
    """Parse sleep annotations from file.

    Args:
        annotation_file: Path to annotation file

    Returns:
        Dictionary mapping epoch index to stage
    """
    # Placeholder - would parse actual annotation format
    return {}


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Train sleep stage probe")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to Sleep-EDF dataset",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory to save checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")

    args = parser.parse_args()

    # Initialize EEGPT
    eegpt_model = EEGPTModel()

    # Train probe
    probe, metrics = train_sleep_probe(
        data_dir=args.data_dir,
        eegpt_model=eegpt_model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir,
    )

    logging.info("Training complete!")
    logging.info(f"Final validation accuracy: {metrics['val_accuracy']:.4f}")
