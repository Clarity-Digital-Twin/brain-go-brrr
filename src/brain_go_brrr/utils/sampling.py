"""Sampling utilities for handling imbalanced datasets."""

import json
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import WeightedRandomSampler


def compute_class_weights(
    labels: npt.NDArray[np.int_], method: Literal["counts", "cb"] = "counts", beta: float = 0.9999
) -> torch.Tensor:
    """Compute class weights for imbalanced datasets.

    Args:
        labels: Array of integer labels
        method:
            - "counts": Inverse frequency weighting (1/n_c)
            - "cb": Class-balanced loss weighting (1-β)/(1-β^n_c)
        beta: Beta parameter for CB loss (typically 0.99-0.9999)

    Returns:
        Tensor of class weights normalized to sum to num_classes
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    num_classes = len(unique_labels)

    if method == "counts":
        # Inverse frequency
        weights = 1.0 / counts
    elif method == "cb":
        # Class-balanced loss: (1-β)/(1-β^n_c)
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / effective_num
    else:
        raise ValueError(f"Unknown method: {method}")

    # Normalize weights to sum to num_classes
    weights = weights / weights.sum() * num_classes

    # Convert to tensor
    weight_tensor = torch.tensor(weights, dtype=torch.float32)

    return weight_tensor


def load_cache_labels(cache_dir: Path, split: str = "train") -> npt.NDArray[np.int_]:
    """Load labels from TUEV cache index.

    Args:
        cache_dir: Path to cache directory
        split: "train" or "eval"

    Returns:
        Array of integer labels
    """
    index_path = cache_dir / split / "index.json"

    with index_path.open() as f:
        index_data = json.load(f)

    labels = np.array([item['label'] for item in index_data['segments']])
    return labels


def create_weighted_sampler(
    labels: npt.NDArray[np.int_], class_weights: torch.Tensor, replacement: bool = True
) -> WeightedRandomSampler:
    """Create a weighted sampler for balanced batch sampling.

    Args:
        labels: Array of integer labels for the dataset
        class_weights: Per-class weights from compute_class_weights
        replacement: Whether to sample with replacement

    Returns:
        WeightedRandomSampler configured for the dataset
    """
    # Create per-sample weights based on class weights
    sample_weights_list = [class_weights[label].item() for label in labels]

    # Create sampler (WeightedRandomSampler expects Sequence[float], not Tensor)
    sampler = WeightedRandomSampler(
        weights=sample_weights_list, num_samples=len(labels), replacement=replacement
    )

    return sampler


def get_minority_classes(labels: npt.NDArray[np.int_], threshold: float = 0.1) -> list[int]:
    """Identify minority classes based on frequency threshold.

    Args:
        labels: Array of integer labels
        threshold: Frequency threshold (e.g., 0.1 = classes with <10% of samples)

    Returns:
        List of minority class indices
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    frequencies = counts / total

    minority_classes_array = unique_labels[frequencies < threshold]
    minority_classes: list[int] = minority_classes_array.tolist()

    return minority_classes


def print_class_distribution(
    labels: npt.NDArray[np.int_], class_names: dict[int, str] | None = None
) -> None:
    """Print class distribution statistics.

    Args:
        labels: Array of integer labels
        class_names: Optional mapping from class index to name
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    total = len(labels)

    if class_names is None:
        class_names = {i: f"Class {i}" for i in unique_labels}

    print("\nClass Distribution:")
    print("-" * 50)
    for label, count in zip(unique_labels, counts, strict=False):
        name = class_names.get(label, f"Class {label}")
        percentage = count / total * 100
        print(f"{name:15s}: {count:6d} ({percentage:5.1f}%)")
    print("-" * 50)
    print(f"Total samples: {total}")
    print(
        f"Class imbalance ratio: {counts.max()}/{counts.min()} = {counts.max() / counts.min():.1f}:1"
    )


def augment_minority_sample(
    signal: torch.Tensor,
    label: int,
    minority_classes: list[int],
    shift_ms: int = 200,
    jitter_uv: float = 5.0,
    noise_uv: float = 5.0,
    augment_prob: float = 0.3,
    sampling_rate: int = 200,
) -> torch.Tensor:
    """Apply augmentation to minority class samples.

    Args:
        signal: EEG signal tensor [channels, time_points]
        label: Class label
        minority_classes: List of minority class indices
        shift_ms: Max time shift in milliseconds
        jitter_uv: Amplitude jitter in microvolts
        noise_uv: Noise std in microvolts
        augment_prob: Probability of applying augmentation
        sampling_rate: Sampling rate in Hz

    Returns:
        Augmented signal (or original if not minority class)
    """
    # Only augment minority classes with given probability
    if label not in minority_classes or np.random.random() > augment_prob:
        return signal

    augmented = signal.clone()

    # Time shift (within bounds)
    if shift_ms > 0:
        shift_samples = int(shift_ms * sampling_rate / 1000)
        actual_shift = np.random.randint(-shift_samples, shift_samples + 1)
        if actual_shift != 0:
            augmented = torch.roll(augmented, shifts=actual_shift, dims=-1)

    # Amplitude jitter (multiplicative)
    if jitter_uv > 0:
        # Convert to relative scale (assuming signal is in Volts)
        # jitter_uv is in microvolts, so convert to volts
        jitter_scale = (
            1.0 + (np.random.uniform(-jitter_uv, jitter_uv) / 1e6) / augmented.std().item()
        )
        augmented = augmented * jitter_scale

    # Additive noise
    if noise_uv > 0:
        # Convert noise from microvolts to volts
        noise_std_volts = noise_uv / 1e6
        noise = torch.randn_like(augmented) * noise_std_volts
        augmented = augmented + noise

    return augmented
