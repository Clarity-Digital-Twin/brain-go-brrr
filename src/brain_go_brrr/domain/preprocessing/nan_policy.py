"""NaN handling policy for EEG data preprocessing.

This module provides a consistent policy for handling NaN and Inf values
in input data across the entire codebase.
"""

import logging
from typing import TypeVar

import numpy as np
import numpy.typing as npt
import torch

logger = logging.getLogger(__name__)

T = TypeVar("T", npt.NDArray[np.float32], torch.Tensor)


def validate_no_nan(data: T, name: str = "input") -> T:
    """Validate that data contains no NaN or Inf values.

    Args:
        data: Input data (numpy array or torch tensor)
        name: Name of the data for error messages

    Returns:
        The input data unchanged if valid

    Raises:
        ValueError: If data contains NaN or Inf values
    """
    if isinstance(data, np.ndarray):
        if np.isnan(data).any():
            raise ValueError(f"NaN detected in {name}")
        if np.isinf(data).any():
            raise ValueError(f"Inf detected in {name}")
    elif isinstance(data, torch.Tensor):
        if torch.isnan(data).any():
            raise ValueError(f"NaN detected in {name}")
        if torch.isinf(data).any():
            raise ValueError(f"Inf detected in {name}")
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

    return data


def sanitize_data(data: T, method: str = "zero", name: str = "input") -> T:
    """Sanitize data by replacing NaN/Inf values.

    Args:
        data: Input data (numpy array or torch tensor)
        method: Replacement method - "zero", "median", or "mean"
        name: Name of the data for logging

    Returns:
        Sanitized data with NaN/Inf replaced
    """
    if isinstance(data, np.ndarray):
        nan_mask = np.isnan(data)
        inf_mask = np.isinf(data)

        if nan_mask.any() or inf_mask.any():
            logger.warning(f"Sanitizing {name}: {nan_mask.sum()} NaN, {inf_mask.sum()} Inf values")

            if method == "zero":
                data = np.where(nan_mask | inf_mask, 0, data)
            elif method == "median":
                # Channel-wise median for EEG data
                if data.ndim >= 2:
                    for ch in range(data.shape[0]):
                        ch_data = data[ch]
                        valid = ~(np.isnan(ch_data) | np.isinf(ch_data))
                        if valid.any():
                            median_val = np.median(ch_data[valid])
                            data[ch] = np.where(valid, ch_data, median_val)
                else:
                    valid = ~(nan_mask | inf_mask)
                    if valid.any():
                        median_val = np.median(data[valid])
                        data = np.where(valid, data, median_val)
            elif method == "mean":
                # Channel-wise mean for EEG data
                if data.ndim >= 2:
                    for ch in range(data.shape[0]):
                        ch_data = data[ch]
                        valid = ~(np.isnan(ch_data) | np.isinf(ch_data))
                        if valid.any():
                            mean_val = np.mean(ch_data[valid])
                            data[ch] = np.where(valid, ch_data, mean_val)
                else:
                    valid = ~(nan_mask | inf_mask)
                    if valid.any():
                        mean_val = np.mean(data[valid])
                        data = np.where(valid, data, mean_val)
            else:
                raise ValueError(f"Unknown sanitization method: {method}")

    elif isinstance(data, torch.Tensor):
        nan_mask = torch.isnan(data)
        inf_mask = torch.isinf(data)

        if nan_mask.any() or inf_mask.any():
            logger.warning(f"Sanitizing {name}: {nan_mask.sum()} NaN, {inf_mask.sum()} Inf values")

            if method == "zero":
                data = torch.where(nan_mask | inf_mask, torch.zeros_like(data), data)
            elif method in ["median", "mean"]:
                # Convert to numpy for easier median/mean calculation
                data_np = data.cpu().numpy()
                data_np = sanitize_data(data_np, method=method, name=name)
                data = torch.from_numpy(data_np).to(data.device)
            else:
                raise ValueError(f"Unknown sanitization method: {method}")
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

    return data
