"""Example inference script for EEGPT Linear Probe.

Shows how to load a trained probe and make predictions.
"""

import sys
from pathlib import Path
from typing import Any

import mne
import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from brain_go_brrr.tasks.abnormality_detection import AbnormalityDetectionProbe  # noqa: E402


def load_trained_probe(
    eegpt_checkpoint: Path, probe_checkpoint: Path, device: str = "cuda"
) -> AbnormalityDetectionProbe:
    """Load a trained linear probe.

    Args:
        eegpt_checkpoint: Path to EEGPT pretrained weights
        probe_checkpoint: Path to trained probe weights
        device: Device to load model on

    Returns:
        Loaded probe model
    """
    # Initialize model
    model = AbnormalityDetectionProbe(checkpoint_path=eegpt_checkpoint)

    # Load probe weights
    model.load_probe(probe_checkpoint)

    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()

    return model


def predict_eeg_file(
    model: AbnormalityDetectionProbe,
    edf_path: Path,
    window_duration: float = 30.0,
    sampling_rate: int = 256,
    device: str = "cuda",
) -> dict[str, Any]:
    """Make predictions on an EEG file.

    Args:
        model: Trained probe model
        edf_path: Path to EDF file
        window_duration: Window duration in seconds
        sampling_rate: Target sampling rate
        device: Device for inference

    Returns:
        Dictionary with predictions and probabilities
    """
    # Load EEG data
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    # Basic preprocessing
    raw.filter(0.5, 50.0, fir_design="firwin", verbose=False)
    raw.notch_filter(60.0, fir_design="firwin", verbose=False)

    # Resample if needed
    if raw.info["sfreq"] != sampling_rate:
        raw.resample(sampling_rate)

    # Get required channels
    required_channels = AbnormalityDetectionProbe.TUAB_CHANNELS
    available_channels = [ch for ch in required_channels if ch in raw.ch_names]
    raw.pick_channels(available_channels, ordered=True)

    # Get data
    data = raw.get_data()

    # Pad channels if needed
    if data.shape[0] < len(required_channels):
        padding = np.zeros((len(required_channels) - data.shape[0], data.shape[1]))
        data = np.vstack([data, padding])

    # Create windows
    window_samples = int(window_duration * sampling_rate)
    n_windows = int(data.shape[1] / window_samples)

    predictions = []
    probabilities = []
    confidences = []

    with torch.no_grad():
        for i in range(n_windows):
            # Extract window
            start = i * window_samples
            end = start + window_samples

            if end > data.shape[1]:
                break

            window = data[:, start:end]

            # Convert to tensor and add batch dimension
            window_tensor = torch.from_numpy(window).float().unsqueeze(0).to(device)

            # Get predictions
            results = model.predict_with_confidence(window_tensor)

            predictions.append(results["predictions"].cpu().item())
            probabilities.append(results["probabilities"].cpu().item())
            confidences.append(results["confidence"].cpu().item())

    # Aggregate results
    avg_abnormal_prob = np.mean(probabilities)
    overall_prediction = "abnormal" if avg_abnormal_prob > 0.5 else "normal"

    return {
        "file": edf_path.name,
        "n_windows": len(predictions),
        "window_predictions": predictions,
        "window_probabilities": probabilities,
        "window_confidences": confidences,
        "average_abnormal_probability": avg_abnormal_prob,
        "overall_prediction": overall_prediction,
        "confident_abnormal_windows": sum(p > 0.7 for p in probabilities),
        "confident_normal_windows": sum(p < 0.3 for p in probabilities),
    }


def main():
    """Example usage."""
    # Paths
    eegpt_checkpoint = Path("data/models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt")
    probe_checkpoint = Path("experiments/eegpt_linear_probe/checkpoints/tuab_probe_best.pth")

    # Example EDF file
    edf_file = Path("data/datasets/external/tuh_eeg_abnormal/v3.0.0/test/abnormal/example.edf")

    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model = load_trained_probe(eegpt_checkpoint, probe_checkpoint, device)
    print(f"Model has {model.get_num_trainable_params():,} trainable parameters")

    # Make predictions
    print(f"\nAnalyzing {edf_file.name}...")
    results = predict_eeg_file(model, edf_file, device=device)

    # Print results
    print("\n=== RESULTS ===")
    print(f"File: {results['file']}")
    print(f"Windows analyzed: {results['n_windows']}")
    print(f"Overall prediction: {results['overall_prediction'].upper()}")
    print(f"Average abnormal probability: {results['average_abnormal_probability']:.3f}")
    print(f"Confident abnormal windows: {results['confident_abnormal_windows']}")
    print(f"Confident normal windows: {results['confident_normal_windows']}")

    # Window-by-window results
    print("\nWindow-by-window predictions:")
    for i, (pred, prob, conf) in enumerate(
        zip(
            results["window_predictions"],
            results["window_probabilities"],
            results["window_confidences"],
            strict=False,
        )
    ):
        label = "ABNORMAL" if pred == 1 else "NORMAL"
        print(f"  Window {i + 1}: {label} (prob={prob:.3f}, confidence={conf:.3f})")


if __name__ == "__main__":
    main()
