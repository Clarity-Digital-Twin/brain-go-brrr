"""Simple but improved mock for EEGPT that works with existing tests."""

from unittest.mock import MagicMock

import numpy as np
import torch


def create_improved_mock_eegpt():
    """Create an improved mock that returns proper embeddings instead of predictions."""
    mock = MagicMock()
    mock.is_loaded = True

    # Counter to track calls and return varied embeddings
    mock._call_count = 0

    def mock_extract_features(window_tensor):
        """Return proper 512-dimensional embeddings based on window characteristics."""
        # Increment call count
        mock._call_count += 1

        # Create base embedding
        embedding = np.random.randn(1, 512).astype(np.float32)

        # Analyze window to create meaningful embeddings
        window_np = window_tensor.cpu().numpy()

        # Calculate some basic statistics
        window_std = np.std(window_np)
        window_max = np.max(np.abs(window_np))

        # Modify embedding based on window characteristics
        if window_std > 2.0 or window_max > 5.0:
            # High variance/amplitude - make embedding indicate abnormality
            embedding[:, :100] += 1.0  # Boost first 100 features
            embedding[:, 200:300] -= 0.5  # Suppress middle features
        else:
            # Normal variance - make embedding indicate normal
            embedding[:, 100:200] += 0.5  # Different pattern
            embedding[:, 300:400] += 0.3

        # Add variation based on call count to ensure different windows get different embeddings
        embedding += np.sin(mock._call_count * 0.1) * 0.1

        # Normalize embedding (EEGPT typically produces normalized embeddings)
        embedding = embedding / (np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-8)

        return embedding

    mock.extract_features = mock_extract_features
    return mock


def create_trained_classifier_weights():
    """Create classifier weights that produce meaningful predictions from embeddings."""
    # This simulates a trained classifier that can distinguish embedding patterns

    # Create a simple linear classifier that maps certain embedding patterns to abnormal
    input_dim = 512
    hidden_dim1 = 256
    hidden_dim2 = 128
    output_dim = 2

    weights = {}

    # First layer weights - designed to detect patterns in embeddings
    w1 = np.random.randn(hidden_dim1, input_dim).astype(np.float32) * 0.01
    # Make first 50 neurons sensitive to the "abnormal" pattern (high values in first 100 dims)
    w1[:50, :100] = 0.05
    # Make next 50 sensitive to "normal" pattern
    w1[50:100, 100:200] = 0.05
    weights["0.weight"] = torch.from_numpy(w1)
    weights["0.bias"] = torch.zeros(hidden_dim1)

    # Second layer weights
    w2 = np.random.randn(hidden_dim2, hidden_dim1).astype(np.float32) * 0.01
    weights["4.weight"] = torch.from_numpy(w2)
    weights["4.bias"] = torch.zeros(hidden_dim2)

    # Output layer - map to class predictions
    w3 = np.zeros((output_dim, hidden_dim2), dtype=np.float32)
    # Normal class (index 0) - activated by "normal" pattern neurons
    w3[0, 25:50] = 0.1
    # Abnormal class (index 1) - activated by "abnormal" pattern neurons
    w3[1, :25] = 0.1
    weights["8.weight"] = torch.from_numpy(w3)
    weights["8.bias"] = torch.tensor([0.1, -0.1])  # Slight bias toward normal

    return weights


def update_existing_tests_mock(detector, mock_model):
    """Update detector to use improved mocking without changing test structure."""
    # Replace the mock's extract_features with our improved version
    improved_mock = create_improved_mock_eegpt()
    mock_model.extract_features = improved_mock.extract_features
    mock_model.is_loaded = True

    # Update classifier weights to produce meaningful predictions
    weights = create_trained_classifier_weights()

    # Apply weights to classifier (carefully, as the state dict keys might differ)
    try:
        # Get current state dict to match keys
        current_state = detector.classifier.state_dict()

        # Map our weights to the actual keys
        for key in current_state:
            if "weight" in key:
                layer_idx = int(key.split(".")[0])
                if layer_idx == 0 and "0.weight" in weights:
                    current_state[key] = weights["0.weight"]
                elif layer_idx == 4 and "4.weight" in weights:
                    current_state[key] = weights["4.weight"]
                elif layer_idx == 8 and "8.weight" in weights:
                    current_state[key] = weights["8.weight"]
            elif "bias" in key:
                layer_idx = int(key.split(".")[0])
                if layer_idx == 0 and "0.bias" in weights:
                    current_state[key] = weights["0.bias"]
                elif layer_idx == 4 and "4.bias" in weights:
                    current_state[key] = weights["4.bias"]
                elif layer_idx == 8 and "8.bias" in weights:
                    current_state[key] = weights["8.bias"]

        detector.classifier.load_state_dict(current_state, strict=False)
    except Exception:
        # If weight loading fails, at least we have better embeddings
        pass

    return detector
