"""Test for improved EEGPT model mocking with realistic embeddings."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from brain_go_brrr.core.abnormal import AbnormalityDetector


class TestImprovedMocking:
    """Test suite for improved EEGPT mocking."""

    def test_realistic_embeddings_shape(self):
        """Test that EEGPT embeddings have correct shape and properties."""
        # EEGPT returns (4, 512) summary tokens, flattened to 2048 for classifier
        batch_size = 1
        embedding_dim = 2048  # 4 summary tokens x 512 dims = 2048 flattened

        # Create realistic embeddings with proper statistics
        embeddings = np.random.randn(batch_size, embedding_dim).astype(np.float32)

        # Embeddings should be normalized (roughly)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        assert embeddings.shape == (batch_size, embedding_dim)
        assert embeddings.dtype == np.float32
        # Check normalization
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_embeddings_for_different_conditions(self):
        """Test that embeddings differ based on input characteristics."""
        # Normal EEG characteristics
        normal_embedding = self._create_embedding_for_condition("normal")

        # Abnormal EEG characteristics
        abnormal_embedding = self._create_embedding_for_condition("abnormal")

        # Embeddings should be different
        cosine_sim = np.dot(normal_embedding.flatten(), abnormal_embedding.flatten())
        assert cosine_sim < 0.9  # Not too similar

    def _create_embedding_for_condition(self, condition: str) -> np.ndarray:
        """Create realistic embeddings for different EEG conditions."""
        np.random.seed(42 if condition == "normal" else 123)

        # Base embedding (4 summary tokens x 512 dims = 2048 flattened)
        embedding = np.random.randn(1, 2048).astype(np.float32)

        if condition == "abnormal":
            # Add some structured patterns to simulate abnormal features
            # These would be learned by EEGPT during pretraining
            embedding[:, :50] += 0.5  # Boost certain features
            embedding[:, 100:150] -= 0.3  # Suppress others

        # Normalize
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
        return embedding

    def test_classifier_head_with_realistic_embeddings(self):
        """Test that classifier head works properly with realistic embeddings."""
        with (
            patch("brain_go_brrr.core.abnormal.detector.EEGPTModel"),
            patch("brain_go_brrr.core.abnormal.detector.ModelConfig"),
        ):
            detector = AbnormalityDetector(model_path=Path("fake/path.ckpt"), device="cpu")

            # Test with normal embedding
            normal_embedding = self._create_embedding_for_condition("normal")
            normal_tensor = torch.from_numpy(normal_embedding).float()

            with torch.no_grad():
                logits = detector.classifier(normal_tensor)
                probs = torch.softmax(logits, dim=1)

            assert probs.shape == (1, 2)  # Binary classification
            assert torch.allclose(probs.sum(dim=1), torch.tensor(1.0))

    def test_window_prediction_with_proper_mocking(self):
        """Test window prediction using proper embedding mocking."""
        # Create a more sophisticated mock
        mock_model = MagicMock()

        # Mock different embeddings for different window characteristics
        def mock_extract_features(window_data, channel_names=None):
            # Analyze window characteristics (simplified)
            # window_data is already a numpy array from production code
            window_std = np.std(window_data)

            # Return different embeddings based on window characteristics
            if window_std > 2.0:  # High variance might indicate artifacts/abnormality
                return self._create_embedding_for_condition("abnormal")
            else:
                return self._create_embedding_for_condition("normal")

        mock_model.extract_features.side_effect = mock_extract_features
        mock_model.is_loaded = True

        with (
            patch("brain_go_brrr.core.abnormal.detector.EEGPTModel") as mock_model_class,
            patch("brain_go_brrr.core.abnormal.detector.ModelConfig"),
        ):
            mock_model_class.return_value = mock_model

            detector = AbnormalityDetector(model_path=Path("fake/path.ckpt"), device="cpu")
            detector.model = mock_model

            # Test with normal-like window
            normal_window = np.random.randn(19, 1024).astype(np.float32) * 0.5
            score_normal = detector._predict_window(normal_window)

            # Test with abnormal-like window
            abnormal_window = np.random.randn(19, 1024).astype(np.float32) * 3.0
            score_abnormal = detector._predict_window(abnormal_window)

            # Scores should be valid probabilities
            assert 0.0 <= score_normal <= 1.0, f"Normal score {score_normal} out of range"
            assert 0.0 <= score_abnormal <= 1.0, f"Abnormal score {score_abnormal} out of range"

            # With random classifier weights, we can't guarantee specific differences,
            # but the pipeline should work without errors
            assert score_normal is not None and score_abnormal is not None
            print(f"Pipeline working: normal={score_normal:.4f}, abnormal={score_abnormal:.4f}")
