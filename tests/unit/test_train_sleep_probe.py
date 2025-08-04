"""Test training script for sleep probe."""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from brain_go_brrr.models.linear_probe import SleepStageProbe
from brain_go_brrr.training.sleep_probe_trainer import (
    SleepDataset,
    SleepProbeTrainer,
    evaluate_probe,
    train_sleep_probe,
)


class TestSleepProbeTraining:
    """Test sleep probe training functionality."""

    @pytest.fixture
    def mock_eegpt_model(self):
        """Mock EEGPT model for feature extraction."""
        mock = Mock()
        # Return consistent features
        mock.extract_features.return_value = np.random.randn(4, 512)
        mock.is_loaded = True
        return mock

    @pytest.fixture
    def mock_sleep_data(self):
        """Create mock sleep EEG data."""
        # Simulate 10 windows of data
        n_windows = 10
        n_channels = 19
        window_samples = 1024  # 4 seconds at 256 Hz

        windows = []
        labels = []

        for i in range(n_windows):
            # Create synthetic EEG data
            window = np.random.randn(n_channels, window_samples) * 10
            windows.append(window)

            # Cycle through sleep stages
            label = i % 5  # 0=W, 1=N1, 2=N2, 3=N3, 4=REM
            labels.append(label)

        return windows, labels

    def test_sleep_dataset_creation(self, mock_sleep_data):
        """Test SleepDataset can be created with windows and labels."""
        windows, labels = mock_sleep_data

        dataset = SleepDataset(windows, labels)

        assert len(dataset) == len(windows)

        # Test indexing
        window, label = dataset[0]
        assert isinstance(window, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert window.shape == (19, 1024)
        assert label.item() == 0

    @pytest.mark.slow
    def test_sleep_probe_trainer_init(self, mock_eegpt_model):
        """Test SleepProbeTrainer initialization."""
        probe = SleepStageProbe()
        trainer = SleepProbeTrainer(
            probe=probe,
            eegpt_model=mock_eegpt_model,
            learning_rate=1e-3,
            batch_size=4,
        )

        assert trainer.probe == probe
        assert trainer.eegpt_model == mock_eegpt_model
        assert trainer.batch_size == 4
        assert isinstance(trainer.optimizer, torch.optim.Adam)
        assert isinstance(trainer.criterion, torch.nn.CrossEntropyLoss)

    def test_training_step(self, mock_eegpt_model, mock_sleep_data):
        """Test a single training step."""
        windows, labels = mock_sleep_data
        dataset = SleepDataset(windows, labels)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        probe = SleepStageProbe()
        trainer = SleepProbeTrainer(probe, mock_eegpt_model)

        # Get a batch
        batch_windows, batch_labels = next(iter(dataloader))

        # Perform training step
        loss = trainer.train_step(batch_windows, batch_labels)

        assert isinstance(loss, float)
        assert loss > 0  # Should have some loss initially

    def test_full_training_epoch(self, mock_eegpt_model, mock_sleep_data):
        """Test a full training epoch."""
        windows, labels = mock_sleep_data
        dataset = SleepDataset(windows, labels)

        probe = SleepStageProbe()
        trainer = SleepProbeTrainer(probe, mock_eegpt_model, batch_size=2)

        # Train one epoch
        avg_loss = trainer.train_epoch(dataset)

        assert isinstance(avg_loss, float)
        assert avg_loss > 0

    def test_evaluation(self, mock_eegpt_model, mock_sleep_data):
        """Test probe evaluation."""
        windows, labels = mock_sleep_data
        dataset = SleepDataset(windows, labels)

        probe = SleepStageProbe()

        # Evaluate untrained probe
        accuracy, confusion_matrix = evaluate_probe(probe, mock_eegpt_model, dataset)

        assert 0 <= accuracy <= 1
        assert confusion_matrix.shape == (5, 5)
        assert confusion_matrix.sum() == len(dataset)

    def test_train_sleep_probe_integration(self, mock_eegpt_model, tmp_path):
        """Test full training pipeline."""
        # Create mock Sleep-EDF data directory
        data_dir = tmp_path / "sleep-edf"
        data_dir.mkdir()

        # Create a mock EDF file
        edf_file = data_dir / "SC4001E0-PSG.edf"
        edf_file.write_bytes(b"mock edf data")

        # Mock the EDF loading
        with patch("brain_go_brrr.training.sleep_probe_trainer.load_sleep_edf_data") as mock_load:
            # Return mock windows and labels
            mock_load.return_value = (
                [np.random.randn(19, 1024) for _ in range(100)],
                [i % 5 for i in range(100)],
            )

            # Train the probe
            trained_probe, metrics = train_sleep_probe(
                data_dir=data_dir,
                eegpt_model=mock_eegpt_model,
                num_epochs=2,
                batch_size=16,
                learning_rate=1e-3,
                validation_split=0.2,
            )

            assert isinstance(trained_probe, SleepStageProbe)
            assert "train_accuracy" in metrics
            assert "val_accuracy" in metrics
            assert "train_losses" in metrics
            assert len(metrics["train_losses"]) == 2  # 2 epochs

    def test_checkpoint_saving(self, mock_eegpt_model, mock_sleep_data, tmp_path):
        """Test that training saves checkpoints."""
        windows, labels = mock_sleep_data

        probe = SleepStageProbe()
        trainer = SleepProbeTrainer(probe, mock_eegpt_model)

        # Train and save
        checkpoint_path = tmp_path / "sleep_probe.pt"
        trainer.save_checkpoint(checkpoint_path, epoch=1, accuracy=0.85)

        assert checkpoint_path.exists()

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert checkpoint["epoch"] == 1
        assert checkpoint["accuracy"] == 0.85

    def test_early_stopping(self, mock_eegpt_model):
        """Test early stopping during training."""
        # Create small dataset for quick training
        windows = [np.random.randn(19, 1024) for _ in range(20)]
        labels = [0] * 20  # All same class for easy learning

        dataset = SleepDataset(windows, labels)

        probe = SleepStageProbe()
        trainer = SleepProbeTrainer(probe, mock_eegpt_model, batch_size=5)

        losses = []
        prev_loss = float("inf")
        patience = 3
        no_improve_count = 0

        for _ in range(10):
            loss = trainer.train_epoch(dataset)
            losses.append(loss)

            # Simulate early stopping logic
            if loss >= prev_loss:
                no_improve_count += 1
            else:
                no_improve_count = 0
                prev_loss = loss

            # Stop if no improvement for patience epochs
            if no_improve_count >= patience:
                break

        # Should have some losses recorded
        assert len(losses) >= 1
        # Loss should be reasonable (not NaN or inf)
        assert all(0 <= loss_val < 10 for loss_val in losses)

    def test_data_augmentation(self, mock_sleep_data):
        """Test data augmentation in dataset."""
        windows, labels = mock_sleep_data

        # Create dataset with augmentation
        dataset = SleepDataset(windows, labels, augment=True)

        # Get same window multiple times
        window1, _ = dataset[0]
        window2, _ = dataset[0]

        # Should have some noise added (not exactly equal)
        assert not torch.allclose(window1, window2, atol=1e-6)


class TestSleepEDFDataLoading:
    """Test Sleep-EDF dataset loading utilities."""

    def test_load_sleep_edf_annotations(self, tmp_path):
        """Test loading sleep stage annotations."""
        # Create mock annotation file
        annotation_file = tmp_path / "SC4001E0-PSG-annotations.txt"
        annotation_content = """
Sleep stage W: 0-300
Sleep stage N1: 300-600
Sleep stage N2: 600-1800
Sleep stage N3: 1800-2400
Sleep stage REM: 2400-3000
"""
        annotation_file.write_text(annotation_content)

        with patch(
            "brain_go_brrr.training.sleep_probe_trainer.parse_sleep_annotations"
        ) as mock_parse:
            mock_parse.return_value = {
                0: "W",
                1: "N1",
                2: "N2",
                3: "N2",
                4: "N2",
                5: "N2",
                6: "N3",
                7: "N3",
                8: "REM",
                9: "REM",
            }

            from brain_go_brrr.training.sleep_probe_trainer import load_sleep_annotations

            annotations = load_sleep_annotations(annotation_file)

            assert len(annotations) == 10
            assert annotations[0] == "W"
            assert annotations[2] == "N2"
            assert annotations[8] == "REM"

    def test_window_extraction_from_edf(self):
        """Test extracting 4-second windows from EDF data."""
        from brain_go_brrr.training.sleep_probe_trainer import extract_windows_from_raw

        # Create mock Raw object
        mock_raw = Mock()
        mock_raw.get_data.return_value = np.random.randn(19, 256 * 60)  # 60 seconds
        mock_raw.info = {"sfreq": 256}

        windows = extract_windows_from_raw(mock_raw, window_duration=4.0)

        # Should have 14 windows (60s / 4s = 15, but last partial window is excluded)
        assert len(windows) == 14
        assert windows[0].shape == (19, 1024)  # 4s at 256Hz

    def test_train_val_split(self):
        """Test train/validation split functionality."""
        # Create mock data
        n_windows = 10
        n_channels = 19
        window_samples = 1024

        windows = []
        labels = []

        for i in range(n_windows):
            window = np.random.randn(n_channels, window_samples) * 10
            windows.append(window)
            label = i % 5  # 0=W, 1=N1, 2=N2, 3=N3, 4=REM
            labels.append(label)

        from brain_go_brrr.training.sleep_probe_trainer import split_train_val

        train_data, val_data = split_train_val(windows, labels, val_split=0.2, random_seed=42)

        train_windows, train_labels = train_data
        val_windows, val_labels = val_data

        assert len(train_windows) == 8  # 80% of 10
        assert len(val_windows) == 2  # 20% of 10
        assert len(train_labels) == len(train_windows)
        assert len(val_labels) == len(val_windows)
