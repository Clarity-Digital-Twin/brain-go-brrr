"""Test parallel pipeline - EEGPT and YASA running independently."""

from pathlib import Path

import mne
import numpy as np
import pytest


class TestParallelPipeline:
    """Test that EEGPT and YASA work independently in parallel."""

    @pytest.fixture
    def sample_raw(self):
        """Create sample EEG data."""
        sfreq = 256
        duration = 600  # 10 minutes
        n_channels = 19
        ch_names = [
            "Fp1",
            "Fp2",
            "F3",
            "F4",
            "C3",
            "C4",
            "P3",
            "P4",
            "O1",
            "O2",
            "F7",
            "F8",
            "T3",
            "T4",
            "T5",
            "T6",
            "Fz",
            "Cz",
            "Pz",
        ]

        data = np.random.randn(n_channels, int(sfreq * duration)) * 50e-6
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        return mne.io.RawArray(data, info)

    def test_parallel_pipeline_initialization(self):
        """Test pipeline can be initialized."""
        from brain_go_brrr.core.pipeline import ParallelEEGPipeline

        pipeline = ParallelEEGPipeline()
        assert pipeline is not None
        assert hasattr(pipeline, "eegpt_extractor")
        assert hasattr(pipeline, "sleep_analyzer")

    def test_parallel_processing(self, sample_raw):
        """Test both pipelines run independently."""
        from brain_go_brrr.core.pipeline import ParallelEEGPipeline

        pipeline = ParallelEEGPipeline()
        results = pipeline.process(sample_raw)

        # Check structure
        assert "eegpt" in results
        assert "yasa" in results
        assert "metadata" in results

        # Check EEGPT results
        if results["eegpt"]["status"] == "success":
            assert "embeddings" in results["eegpt"]
            assert "window_times" in results["eegpt"]
            assert results["eegpt"]["embeddings"].shape[1] == 512

        # Check YASA results
        if results["yasa"]["status"] == "success":
            assert "hypnogram" in results["yasa"]
            assert "confidence" in results["yasa"]
            assert "sleep_stats" in results["yasa"]
            assert len(results["yasa"]["hypnogram"]) == 20  # 10 min / 30 sec

    def test_independent_failures(self, sample_raw):
        """Test that services can fail independently without affecting the pipeline."""
        from unittest.mock import patch

        from brain_go_brrr.core.pipeline import ParallelEEGPipeline

        pipeline = ParallelEEGPipeline()

        # Mock only EEGPT to fail, YASA should still work
        with patch.object(
            pipeline.eegpt_extractor,
            "extract_embeddings_with_metadata",
            side_effect=Exception("EEGPT failed"),
        ):
            results = pipeline.process(sample_raw)

            # EEGPT should fail
            assert results["eegpt"]["status"] == "failed"
            assert "EEGPT failed" in results["eegpt"]["error"]

            # YASA may succeed or fail depending on the data
            # The important thing is that EEGPT failure doesn't crash the whole pipeline
            assert "yasa" in results
            assert results["yasa"]["status"] in ["success", "failed"]

    @pytest.mark.integration
    def test_with_real_sleep_edf(self):
        """Test with real Sleep-EDF data."""
        sleep_edf_path = Path("data/datasets/external/sleep-edf/sleep-cassette/SC4001E0-PSG.edf")

        if not sleep_edf_path.exists():
            pytest.skip("Sleep-EDF data not found")

        from brain_go_brrr.core.pipeline import ParallelEEGPipeline

        pipeline = ParallelEEGPipeline()

        # Process just 5 minutes for speed
        raw = mne.io.read_raw_edf(sleep_edf_path, preload=False, verbose=False)
        raw.crop(tmax=300)
        raw.load_data()

        results = pipeline.process(raw)

        # Both should return results
        assert results["eegpt"]["status"] in ["success", "failed"]
        assert results["yasa"]["status"] in ["success", "failed"]

        # At least one should succeed
        assert results["eegpt"]["status"] == "success" or results["yasa"]["status"] == "success"
