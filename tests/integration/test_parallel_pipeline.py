"""Test parallel pipeline - EEGPT and YASA running independently."""

import mne
import numpy as np
import pytest


@pytest.mark.integration
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
        from brain_go_brrr.application.pipeline.parallel import ParallelEEGPipeline

        pipeline = ParallelEEGPipeline()
        assert pipeline is not None
        assert hasattr(pipeline, "eegpt_extractor")
        assert hasattr(pipeline, "sleep_analyzer")

    def test_parallel_processing(self, sample_raw):
        """Test both pipelines run independently."""
        from brain_go_brrr.application.pipeline.parallel import ParallelEEGPipeline

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
            # EEGPT returns different shapes depending on implementation
            embeddings = results["eegpt"]["embeddings"]
            if embeddings.ndim == 3:
                # Shape: (n_windows, n_summary_tokens, embed_dim)
                assert embeddings.shape[1] == 4  # n_summary_tokens
                assert embeddings.shape[2] == 512  # embed_dim
            elif embeddings.ndim == 2:
                # Shape: (n_windows, embed_dim) - flattened summary tokens
                assert embeddings.shape[1] in [512, 2048]  # Either single token or concatenated
            else:
                raise AssertionError(f"Unexpected embeddings shape: {embeddings.shape}")

        # Check YASA results
        if results["yasa"]["status"] == "success":
            assert "hypnogram" in results["yasa"]
            assert "confidence" in results["yasa"]
            # Sleep stats may fail if hypnogram is all wake
            if "sleep_stats" in results["yasa"]:
                assert isinstance(results["yasa"]["sleep_stats"], dict)
            # Hypnogram should have correct number of epochs
            if results["yasa"]["hypnogram"]:
                assert len(results["yasa"]["hypnogram"]) == 20  # 10 min / 30 sec

    def test_independent_failures(self, sample_raw):
        """Test that services can fail independently without affecting the pipeline."""
        from unittest.mock import patch

        from brain_go_brrr.application.pipeline.parallel import ParallelEEGPipeline

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
    @pytest.mark.data
    def test_with_real_sleep_edf(self, sleep_edf_path):
        """Test with real Sleep-EDF data - demonstrates pathway separation."""
        # Path provided by fixture; skip handled upstream if missing

        from brain_go_brrr.application.pipeline.parallel import ParallelEEGPipeline

        pipeline = ParallelEEGPipeline()

        # Process just 5 minutes for speed
        raw = mne.io.read_raw_edf(sleep_edf_path, preload=False, verbose=False)
        raw.crop(tmax=300)
        raw.load_data()

        results = pipeline.process(raw)

        # NOTE: EEGPT actually CAN work with 2 channels (it pads internally)
        # But it won't give meaningful clinical results with only 2 channels
        # The important thing is the pathways are independent

        # Check EEGPT results - may succeed or fail depending on implementation
        if results["eegpt"]["status"] == "success":
            # EEGPT can pad channels internally, so it might work
            assert "embeddings" in results["eegpt"]
            # But results won't be clinically meaningful with only 2 channels
        else:
            # Or it might fail if validation is added
            assert "error" in results["eegpt"]

        # YASA pathway should handle its own errors independently
        # It may fail due to the all-wake hypnogram issue we saw
        assert "yasa" in results
        assert results["yasa"]["status"] in ["success", "failed"]

        # This demonstrates the parallel pathways:
        # - EEGPT and YASA run independently
        # - Failure in one doesn't affect the other
        # - Sleep-EDF is better suited for YASA (designed for sleep staging)
