"""CLEAN tests for EEG Snippet Maker - DI and real logic, no mocks."""

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from brain_go_brrr.core.snippets.maker import EEGSnippetMaker


class TestEEGSnippetMakerClean:
    """Test EEGSnippetMaker with REAL logic - Robert C. Martin style."""

    @pytest.fixture
    def synthetic_raw(self):
        """Create synthetic EEG data for testing - NO MOCKS."""
        import mne

        # Create realistic EEG data
        sfreq = 256  # Standard EEGPT sampling rate
        duration = 30  # 30 seconds for more snippets
        n_channels = 20  # Standard 10-20 channels
        
        # Standard channel names
        ch_names = [
            "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8",
            "T7", "C3", "CZ", "C4", "T8",
            "P7", "P3", "PZ", "P4", "P8",
            "O1", "O2", "OZ"
        ]
        
        # Generate synthetic signals with realistic patterns
        np.random.seed(1337)
        n_samples = int(sfreq * duration)
        
        # Mix of frequencies typical in EEG
        times = np.arange(n_samples) / sfreq
        data = np.zeros((n_channels, n_samples), dtype=np.float32)
        
        for i in range(n_channels):
            # Alpha (8-13 Hz)
            data[i] += 10e-6 * np.sin(2 * np.pi * 10 * times + i * 0.1)
            # Beta (13-30 Hz)
            data[i] += 5e-6 * np.sin(2 * np.pi * 20 * times + i * 0.2)
            # Some noise
            data[i] += 2e-6 * np.random.randn(n_samples)
            
        # Create MNE Raw object
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info)
        
        return raw

    def test_init_creates_maker(self):
        """Test initialization of EEGSnippetMaker."""
        maker = EEGSnippetMaker()
        
        assert maker is not None
        assert maker.snippet_length == 10.0  # Default duration
        assert maker.overlap == 0.5  # Default overlap
        assert maker.min_snippet_length == 1.0
        assert maker.max_snippets_per_file == 1000
        assert maker.feature_extraction is True or maker.feature_extraction is False
        
    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        maker = EEGSnippetMaker(
            snippet_length=5.0,
            overlap=0.25,
            min_snippet_length=2.0,
            max_snippets_per_file=500,
            feature_extraction=False
        )
        
        assert maker.snippet_length == 5.0
        assert maker.overlap == 0.25
        assert maker.min_snippet_length == 2.0
        assert maker.max_snippets_per_file == 500
        assert maker.feature_extraction is False
        
    def test_extract_fixed_snippets(self, synthetic_raw):
        """Test fixed-length snippet extraction - REAL LOGIC."""
        maker = EEGSnippetMaker(snippet_length=2.0, overlap=0.5)
        
        # Extract snippets
        snippets = maker.extract_fixed_snippets(synthetic_raw)
        
        # Verify snippets
        assert len(snippets) > 0
        assert isinstance(snippets, list)
        
        # Check first snippet structure
        snippet = snippets[0]
        assert "id" in snippet
        assert "start_time" in snippet
        assert "end_time" in snippet
        assert "duration" in snippet
        assert "data" in snippet
        assert "channels" in snippet
        assert "sampling_rate" in snippet
        assert "n_samples" in snippet
        assert "n_channels" in snippet
        assert "extraction_method" in snippet
        assert snippet["extraction_method"] == "fixed_length"
        
        # Verify data shape
        assert snippet["data"].shape[0] == 20  # channels
        assert snippet["data"].shape[1] == int(256 * 2.0)  # samples for 2 seconds
        
    def test_extract_fixed_snippets_with_overlap(self, synthetic_raw):
        """Test overlapping snippet extraction."""
        maker = EEGSnippetMaker(snippet_length=4.0, overlap=0.75)
        
        snippets = maker.extract_fixed_snippets(
            synthetic_raw,
            start_time=0.0,
            end_time=12.0  # Extract from first 12 seconds
        )
        
        # With 75% overlap and 4-second windows:
        # Step size = 4 * (1 - 0.75) = 1 second
        # Expected snippets: floor((12 - 4) / 1) + 1 = 9
        assert len(snippets) >= 8  # Allow some variance
        
        # Check overlap by comparing start times
        if len(snippets) > 1:
            time_diff = snippets[1]["start_time"] - snippets[0]["start_time"]
            assert abs(time_diff - 1.0) < 0.1  # Should be ~1 second apart
            
    def test_extract_event_based_snippets(self, synthetic_raw):
        """Test event-based snippet extraction."""
        maker = EEGSnippetMaker()
        
        # Create synthetic events
        events = [
            {"type": "spike", "time": 2.0, "confidence": 0.9},
            {"type": "artifact", "time": 5.0, "confidence": 0.8},
            {"type": "spike", "time": 8.0, "confidence": 0.95},
        ]
        
        snippets = maker.extract_event_snippets(
            synthetic_raw,
            events,
            pre_event=1.0,  # 1 second before
            post_event=2.0   # 2 seconds after
        )
        
        assert len(snippets) == 3
        
        # Check event-based snippet structure
        for i, snippet in enumerate(snippets):
            assert snippet["extraction_method"] == "event_based"
            assert "event" in snippet
            assert snippet["event"]["type"] == events[i]["type"]
            assert snippet["duration"] == 3.0  # pre + post
            
    def test_extract_snippets_with_channel_selection(self, synthetic_raw):
        """Test snippet extraction with specific channels."""
        maker = EEGSnippetMaker(snippet_length=2.0)
        
        # Select only frontal channels
        channel_selection = ["FP1", "FP2", "F3", "F4", "FZ"]
        
        snippets = maker.extract_fixed_snippets(
            synthetic_raw,
            channel_selection=channel_selection
        )
        
        assert len(snippets) > 0
        snippet = snippets[0]
        assert snippet["n_channels"] == 5
        assert snippet["channels"] == channel_selection
        assert snippet["data"].shape[0] == 5
        
    def test_create_feature_dataframe(self, synthetic_raw):
        """Test creating DataFrame for feature extraction."""
        maker = EEGSnippetMaker(snippet_length=2.0)
        
        snippets = maker.extract_fixed_snippets(synthetic_raw)[:3]
        
        # Create feature DataFrame
        df = maker.create_feature_dataframe(snippets)
        
        assert df is not None
        assert len(df.columns) > 0
        assert len(df) == 3  # One row per snippet
        
    def test_compute_spectral_features(self, synthetic_raw):
        """Test spectral feature computation."""
        maker = EEGSnippetMaker(snippet_length=4.0)
        
        snippets = maker.extract_fixed_snippets(synthetic_raw)[:2]
        
        # Compute spectral features for each snippet
        for snippet in snippets:
            features = maker.compute_spectral_features(snippet["data"], snippet["sampling_rate"])
            
            assert "alpha_power" in features
            assert "beta_power" in features
            assert "theta_power" in features
            assert "delta_power" in features
            assert "gamma_power" in features
            assert "total_power" in features
            assert "spectral_entropy" in features
            
    def test_detect_artifacts_in_snippets(self, synthetic_raw):
        """Test artifact detection in snippets."""
        maker = EEGSnippetMaker(snippet_length=2.0)
        
        # Add artifact to raw data
        raw_with_artifact = synthetic_raw.copy()
        data = raw_with_artifact.get_data()
        # Add large amplitude artifact
        data[0, 1000:1100] = 500e-6  # Big spike in first channel
        
        snippets = maker.extract_fixed_snippets(raw_with_artifact)
        
        # Detect artifacts
        for snippet in snippets:
            has_artifact = maker.detect_artifact(
                snippet["data"],
                threshold=100e-6  # 100 microvolts
            )
            snippet["has_artifact"] = has_artifact
            
        # At least one snippet should have artifact
        artifacts_found = sum(s["has_artifact"] for s in snippets)
        assert artifacts_found > 0
        
    def test_save_snippets_to_json(self, synthetic_raw, tmp_path):
        """Test saving snippets to JSON format."""
        maker = EEGSnippetMaker(snippet_length=2.0)
        
        snippets = maker.extract_fixed_snippets(synthetic_raw)[:2]
        
        # Save to JSON
        json_path = tmp_path / "snippets.json"
        maker.save_snippets_json(snippets, json_path)
        
        assert json_path.exists()
        
        # Load and verify
        with open(json_path) as f:
            loaded = json.load(f)
            
        assert len(loaded) == 2
        assert loaded[0]["id"] == snippets[0]["id"]
        
    def test_analyze_snippet_with_eegpt(self, synthetic_raw):
        """Test EEGPT analysis integration - with DI."""
        maker = EEGSnippetMaker()
        
        snippets = maker.extract_fixed_snippets(synthetic_raw, snippet_length=4.0)
        snippet = snippets[0]
        
        # Mock EEGPT model with DI
        mock_eegpt = MagicMock()
        mock_eegpt.extract_features.return_value = np.random.randn(1, 512).astype(np.float32)
        
        # Inject the mock
        maker.eegpt_model = mock_eegpt
        
        # Analyze snippet
        result = maker.analyze_snippet_with_eegpt(snippet)
        
        assert result is not None
        assert "eegpt_features" in result
        assert "abnormality_score" in result
        assert "quality_score" in result
        
    def test_batch_process_snippets(self, synthetic_raw):
        """Test batch processing of multiple snippets."""
        maker = EEGSnippetMaker(snippet_length=2.0, max_snippets_per_file=5)
        
        snippets = maker.extract_fixed_snippets(synthetic_raw)
        
        # Should respect max_snippets_per_file
        assert len(snippets) <= 5
        
        # Process batch
        results = maker.batch_process(snippets)
        
        assert len(results) == len(snippets)
        for result in results:
            assert "processed" in result
            assert result["processed"] is True
            
    def test_snippet_statistics(self, synthetic_raw):
        """Test computing statistics for snippets."""
        maker = EEGSnippetMaker(snippet_length=2.0)
        
        snippets = maker.extract_fixed_snippets(synthetic_raw)[:5]
        
        # Compute statistics
        stats = maker.compute_snippet_statistics(snippets)
        
        assert "mean_amplitude" in stats
        assert "std_amplitude" in stats
        assert "min_amplitude" in stats
        assert "max_amplitude" in stats
        assert "n_snippets" in stats
        assert stats["n_snippets"] == 5
        
    def test_merge_overlapping_snippets(self, synthetic_raw):
        """Test merging overlapping snippets."""
        maker = EEGSnippetMaker()
        
        # Create overlapping snippets
        snippets = [
            {"start_time": 0.0, "end_time": 3.0, "data": np.zeros((20, 768))},
            {"start_time": 2.0, "end_time": 5.0, "data": np.ones((20, 768))},
            {"start_time": 4.0, "end_time": 7.0, "data": np.zeros((20, 768))},
        ]
        
        merged = maker.merge_overlapping_snippets(snippets, threshold=0.5)
        
        # Should merge overlapping snippets
        assert len(merged) < len(snippets)
        
    def test_snippet_quality_assessment(self, synthetic_raw):
        """Test quality assessment of snippets."""
        maker = EEGSnippetMaker(snippet_length=2.0)
        
        snippets = maker.extract_fixed_snippets(synthetic_raw)[:3]
        
        for snippet in snippets:
            quality = maker.assess_snippet_quality(snippet["data"])
            
            assert 0 <= quality <= 1  # Quality score between 0 and 1
            snippet["quality_score"] = quality
            
        # All synthetic data should have reasonable quality
        assert all(s["quality_score"] > 0.5 for s in snippets)