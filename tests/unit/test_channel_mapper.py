"""Test-driven development for channel mapping and validation.
Written BEFORE implementation - pure TDD approach.
"""


class TestChannelMapper:
    """Test channel name mapping from old to modern convention."""

    def test_maps_old_to_modern_naming(self):
        """Old naming (T3,T4,T5,T6) should map to modern (T7,T8,P7,P8)."""
        # Given
        old_channels = ["FP1", "FP2", "T3", "T4", "T5", "T6", "O1", "O2"]

        # When
        from brain_go_brrr.core.channels import ChannelMapper
        mapper = ChannelMapper()
        modern_channels = mapper.standardize_channel_names(old_channels)

        # Then
        assert modern_channels == ["Fp1", "Fp2", "T7", "T8", "P7", "P8", "O1", "O2"]

    def test_handles_mixed_case_consistently(self):
        """Channel names should be case-normalized."""
        # Given
        mixed_case = ["fp1", "FP2", "Fz", "FZ", "cz", "CZ"]

        # When
        from brain_go_brrr.core.channels import ChannelMapper
        mapper = ChannelMapper()
        normalized = mapper.standardize_channel_names(mixed_case)

        # Then
        assert normalized == ["Fp1", "Fp2", "Fz", "Fz", "Cz", "Cz"]

    def test_preserves_already_modern_names(self):
        """Modern channel names should pass through unchanged."""
        # Given
        modern_channels = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"]

        # When
        from brain_go_brrr.core.channels import ChannelMapper
        mapper = ChannelMapper()
        result = mapper.standardize_channel_names(modern_channels)

        # Then
        assert result == modern_channels

    def test_returns_empty_list_for_empty_input(self):
        """Empty input should return empty output."""
        # Given
        empty_channels = []

        # When
        from brain_go_brrr.core.channels import ChannelMapper
        mapper = ChannelMapper()
        result = mapper.standardize_channel_names(empty_channels)

        # Then
        assert result == []


class TestChannelValidator:
    """Test channel validation for minimum requirements."""

    def test_validates_minimum_channel_count(self):
        """Should require at least 19 standard channels."""
        # Given
        insufficient_channels = ["Fp1", "Fp2", "F7", "F3", "Fz"]  # Only 5

        # When
        from brain_go_brrr.core.channels import ChannelValidator
        validator = ChannelValidator()
        is_valid, missing = validator.validate_channels(insufficient_channels)

        # Then
        assert is_valid is False
        assert len(missing) > 0

    def test_identifies_missing_required_channels(self):
        """Should identify which required channels are missing."""
        # Given - missing C3, C4, and others
        partial_channels = [
            "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
            "T7", "Cz", "T8",  # Missing C3, C4
            "P7", "P3", "Pz", "P4", "P8",
            "O1", "O2"
        ]

        # When
        from brain_go_brrr.core.channels import ChannelValidator
        validator = ChannelValidator()
        is_valid, missing = validator.validate_channels(partial_channels)

        # Then
        assert is_valid is False
        assert "C3" in missing
        assert "C4" in missing

    def test_accepts_complete_channel_set(self):
        """Should accept recordings with all required channels."""
        # Given - all 19 required channels
        complete_channels = [
            "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
            "T7", "C3", "Cz", "C4", "T8",
            "P7", "P3", "Pz", "P4", "P8",
            "O1", "O2"
        ]

        # When
        from brain_go_brrr.core.channels import ChannelValidator
        validator = ChannelValidator()
        is_valid, missing = validator.validate_channels(complete_channels)

        # Then
        assert is_valid is True
        assert missing == []

    def test_handles_extra_channels_gracefully(self):
        """Should accept recordings with extra non-standard channels."""
        # Given - standard channels plus extras
        channels_with_extras = [
            "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
            "T7", "C3", "Cz", "C4", "T8",
            "P7", "P3", "Pz", "P4", "P8",
            "O1", "O2",
            "ECG", "EOG1", "EOG2", "EMG"  # Extra channels
        ]

        # When
        from brain_go_brrr.core.channels import ChannelValidator
        validator = ChannelValidator()
        is_valid, missing = validator.validate_channels(channels_with_extras)

        # Then
        assert is_valid is True
        assert missing == []

    def test_get_channel_indices_for_standard_channels(self):
        """Should return indices of standard channels in input list."""
        # Given
        channels = ["ECG", "Fp1", "Fp2", "EMG", "F7", "F3"]

        # When
        from brain_go_brrr.core.channels import ChannelValidator
        validator = ChannelValidator()
        indices = validator.get_standard_channel_indices(channels)

        # Then
        assert indices == {"Fp1": 1, "Fp2": 2, "F7": 4, "F3": 5}


class TestChannelSelector:
    """Test selecting and reordering channels."""

    def test_selects_standard_channels_in_order(self):
        """Should select only standard channels in canonical order."""
        # Given - channels in random order with extras
        input_channels = ["EOG", "O2", "Fp1", "C3", "F7", "Pz", "EMG", "T8", "F3"]

        # When
        from brain_go_brrr.core.channels import ChannelSelector
        selector = ChannelSelector()
        selected, indices = selector.select_standard_channels(input_channels)

        # Then - should be in standard order
        assert selected == ["Fp1", "F7", "F3", "C3", "T8", "Pz", "O2"]
        assert indices == [2, 4, 8, 3, 7, 5, 1]  # Original positions

    def test_returns_empty_for_no_standard_channels(self):
        """Should return empty lists when no standard channels present."""
        # Given
        non_standard = ["ECG", "EOG1", "EOG2", "EMG", "X1", "X2"]

        # When
        from brain_go_brrr.core.channels import ChannelSelector
        selector = ChannelSelector()
        selected, indices = selector.select_standard_channels(non_standard)

        # Then
        assert selected == []
        assert indices == []


class TestChannelIntegration:
    """Integration tests for complete channel processing pipeline."""

    def test_full_pipeline_old_to_modern_validation(self):
        """Complete pipeline: map old names, validate, select."""
        # Given - old naming with extras
        raw_channels = [
            "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8",
            "T3", "C3", "CZ", "C4", "T4",  # Old naming
            "T5", "P3", "PZ", "P4", "T6",   # Old naming
            "O1", "O2", "ECG1", "ECG2"
        ]

        # When
        from brain_go_brrr.core.channels import ChannelProcessor
        processor = ChannelProcessor()
        result = processor.process_channels(raw_channels)

        # Then
        assert result.is_valid is True
        assert result.standardized_names[7] == "T7"  # T3 -> T7
        assert result.standardized_names[11] == "T8"  # T4 -> T8
        assert result.standardized_names[12] == "P7"  # T5 -> P7
        assert result.standardized_names[16] == "P8"  # T6 -> P8
        assert len(result.selected_indices) == 19  # All standard channels
        assert result.missing_channels == []
