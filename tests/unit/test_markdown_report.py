"""Test suite for Markdown report generation from PDF/QC results.

Following TDD approach - tests for converting reports to markdown format.
"""

import pytest


class TestMarkdownReportGeneration:
    """Test markdown report generation from QC results."""

    @pytest.fixture
    def qc_results(self):
        """Mock QC analysis results."""
        return {
            "quality_metrics": {
                "bad_channels": ["T3", "O2", "Fp1"],
                "bad_channel_ratio": 0.21,
                "abnormality_score": 0.83,
                "quality_grade": "POOR",
                "artifact_segments": [
                    {"start": 10.5, "end": 12.3, "type": "muscle", "severity": 0.9},
                    {"start": 45.2, "end": 47.8, "type": "eye_blink", "severity": 0.8},
                    {
                        "start": 120.0,
                        "end": 125.5,
                        "type": "electrode_pop",
                        "severity": 1.0,
                    },
                    {
                        "start": 200.1,
                        "end": 203.4,
                        "type": "movement",
                        "severity": 0.85,
                    },
                    {"start": 310.0, "end": 315.0, "type": "muscle", "severity": 0.95},
                ],
                "channel_positions": {
                    "Fp1": (-0.3, 0.8),
                    "Fp2": (0.3, 0.8),
                    "F3": (-0.5, 0.5),
                    "F4": (0.5, 0.5),
                    "C3": (-0.5, 0),
                    "C4": (0.5, 0),
                    "P3": (-0.5, -0.5),
                    "P4": (0.5, -0.5),
                    "O1": (-0.3, -0.8),
                    "O2": (0.3, -0.8),
                    "T3": (-0.8, 0),
                    "T4": (0.8, 0),
                },
            },
            "processing_info": {
                "file_name": "test_eeg.edf",
                "duration_seconds": 1200,
                "sampling_rate": 256,
                "timestamp": "2025-01-17T10:30:00",
            },
        }

    def test_markdown_report_structure(self):
        """Test that markdown report has required structure."""
        from src.brain_go_brrr.visualization.markdown_report import (
            generate_markdown_report,
        )

        # Test function exists
        assert callable(generate_markdown_report)

    def test_markdown_contains_warning_section(self, qc_results):
        """Test markdown contains warning section for abnormal EEGs."""
        from src.brain_go_brrr.visualization.markdown_report import (
            MarkdownReportGenerator,
        )

        generator = MarkdownReportGenerator()
        markdown = generator.generate_report(qc_results)

        # Verify markdown was created
        assert isinstance(markdown, str)
        assert len(markdown) > 0

        # Check for warning text in abnormal case
        assert "🚨" in markdown  # Urgent emoji present
        assert "URGENT" in markdown
        assert "Expedite read" in markdown

    def test_markdown_summary_statistics(self, qc_results):
        """Test markdown includes summary statistics section."""
        from src.brain_go_brrr.visualization.markdown_report import (
            MarkdownReportGenerator,
        )

        generator = MarkdownReportGenerator()
        markdown = generator.generate_report(qc_results)

        # Check for summary stats
        assert "## Summary Statistics" in markdown
        assert "**Quality Grade**: POOR" in markdown
        assert "**Bad Channels**: 3 (21.0%)" in markdown
        assert "**Abnormality Score**: 0.83" in markdown

    def test_markdown_channel_quality_table(self, qc_results):
        """Test markdown includes channel quality table."""
        from src.brain_go_brrr.visualization.markdown_report import (
            MarkdownReportGenerator,
        )

        generator = MarkdownReportGenerator()
        markdown = generator.generate_report(qc_results)

        # Check for channel table
        assert "## Channel Quality" in markdown
        assert "| Channel | Status |" in markdown
        assert "| T3 | ❌ Bad |" in markdown
        assert "| C3 | ✅ Good |" in markdown

    def test_markdown_artifact_summary(self, qc_results):
        """Test markdown includes artifact summary."""
        from src.brain_go_brrr.visualization.markdown_report import (
            MarkdownReportGenerator,
        )

        generator = MarkdownReportGenerator()
        markdown = generator.generate_report(qc_results)

        # Check for artifact section
        assert "## Detected Artifacts" in markdown
        assert "electrode_pop" in markdown
        assert "muscle" in markdown
        assert "1.00" in markdown  # Severity value in table

    def test_markdown_for_normal_eeg(self):
        """Test markdown report for normal EEG (no warning)."""
        from src.brain_go_brrr.visualization.markdown_report import (
            MarkdownReportGenerator,
        )

        # Create normal results
        normal_results = {
            "quality_metrics": {
                "bad_channels": [],
                "bad_channel_ratio": 0.0,
                "abnormality_score": 0.15,
                "quality_grade": "EXCELLENT",
                "artifact_segments": [],
            },
            "processing_info": {
                "file_name": "normal_eeg.edf",
                "timestamp": "2025-01-17T10:30:00",
            },
        }

        generator = MarkdownReportGenerator()
        markdown = generator.generate_report(normal_results)

        # Should NOT have warning
        assert "⚠️ WARNING" not in markdown
        assert "🚨 URGENT" not in markdown
        # Should have positive message
        assert "✅" in markdown or "NORMAL" in markdown

    def test_markdown_file_save(self, qc_results, tmp_path):
        """Test saving markdown report to file."""
        from src.brain_go_brrr.visualization.markdown_report import (
            MarkdownReportGenerator,
        )

        generator = MarkdownReportGenerator()
        output_path = tmp_path / "test_report.md"

        # Save to file
        generator.save_report(qc_results, output_path)

        # Verify file exists and has content
        assert output_path.exists()
        content = output_path.read_text()
        assert len(content) > 0
        assert "# EEG Quality Control Report" in content

    def test_markdown_metadata_section(self, qc_results):
        """Test markdown includes metadata section."""
        from src.brain_go_brrr.visualization.markdown_report import (
            MarkdownReportGenerator,
        )

        generator = MarkdownReportGenerator()
        markdown = generator.generate_report(qc_results)

        # Check for metadata
        assert "## File Information" in markdown
        assert "**File**: test_eeg.edf" in markdown
        assert "**Duration**: 1200.0 seconds" in markdown
        assert "**Sampling Rate**: 256 Hz" in markdown

    def test_triage_flag_formatting(self):
        """Test different triage flags have appropriate formatting."""
        from src.brain_go_brrr.visualization.markdown_report import get_triage_emoji

        assert get_triage_emoji("URGENT") == "🚨"
        assert get_triage_emoji("EXPEDITE") == "⚠️"
        assert get_triage_emoji("ROUTINE") == "📋"
        assert get_triage_emoji("NORMAL") == "✅"

    def test_markdown_generation_performance(self, qc_results):
        """Test markdown generation is fast (<1 second)."""
        import time

        from src.brain_go_brrr.visualization.markdown_report import (
            MarkdownReportGenerator,
        )

        generator = MarkdownReportGenerator()

        start_time = time.time()
        generator.generate_report(qc_results)
        generation_time = time.time() - start_time

        # Should generate quickly
        assert generation_time < 1.0, f"Markdown generation too slow: {generation_time:.2f}s"

    def test_markdown_ascii_electrode_map(self, qc_results):
        """Test markdown includes ASCII representation of electrode map."""
        from src.brain_go_brrr.visualization.markdown_report import (
            MarkdownReportGenerator,
        )

        generator = MarkdownReportGenerator()
        markdown = generator.generate_report(qc_results)

        # Check for ASCII electrode map
        assert "## Electrode Map" in markdown
        assert "```" in markdown  # Code block for ASCII art

    def test_convert_pdf_results_to_markdown(self, qc_results):
        """Test converting existing PDF results to markdown."""
        from src.brain_go_brrr.visualization.markdown_report import (
            convert_results_to_markdown,
        )

        # This should work with the same results used for PDF
        markdown = convert_results_to_markdown(qc_results)

        assert isinstance(markdown, str)
        assert len(markdown) > 0
        assert "Quality Grade" in markdown


class TestMarkdownIntegration:
    """Test markdown report integration with API."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient

        from brain_go_brrr.api.main import app

        return TestClient(app)

    def test_api_detailed_includes_markdown(self, client):
        """Test API detailed endpoint can include markdown."""
        from unittest.mock import MagicMock, patch

        # Mock Redis
        mock_cache = MagicMock()
        mock_cache.connected = False  # Simulate Redis not being available

        with (
            patch("mne.io.read_raw_edf"),
            patch("brain_go_brrr.api.cache.get_cache", return_value=mock_cache),
            patch("brain_go_brrr.api.routers.qc.get_cache", return_value=mock_cache),
        ):
            # Create a valid EDF header
            from tests.conftest import valid_edf_content

            edf_content = valid_edf_content()
            files = {"file": ("test.edf", edf_content, "application/octet-stream")}
            response = client.post(
                "/api/v1/eeg/analyze/detailed",
                files=files,
                params={"include_report": True},
            )

        assert response.status_code == 200
        data = response.json()

        # Future: Check for markdown in response
        assert "detailed" in data
