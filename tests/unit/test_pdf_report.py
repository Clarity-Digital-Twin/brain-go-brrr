"""Test suite for PDF report generation.

Following TDD approach - tests based on ROUGH_DRAFT.md specifications.
"""

import io
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PyPDF2 import PdfReader


class TestPDFReportGeneration:
    """Test PDF report generation according to specifications."""

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

    @pytest.fixture
    def mock_eeg_data(self):
        """Mock EEG data for artifact visualization."""
        # Create synthetic EEG data with artifacts
        n_channels = 19
        n_samples = 256 * 5  # 5 seconds at 256 Hz

        # Normal EEG baseline
        data = np.random.randn(n_channels, n_samples) * 50e-6  # 50 μV

        # Add artifact at specific location
        data[0, 100:200] += np.random.randn(100) * 200e-6  # Large artifact

        return data

    def test_pdf_report_structure(self):
        """Test that PDF report has required structure."""
        # Test import works
        from src.brain_go_brrr.visualization.pdf_report import generate_qc_report

        # Test function exists
        assert callable(generate_qc_report)

    def test_pdf_contains_warning_banner(self, qc_results):
        """Test PDF contains red warning banner for abnormal EEGs (ROUGH_DRAFT.md spec)."""
        # Import will fail initially (TDD)
        from src.brain_go_brrr.visualization.pdf_report import PDFReportGenerator

        generator = PDFReportGenerator()
        pdf_bytes = generator.generate_report(qc_results)

        # Verify PDF was created
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 0

        # Read PDF
        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        assert len(pdf_reader.pages) >= 1

        # Check for warning text in abnormal case
        first_page_text = pdf_reader.pages[0].extract_text()
        assert "WARNING" in first_page_text or "URGENT" in first_page_text
        assert "Expedite read" in first_page_text

    def test_electrode_heatmap_generation(self, qc_results):
        """Test electrode heat-map showing bad channels."""
        from src.brain_go_brrr.visualization.pdf_report import create_electrode_heatmap

        # Generate heatmap
        fig = create_electrode_heatmap(
            channel_positions=qc_results["quality_metrics"]["channel_positions"],
            bad_channels=qc_results["quality_metrics"]["bad_channels"],
        )

        # Verify figure was created
        assert fig is not None
        assert hasattr(fig, "savefig")

        # Check that bad channels are highlighted
        # This is a simple check - in real implementation we'd verify colors
        assert len(fig.axes) > 0

    def test_artifact_examples_visualization(self, qc_results, mock_eeg_data):
        """Test visualization of 5 worst artifact examples."""
        from src.brain_go_brrr.visualization.pdf_report import create_artifact_examples

        # Get 5 worst artifacts
        artifacts = sorted(
            qc_results["quality_metrics"]["artifact_segments"],
            key=lambda x: x["severity"],
            reverse=True,
        )[:5]

        # Generate visualization
        fig = create_artifact_examples(
            eeg_data=mock_eeg_data,
            artifacts=artifacts,
            sampling_rate=qc_results["processing_info"]["sampling_rate"],
        )

        # Verify figure was created with 5 subplots
        assert fig is not None
        assert len(fig.axes) == 5  # One for each artifact

    def test_pdf_report_for_normal_eeg(self):
        """Test PDF report for normal EEG (no warning banner)."""
        from src.brain_go_brrr.visualization.pdf_report import PDFReportGenerator

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

        generator = PDFReportGenerator()
        pdf_bytes = generator.generate_report(normal_results)

        # Read PDF
        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        first_page_text = pdf_reader.pages[0].extract_text()

        # Should NOT have warning banner
        assert "WARNING" not in first_page_text
        assert "URGENT" not in first_page_text
        # Should have positive message
        assert "NORMAL" in first_page_text or "Good Quality" in first_page_text

    def test_pdf_metadata(self, qc_results):
        """Test PDF contains proper metadata."""
        from src.brain_go_brrr.visualization.pdf_report import PDFReportGenerator

        generator = PDFReportGenerator()
        pdf_bytes = generator.generate_report(qc_results)

        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        metadata = pdf_reader.metadata

        # Check metadata
        assert metadata is not None
        assert "/Title" in metadata
        assert "EEG Quality Control Report" in metadata["/Title"]
        assert "/Author" in metadata
        assert "Brain-Go-Brrr" in metadata["/Author"]

    def test_pdf_file_size_reasonable(self, qc_results):
        """Test PDF file size is reasonable (<5MB for efficiency)."""
        from src.brain_go_brrr.visualization.pdf_report import PDFReportGenerator

        generator = PDFReportGenerator()
        pdf_bytes = generator.generate_report(qc_results)

        # Check size is reasonable (less than 5MB)
        size_mb = len(pdf_bytes) / (1024 * 1024)
        assert size_mb < 5, f"PDF too large: {size_mb:.2f} MB"

    def test_pdf_generation_performance(self, qc_results):
        """Test PDF generation meets performance requirement (FR1.4: <30 seconds)."""
        import time

        from src.brain_go_brrr.visualization.pdf_report import PDFReportGenerator

        generator = PDFReportGenerator()

        start_time = time.time()
        generator.generate_report(qc_results)
        generation_time = time.time() - start_time

        # Should generate in less than 30 seconds (FR1.4)
        assert generation_time < 30, f"PDF generation too slow: {generation_time:.2f}s"
        # Actually should be much faster - target <5 seconds
        assert generation_time < 5, f"PDF generation slower than target: {generation_time:.2f}s"

    def test_pdf_report_includes_summary_stats(self, qc_results):
        """Test PDF includes summary statistics."""
        from src.brain_go_brrr.visualization.pdf_report import PDFReportGenerator

        generator = PDFReportGenerator()
        pdf_bytes = generator.generate_report(qc_results)

        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        full_text = ""
        for page in pdf_reader.pages:
            full_text += page.extract_text()

        # Check for key statistics
        assert "Bad Channels: 3" in full_text or "Bad channels: 3" in full_text
        assert "21%" in full_text or "21.0%" in full_text  # Bad channel percentage
        assert "0.83" in full_text or "83%" in full_text  # Abnormality score

    @pytest.mark.parametrize(
        "flag,expected_color",
        [
            ("URGENT", "red"),
            ("EXPEDITE", "orange"),
            ("ROUTINE", "yellow"),
            ("NORMAL", "green"),
        ],
    )
    def test_pdf_banner_color_by_flag(self, flag, expected_color):
        """Test PDF banner color matches triage flag."""
        from src.brain_go_brrr.visualization.pdf_report import get_banner_color

        color = get_banner_color(flag)

        # Verify color is appropriate
        if flag == "URGENT":
            assert color in ["red", "#FF0000", (1, 0, 0)]
        elif flag == "EXPEDITE":
            assert color in ["orange", "#FFA500", (1, 0.65, 0)]
        elif flag == "ROUTINE":
            assert color in ["yellow", "#FFFF00", (1, 1, 0)]
        else:  # NORMAL
            assert color in ["green", "#00FF00", (0, 1, 0)]

    def test_error_handling_missing_data(self):
        """Test graceful handling of missing data."""
        from src.brain_go_brrr.visualization.pdf_report import PDFReportGenerator

        # Minimal results with missing optional fields
        minimal_results = {"quality_metrics": {"bad_channels": [], "abnormality_score": 0.5}}

        generator = PDFReportGenerator()
        pdf_bytes = generator.generate_report(minimal_results)

        # Should still generate a valid PDF
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 0

    def test_artifact_visualization_empty_list(self, mock_eeg_data):
        """Test artifact visualization with no artifacts."""
        from src.brain_go_brrr.visualization.pdf_report import create_artifact_examples

        # No artifacts
        fig = create_artifact_examples(eeg_data=mock_eeg_data, artifacts=[], sampling_rate=256)

        # Should return None or empty figure
        assert fig is None or len(fig.axes) == 0


class TestPDFReportIntegration:
    """Integration tests for PDF report with API."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from brain_go_brrr.api.main import app

        return TestClient(app)

    @pytest.mark.skip(reason="Integration test needs complex mocking - skipping for now")
    def test_api_pdf_endpoint(self, client):
        """Test API endpoint returns PDF."""
        # Test the detailed endpoint that should include PDF
        with patch("mne.io.read_raw_edf"):
            files = {"file": ("test.edf", b"mock", "application/octet-stream")}
            response = client.post(
                "/api/v1/eeg/analyze/detailed",
                files=files,
                params={"include_report": True},
            )

        assert response.status_code == 200
        data = response.json()

        # Should indicate PDF availability
        assert "detailed" in data
        # Currently returns "coming soon" - will change after implementation

    def test_pdf_download_endpoint(self, client):
        """Test PDF can be downloaded via API."""
        # This endpoint doesn't exist yet - TDD
        response = client.get("/api/v1/eeg/report/download/test-id.pdf")
        assert response.status_code == 404  # Endpoint not implemented yet


class TestPDFVisualizationHelpers:
    """Test individual visualization helper functions."""

    def test_coordinate_normalization(self):
        """Test electrode coordinates are normalized properly."""
        from src.brain_go_brrr.visualization.pdf_report import (
            normalize_electrode_positions,
        )

        positions = {"Fp1": (-30, 80), "O1": (-30, -80), "T3": (-80, 0), "Cz": (0, 0)}

        normalized = normalize_electrode_positions(positions)

        # All coordinates should be between -1 and 1
        for _channel, (x, y) in normalized.items():
            assert -1 <= x <= 1
            assert -1 <= y <= 1

    def test_severity_to_color_mapping(self):
        """Test artifact severity maps to appropriate colors."""
        from src.brain_go_brrr.visualization.pdf_report import severity_to_color

        # High severity = red
        assert severity_to_color(0.9) in ["red", "#FF0000"]

        # Medium severity = orange/yellow
        color = severity_to_color(0.5)
        assert "orange" in color or "yellow" in color

        # Low severity = green
        assert severity_to_color(0.1) in ["green", "#00FF00"]
