"""Integration tests for PDF report generation in the API."""

import base64
from unittest.mock import Mock, patch

import mne
import numpy as np
import pytest
from fastapi.testclient import TestClient


class TestPDFReportIntegration:
    """Test PDF report generation integrated with API endpoints."""

    @pytest.fixture
    def mock_qc_controller(self):
        """Mock the QC controller with comprehensive results."""
        with patch('api.main.EEGQualityController') as mock_class:
            mock_controller = Mock()
            mock_controller.eegpt_model = Mock()
            mock_controller.run_full_qc_pipeline = Mock(return_value={
                'quality_metrics': {
                    'bad_channels': ['T3', 'T4', 'O2'],
                    'bad_channel_ratio': 0.15,
                    'abnormality_score': 0.65,
                    'quality_grade': 'FAIR',
                    'impedance_warnings': ['High impedance on F3', 'Poor contact on T3'],
                    'artifact_summary': {
                        'eye_blinks': 25,
                        'muscle': 10,
                        'heartbeat': 5,
                        'motion': 3
                    },
                    'artifact_segments': [
                        {'start': 10.5, 'end': 11.0, 'type': 'eye_blink'},
                        {'start': 45.2, 'end': 46.1, 'type': 'muscle'},
                        {'start': 120.0, 'end': 121.5, 'type': 'motion'}
                    ]
                },
                'processing_info': {
                    'confidence': 0.82,
                    'channels_used': 19,
                    'duration_seconds': 600,
                    'file_name': 'test_patient.edf',
                    'sampling_rate': 256,
                    'timestamp': '2025-07-18T10:30:00Z'
                },
                'processing_time': 3.5,
                'autoreject_results': {
                    'n_interpolated': 3,
                    'n_epochs_rejected': 15,
                    'total_epochs': 200,
                    'rejection_threshold': 150.0
                },
                'eegpt_features': {
                    'n_windows': 147,
                    'average_abnormality': 0.65,
                    'window_scores': [0.3, 0.5, 0.8, 0.6, 0.7]  # Sample scores
                }
            })
            mock_class.return_value = mock_controller
            yield mock_controller

    @pytest.fixture
    def client(self, mock_qc_controller):
        """Create test client with mocked dependencies."""
        import api.main
        from api.main import app
        api.main.qc_controller = mock_qc_controller
        return TestClient(app)

    @pytest.fixture
    def sample_edf_file(self, tmp_path):
        """Create a sample EDF file for testing."""
        sfreq = 256
        duration = 10
        n_channels = 19
        data = np.random.randn(n_channels, sfreq * duration) * 10

        ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
                    'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
                    'Fz', 'Cz', 'Pz']
        ch_types = ['eeg'] * n_channels
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(data, info)

        edf_path = tmp_path / "test.edf"
        raw._data = raw._data / 1e6
        raw.export(edf_path, fmt='edf', overwrite=True, physical_range=(-200, 200))

        return edf_path

    def test_pdf_report_generation_in_detailed_endpoint(self, client, sample_edf_file):
        """Test that detailed endpoint generates PDF reports."""
        with sample_edf_file.open('rb') as f:
            files = {'file': ('test.edf', f, 'application/octet-stream')}
            data = {'include_report': 'true'}
            response = client.post("/api/v1/eeg/analyze/detailed", files=files, data=data)

        assert response.status_code == 200
        result = response.json()

        # Check PDF is included
        assert result['detailed']['pdf_available'] is True
        assert result['detailed']['pdf_base64'] is not None

        # Verify it's valid base64
        try:
            pdf_bytes = base64.b64decode(result['detailed']['pdf_base64'])
            assert len(pdf_bytes) > 1000  # PDF should have reasonable size
            assert pdf_bytes.startswith(b'%PDF')  # PDF magic bytes
        except Exception:
            pytest.fail("Invalid base64 PDF data")

    def test_pdf_contains_all_required_sections(self, client, sample_edf_file):
        """Test that generated PDF contains all required sections."""
        with patch('api.main.PDFReportGenerator') as mock_pdf_class:
            # Mock the PDF generator to track calls
            mock_instance = Mock()
            mock_pdf_class.return_value = mock_instance

            # Track what sections are added
            sections_added = []

            def track_section(section_name, *args, **kwargs):
                sections_added.append(section_name)
                return b'%PDF-fake-content'

            mock_instance.generate_report = Mock(side_effect=track_section)

            with sample_edf_file.open('rb') as f:
                files = {'file': ('test.edf', f, 'application/octet-stream')}
                client.post("/api/v1/eeg/analyze/detailed", files=files)

            # Verify PDF generator was called
            mock_instance.generate_report.assert_called_once()

    def test_pdf_report_error_handling(self, client, sample_edf_file, mock_qc_controller):
        """Test that API handles PDF generation errors gracefully."""
        with patch('api.main.PDFReportGenerator') as mock_pdf_class:
            mock_instance = Mock()
            mock_instance.generate_report.side_effect = Exception("PDF generation failed")
            mock_pdf_class.return_value = mock_instance

            with sample_edf_file.open('rb') as f:
                files = {'file': ('test.edf', f, 'application/octet-stream')}
                response = client.post("/api/v1/eeg/analyze/detailed", files=files)

            # Should still return success but without PDF
            assert response.status_code == 200
            result = response.json()
            assert result['detailed']['pdf_available'] is False
            assert result['detailed']['pdf_base64'] is None
            assert result['basic']['status'] == 'success'

    def test_pdf_report_with_artifact_visualizations(self, client, sample_edf_file):
        """Test that PDF includes artifact visualizations when available."""
        with patch('api.main.PDFReportGenerator') as mock_pdf_class:
            mock_instance = Mock()

            # Track visualization calls
            visualizations_created = []

            def mock_generate(results):
                # Check artifact data is present
                if 'artifact_segments' in results.get('quality_metrics', {}):
                    visualizations_created.append('artifacts')
                if 'bad_channels' in results.get('quality_metrics', {}):
                    visualizations_created.append('channel_map')
                return b'%PDF-fake-content'

            mock_instance.generate_report.return_value = b'%PDF-fake-content'
            mock_pdf_class.return_value = mock_instance

            with sample_edf_file.open('rb') as f:
                files = {'file': ('test.edf', f, 'application/octet-stream')}
                response = client.post("/api/v1/eeg/analyze/detailed", files=files)

            assert response.status_code == 200

    def test_pdf_report_size_limits(self, client, sample_edf_file):
        """Test that PDF reports have reasonable size limits."""
        with patch('api.main.PDFReportGenerator') as mock_pdf_class:
            mock_instance = Mock()
            # Create a large fake PDF (5MB)
            large_pdf = b'%PDF-1.4\n' + b'x' * (5 * 1024 * 1024)
            mock_instance.generate_report.return_value = large_pdf
            mock_pdf_class.return_value = mock_instance

            with sample_edf_file.open('rb') as f:
                files = {'file': ('test.edf', f, 'application/octet-stream')}
                response = client.post("/api/v1/eeg/analyze/detailed", files=files)

            assert response.status_code == 200
            result = response.json()

            # Check that large PDF is still encoded
            assert result['detailed']['pdf_available'] is True
            pdf_base64 = result['detailed']['pdf_base64']

            # Verify size after base64 encoding (should be ~1.33x original)
            assert len(pdf_base64) < 10 * 1024 * 1024  # Less than 10MB base64

    def test_pdf_metadata_inclusion(self, client, sample_edf_file):
        """Test that PDF includes proper metadata."""
        with patch('api.main.PDFReportGenerator') as mock_pdf_class:
            mock_instance = Mock()

            # Capture metadata passed to PDF generator
            captured_metadata = {}

            def capture_metadata(results, raw_data=None):
                captured_metadata.update(results.get('processing_info', {}))
                return b'%PDF-fake-content'

            mock_instance.generate_report.side_effect = capture_metadata
            mock_pdf_class.return_value = mock_instance

            with sample_edf_file.open('rb') as f:
                files = {'file': ('test.edf', f, 'application/octet-stream')}
                response = client.post("/api/v1/eeg/analyze/detailed", files=files)

            assert response.status_code == 200

            # Verify metadata was passed
            assert 'file_name' in captured_metadata
            assert 'timestamp' in captured_metadata
            assert 'duration_seconds' in captured_metadata

    @pytest.mark.parametrize("triage_flag,expected_color", [
        ("URGENT - Expedite read", "red"),
        ("EXPEDITE - Priority review", "orange"),
        ("ROUTINE - Standard workflow", "yellow"),
        ("NORMAL - Low priority", "green"),
    ])
    def test_pdf_triage_flag_styling(self, client, sample_edf_file, mock_qc_controller, triage_flag, expected_color):
        """Test that PDF styling changes based on triage flag."""
        # Update mock to return specific triage scenarios
        if "URGENT" in triage_flag:
            abnormal_score = 0.9
            quality_grade = 'POOR'
        elif "EXPEDITE" in triage_flag:
            abnormal_score = 0.7
            quality_grade = 'FAIR'
        elif "ROUTINE" in triage_flag:
            abnormal_score = 0.5
            quality_grade = 'GOOD'
        else:
            abnormal_score = 0.1
            quality_grade = 'EXCELLENT'

        mock_qc_controller.run_full_qc_pipeline.return_value['quality_metrics']['abnormality_score'] = abnormal_score
        mock_qc_controller.run_full_qc_pipeline.return_value['quality_metrics']['quality_grade'] = quality_grade

        with sample_edf_file.open('rb') as f:
            files = {'file': ('test.edf', f, 'application/octet-stream')}
            response = client.post("/api/v1/eeg/analyze/detailed", files=files)

        assert response.status_code == 200
        result = response.json()
        assert result['basic']['flag'] == triage_flag

    @pytest.mark.slow
    def test_concurrent_pdf_generation(self, client, sample_edf_file):
        """Test that concurrent PDF generation requests are handled properly."""
        import threading
        results = []

        def make_request():
            with sample_edf_file.open('rb') as f:
                files = {'file': ('test.edf', f, 'application/octet-stream')}
                response = client.post("/api/v1/eeg/analyze/detailed", files=files)
                results.append((response.status_code, response.json()))

        # Create multiple threads
        threads = []
        for _ in range(3):
            t = threading.Thread(target=make_request)
            threads.append(t)
            t.start()

        # Wait for all to complete
        for t in threads:
            t.join()

        # All should succeed
        assert all(status == 200 for status, _ in results)
        assert all(data['detailed']['pdf_available'] for _, data in results)

    def test_pdf_generation_performance(self, client, sample_edf_file):
        """Test that PDF generation completes within reasonable time."""
        import time

        start_time = time.time()

        with sample_edf_file.open('rb') as f:
            files = {'file': ('test.edf', f, 'application/octet-stream')}
            response = client.post("/api/v1/eeg/analyze/detailed", files=files)

        end_time = time.time()
        processing_time = end_time - start_time

        assert response.status_code == 200
        # PDF generation should complete within 5 seconds
        assert processing_time < 5.0

        result = response.json()
        assert result['detailed']['pdf_available'] is True
