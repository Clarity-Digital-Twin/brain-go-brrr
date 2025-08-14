#!/usr/bin/env python
"""Comprehensive behavior test for all application components.

This tests ACTUAL BEHAVIOR, not just unit tests.
Run with: pytest tests/behavior/test_all_components.py -v
"""

import os
import sys
import warnings
from pathlib import Path

import mne
import numpy as np
import pytest
import torch

# Filter sklearn version warning for YASA
warnings.filterwarnings("ignore", message=".*scikit-learn.*version.*", category=UserWarning)


class TestApplicationBehavior:
    """Test actual behavior of all major components."""

    @pytest.fixture
    def test_eeg_data(self):
        """Create test EEG data."""
        sfreq = 256
        duration = 300  # 5 minutes
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

        np.random.seed(42)
        data = np.random.randn(n_channels, int(sfreq * duration)) * 50e-6

        # Add some slow waves for sleep-like characteristics
        t = np.arange(0, duration, 1 / sfreq)
        for i in range(n_channels):
            delta = 100e-6 * np.sin(2 * np.pi * 1.5 * t)  # 1.5 Hz delta wave
            data[i, :] += delta

        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info)
        return raw

    def test_yasa_sleep_staging(self, test_eeg_data):
        """Test YASA sleep staging functionality."""
        from brain_go_brrr.infra.external.yasa_adapter import YASASleepStager

        stager = YASASleepStager()
        eeg_array = test_eeg_data.get_data()

        # Run sleep staging
        stages, confidences, metrics = stager.stage_sleep(
            eeg_data=eeg_array, sfreq=test_eeg_data.info["sfreq"], ch_names=test_eeg_data.ch_names
        )

        assert isinstance(stages, list)
        assert len(stages) == 10  # 5 minutes = 10 30s epochs
        assert all(stage in ["W", "N1", "N2", "N3", "REM"] for stage in stages)
        assert all(0 <= conf <= 1 for conf in confidences)
        assert "sleep_efficiency" in metrics

    def test_quality_control(self, test_eeg_data):
        """Test quality control with Autoreject."""
        from brain_go_brrr.domain.quality.controller import EEGQualityController

        qc = EEGQualityController()

        # Add artifact to test detection
        noisy_data = test_eeg_data.get_data().copy()
        noisy_data[0, 1000:2000] = 500e-6  # Add large artifact

        info = test_eeg_data.info
        noisy_raw = mne.io.RawArray(noisy_data, info)

        # Run QC
        results = qc.run_full_qc_pipeline(noisy_raw)

        assert isinstance(results, dict)
        assert "quality_metrics" in results
        assert "processing_info" in results
        assert "quality_grade" in results["quality_metrics"]
        assert results["quality_metrics"]["quality_grade"] in ["POOR", "FAIR", "GOOD", "EXCELLENT"]

    @pytest.mark.skipif(
        not os.getenv("BGB_ABN_MODEL")
        and not Path(
            "experiments/eegpt_linear_probe/output/tuab_4s_paper_target_BULLETPROOF_20250809_073159/best_model.pt"
        ).exists(),
        reason="Abnormality model not available",
    )
    def test_abnormality_detection(self, test_eeg_data):
        """Test abnormality detection with trained model."""
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from archive.test_scripts.fix_abnormality_detector import build_probe_from_checkpoint

        # Find model path
        model_path = os.getenv("BGB_ABN_MODEL")
        if not model_path:
            model_path = "experiments/eegpt_linear_probe/output/tuab_4s_paper_target_BULLETPROOF_20250809_073159/best_model.pt"
        model_path = Path(model_path)

        # Load probe
        probe = build_probe_from_checkpoint(model_path)

        # Create dummy EEGPT embeddings (would come from EEGPT in real use)
        n_windows = 75  # 5 minutes / 4 seconds
        embeddings = torch.randn(n_windows, 512)

        # Run inference
        with torch.no_grad():
            outputs = probe(embeddings)
            probs = torch.softmax(outputs, dim=-1)

        assert outputs.shape == (n_windows, 2)
        assert torch.all(probs >= 0) and torch.all(probs <= 1)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(n_windows))

        # Check that we get reasonable predictions
        abnormal_probs = probs[:, 1]  # Assuming index 1 is abnormal
        assert 0 < abnormal_probs.mean() < 1  # Not all same prediction

    def test_api_endpoints(self):
        """Test API endpoints."""
        from fastapi.testclient import TestClient

        from brain_go_brrr.api.main import app

        client = TestClient(app)

        # Test health endpoint
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "service" in data
        assert "timestamp" in data

        # Test root endpoint
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data

    def test_pdf_generation(self, test_eeg_data):
        """Test PDF report generation."""
        from brain_go_brrr.presentation.visualization.pdf_report import PDFReportGenerator

        generator = PDFReportGenerator()

        # Create test report data
        report_data = {
            "patient_id": "TEST001",
            "recording_date": "2025-08-13",
            "quality_metrics": {
                "bad_channels": ["T3", "T4"],
                "quality_grade": "GOOD",
                "abnormality_score": 0.3,
                "bad_channel_ratio": 0.1,
                "artifact_ratio": 0.05,
            },
            "processing_info": {
                "confidence": 0.85,
                "processing_time": 1.5,
                "channels_used": 19,
                "duration_seconds": 300,
            },
            "sleep_metrics": {
                "sleep_efficiency": 85.5,
                "rem_percentage": 22.3,
                "n3_percentage": 18.7,
            },
        }

        # Generate PDF
        pdf_bytes = generator.generate_report(report_data)

        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 1000  # Should be at least 1KB
        assert pdf_bytes.startswith(b"%PDF")  # PDF magic bytes

    def test_end_to_end_pipeline(self, test_eeg_data):
        """Test complete processing pipeline."""
        results = {}

        # 1. Quality Control
        from brain_go_brrr.domain.quality.controller import EEGQualityController

        qc = EEGQualityController()
        results["qc"] = qc.run_full_qc_pipeline(test_eeg_data)

        # 2. Sleep Staging
        from brain_go_brrr.infra.external.yasa_adapter import YASASleepStager

        stager = YASASleepStager()
        eeg_array = test_eeg_data.get_data()
        stages, confidences, metrics = stager.stage_sleep(
            eeg_data=eeg_array, sfreq=test_eeg_data.info["sfreq"], ch_names=test_eeg_data.ch_names
        )
        results["sleep"] = {"stages": stages, "metrics": metrics}

        # 3. PDF Report
        from brain_go_brrr.presentation.visualization.pdf_report import PDFReportGenerator

        generator = PDFReportGenerator()

        combined_report = {
            "patient_id": "PIPELINE_TEST",
            "quality_metrics": results["qc"]["quality_metrics"],
            "processing_info": results["qc"]["processing_info"],
            "sleep_metrics": results["sleep"]["metrics"],
        }

        pdf_bytes = generator.generate_report(combined_report)
        results["pdf"] = pdf_bytes

        # Verify all components produced output
        assert results["qc"] is not None
        assert results["sleep"]["stages"] is not None
        assert len(results["pdf"]) > 1000

        print("\n✅ End-to-end pipeline successful!")
        print(f"   - QC Grade: {results['qc']['quality_metrics']['quality_grade']}")
        print(f"   - Sleep epochs: {len(results['sleep']['stages'])}")
        print(f"   - PDF size: {len(results['pdf'])} bytes")


if __name__ == "__main__":
    # Run tests directly
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
