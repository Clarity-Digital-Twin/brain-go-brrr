#!/usr/bin/env python
"""Phase 4 TDD: Hierarchical EEG Analysis Pipeline Tests.

Following the architecture:
EEG → Normal/Abnormal → (if abnormal) → Epileptiform categorization
                     → (parallel) → Sleep staging
"""

from pathlib import Path

import numpy as np
import pytest

from brain_go_brrr.services.hierarchical_pipeline import (
    AbnormalityScreener,
    AnalysisResult,
    EpileptiformDetector,
    HierarchicalEEGAnalyzer,
    ParallelExecutor,
    PipelineConfig,
)


class TestHierarchicalPipeline:
    """Test the hierarchical EEG analysis pipeline following TDD."""

    def test_pipeline_configuration(self):
        """Test pipeline can be configured with all components."""
        config = PipelineConfig(
            enable_abnormality_screening=True,
            enable_epileptiform_detection=True,
            enable_sleep_staging=True,
            abnormal_threshold=0.5,
            confidence_threshold=0.8,
            parallel_execution=True,
        )

        pipeline = HierarchicalEEGAnalyzer(config)

        assert pipeline.config.enable_abnormality_screening
        assert pipeline.config.enable_epileptiform_detection
        assert pipeline.config.enable_sleep_staging
        assert pipeline.config.parallel_execution

    def test_normal_eeg_bypasses_epileptiform_detection(self):
        """Test that normal EEG doesn't trigger epileptiform detection."""
        config = PipelineConfig(
            enable_abnormality_screening=True,
            enable_epileptiform_detection=True,
            abnormal_threshold=0.5,
        )

        pipeline = HierarchicalEEGAnalyzer(config)

        # Create mock normal EEG data
        normal_eeg = np.random.randn(19, 2048)  # 19 channels, 8s @ 256Hz

        result = pipeline.analyze(normal_eeg)

        assert result.abnormality_score < 0.5
        assert result.is_abnormal is False
        assert result.epileptiform_events is None  # Not computed for normal
        assert "Skipped epileptiform detection for normal EEG" in result.processing_notes

    def test_abnormal_eeg_triggers_epileptiform_detection(self):
        """Test that abnormal EEG triggers epileptiform detection."""
        config = PipelineConfig(
            enable_abnormality_screening=True,
            enable_epileptiform_detection=True,
            abnormal_threshold=0.5,
        )

        pipeline = HierarchicalEEGAnalyzer(config)

        # Create mock abnormal EEG data with spike-like patterns
        abnormal_eeg = np.random.randn(19, 2048)
        # Add artificial spikes with high amplitude (>30 to trigger abnormality)
        for ch in range(5):  # Add spikes to first 5 channels
            spike_times = [500, 1000, 1500]
            for t in spike_times:
                # Create high amplitude spikes (max ~50) to trigger abnormality detection
                abnormal_eeg[ch, t : t + 10] += 50 * np.exp(-np.arange(10) / 2)

        result = pipeline.analyze(abnormal_eeg)

        assert result.abnormality_score > 0.5
        assert result.is_abnormal is True
        assert result.epileptiform_events is not None
        assert len(result.epileptiform_events) > 0
        assert result.epileptiform_events[0]["type"] in ["spike", "sharp_wave", "spike_wave"]

    def test_parallel_sleep_staging(self):
        """Test sleep staging runs in parallel with main pipeline."""
        config = PipelineConfig(
            enable_abnormality_screening=True,
            enable_sleep_staging=True,
            parallel_execution=True,
            use_yasa_sleep_staging=False,  # Use mock for this test
        )

        pipeline = HierarchicalEEGAnalyzer(config)

        # Create 30s of EEG data (minimum for sleep staging)
        sleep_eeg = np.random.randn(19, 7680)  # 30s @ 256Hz

        result = pipeline.analyze(sleep_eeg)

        assert result.sleep_stage is not None
        assert result.sleep_stage in ["W", "N1", "N2", "N3", "REM"]
        assert result.sleep_confidence > 0
        assert "Sleep staging completed in parallel" in result.processing_notes

    def test_confidence_based_triage(self):
        """Test triage flags based on confidence scores."""
        config = PipelineConfig(enable_abnormality_screening=True, confidence_threshold=0.8)

        pipeline = HierarchicalEEGAnalyzer(config)

        # Test high confidence abnormal
        high_conf_abnormal = pipeline.analyze(self._create_clear_abnormal_eeg())
        assert high_conf_abnormal.triage_flag == "urgent"
        assert high_conf_abnormal.confidence > 0.9

        # Test low confidence abnormal
        low_conf_abnormal = pipeline.analyze(self._create_ambiguous_eeg())
        assert low_conf_abnormal.triage_flag == "review"
        assert 0.5 < low_conf_abnormal.confidence < 0.8

        # Test normal
        normal = pipeline.analyze(self._create_normal_eeg())
        assert normal.triage_flag == "routine"

    def test_processing_time_constraints(self):
        """Test pipeline meets performance requirements."""
        config = PipelineConfig(
            enable_abnormality_screening=True,
            enable_epileptiform_detection=True,
            enable_sleep_staging=True,
            parallel_execution=True,
        )

        pipeline = HierarchicalEEGAnalyzer(config)

        # 5 minutes of EEG data
        long_eeg = np.random.randn(19, 76800)  # 5min @ 256Hz

        import time

        start = time.time()
        result = pipeline.analyze(long_eeg)
        elapsed = time.time() - start

        assert elapsed < 30  # Should process 5min in <30s
        assert result.processing_time_ms < 30000
        assert result.processing_time_ms == pytest.approx(elapsed * 1000, rel=0.1)

    def test_batch_processing(self):
        """Test processing multiple EEG segments efficiently."""
        config = PipelineConfig(
            enable_abnormality_screening=True, enable_epileptiform_detection=True, batch_size=32
        )

        pipeline = HierarchicalEEGAnalyzer(config)

        # Create batch of EEG windows
        batch = [np.random.randn(19, 2048) for _ in range(32)]

        results = pipeline.analyze_batch(batch)

        assert len(results) == 32
        assert all(isinstance(r, AnalysisResult) for r in results)
        assert results[0].batch_processed is True

    def test_error_handling_and_fallbacks(self):
        """Test graceful error handling in pipeline."""
        config = PipelineConfig(
            enable_abnormality_screening=True,
            enable_epileptiform_detection=True,
            enable_error_fallback=True,
        )

        pipeline = HierarchicalEEGAnalyzer(config)

        # Test with corrupted data
        corrupted_eeg = np.full((19, 2048), np.nan)

        result = pipeline.analyze(corrupted_eeg)

        assert result.has_errors is True
        assert "NaN values detected" in result.error_messages
        assert result.abnormality_score == 0.5  # Default fallback
        assert result.triage_flag == "review"  # Force human review

    def test_feature_importance_reporting(self):
        """Test pipeline provides interpretability features."""
        config = PipelineConfig(enable_abnormality_screening=True, enable_feature_importance=True)

        pipeline = HierarchicalEEGAnalyzer(config)

        eeg = np.random.randn(19, 2048)
        result = pipeline.analyze(eeg)

        assert result.feature_importance is not None
        assert "channel_contributions" in result.feature_importance
        assert "temporal_regions" in result.feature_importance
        assert len(result.feature_importance["channel_contributions"]) == 19

    def test_pipeline_state_persistence(self):
        """Test pipeline can save and load state."""
        config = PipelineConfig(
            enable_abnormality_screening=True, checkpoint_dir=Path("/tmp/pipeline_checkpoints")
        )

        pipeline = HierarchicalEEGAnalyzer(config)

        # Process some data to build internal state
        for _ in range(10):
            pipeline.analyze(np.random.randn(19, 2048))

        # Save state
        checkpoint_path = pipeline.save_checkpoint("test_checkpoint")
        assert checkpoint_path.exists()

        # Create new pipeline and load state
        new_pipeline = HierarchicalEEGAnalyzer(config)
        new_pipeline.load_checkpoint(checkpoint_path)

        assert new_pipeline.samples_processed == pipeline.samples_processed
        assert new_pipeline.running_statistics == pipeline.running_statistics

    # Helper methods
    def _create_normal_eeg(self) -> np.ndarray:
        """Create realistic normal EEG."""
        # Dominant alpha rhythm (8-12 Hz)
        t = np.linspace(0, 8, 2048)
        eeg = np.zeros((19, 2048))
        for ch in range(19):
            eeg[ch] = 0.5 * np.sin(2 * np.pi * 10 * t + np.random.rand())
            eeg[ch] += 0.1 * np.random.randn(2048)  # Add noise
        return eeg

    def _create_clear_abnormal_eeg(self) -> np.ndarray:
        """Create clearly abnormal EEG with epileptiform activity."""
        eeg = self._create_normal_eeg()
        # Add clear spikes
        for ch in range(10):
            for spike_time in [512, 1024, 1536]:
                spike = 10 * np.exp(-np.arange(50) / 10)
                eeg[ch, spike_time : spike_time + 50] += spike
        return eeg

    def _create_ambiguous_eeg(self) -> np.ndarray:
        """Create ambiguous EEG that's hard to classify."""
        normal = self._create_normal_eeg()
        abnormal = self._create_clear_abnormal_eeg()
        # Mix 70% normal, 30% abnormal
        return 0.7 * normal + 0.3 * abnormal


class TestAbnormalityScreener:
    """Test the abnormality screening component."""

    def test_screener_with_eegpt_features(self):
        """Test screener uses EEGPT features correctly."""
        screener = AbnormalityScreener(
            model_path=Path("data/models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"),
            use_pretrained_features=True,
        )

        eeg = np.random.randn(19, 2048)
        score, features = screener.screen(eeg, return_features=True)

        assert 0 <= score <= 1
        assert features.shape == (768,)  # EEGPT feature dimension
        assert not np.any(np.isnan(features))

    def test_screener_calibration(self):
        """Test probability calibration of screener."""
        screener = AbnormalityScreener(calibrated=True)

        # Test on 100 samples
        scores = []
        for _ in range(100):
            eeg = np.random.randn(19, 2048)
            score = screener.screen(eeg)
            scores.append(score)

        # Scores should be well-distributed if calibrated
        scores = np.array(scores)
        assert 0.2 < np.mean(scores) < 0.8  # Not all clustered at extremes
        assert np.std(scores) > 0.1  # Good variance


class TestEpileptiformDetector:
    """Test epileptiform event detection."""

    def test_spike_detection(self):
        """Test detection of individual spikes."""
        detector = EpileptiformDetector(
            sensitivity="high", min_spike_duration_ms=20, max_spike_duration_ms=70
        )

        # Create EEG with clear spike
        eeg = np.random.randn(19, 2048) * 10
        spike_channel = 5
        spike_time = 1000
        spike_shape = 50 * np.exp(-np.arange(30) / 5) * np.sin(np.arange(30) / 2)
        eeg[spike_channel, spike_time : spike_time + 30] += spike_shape

        events = detector.detect(eeg)

        assert len(events) >= 1
        spike_event = events[0]
        assert spike_event["type"] == "spike"
        assert spike_event["channel"] == spike_channel
        assert abs(spike_event["time_ms"] - (spike_time / 256 * 1000)) < 50  # Within 50ms

    def test_spike_wave_complex_detection(self):
        """Test detection of spike-wave complexes."""
        detector = EpileptiformDetector()

        # Create spike-wave pattern (3Hz)
        t = np.linspace(0, 8, 2048)
        eeg = np.random.randn(19, 2048) * 5

        # Add 3Hz spike-wave to multiple channels
        for ch in range(5):
            spike_wave = 30 * signal.square(2 * np.pi * 3 * t) * np.exp(-t / 4)
            eeg[ch] += spike_wave

        events = detector.detect(eeg)

        # Should detect spike-wave complex
        sw_events = [e for e in events if e["type"] == "spike_wave_complex"]
        assert len(sw_events) > 0
        assert sw_events[0]["frequency_hz"] == pytest.approx(3.0, rel=0.2)

    def test_polyspike_detection(self):
        """Test detection of polyspikes."""
        detector = EpileptiformDetector()

        eeg = np.random.randn(19, 2048) * 10

        # Create polyspike (multiple spikes in succession)
        polyspike_channel = 10
        spike_times = [1000, 1030, 1060, 1090]  # 30ms apart
        for t in spike_times:
            spike = 40 * np.exp(-np.arange(20) / 3)
            eeg[polyspike_channel, t : t + 20] += spike

        events = detector.detect(eeg)

        polyspike_events = [e for e in events if e["type"] == "polyspike"]
        assert len(polyspike_events) > 0
        assert polyspike_events[0]["spike_count"] >= 3


class TestParallelExecutor:
    """Test parallel execution of pipeline components."""

    async def test_parallel_execution_faster_than_serial(self):
        """Test parallel execution is faster than serial."""
        import asyncio
        import time

        executor = ParallelExecutor(max_workers=4)

        # Define slow tasks
        async def slow_task(duration):
            await asyncio.sleep(duration)
            return duration

        tasks = [slow_task(0.1) for _ in range(4)]

        # Parallel execution
        start = time.time()
        results = await executor.run_parallel(tasks)
        parallel_time = time.time() - start

        # Serial execution
        start = time.time()
        for task in tasks:
            await task
        serial_time = time.time() - start

        assert parallel_time < serial_time * 0.5  # At least 2x speedup
        assert len(results) == 4

    def test_parallel_error_isolation(self):
        """Test errors in one task don't affect others."""
        executor = ParallelExecutor()

        def good_task():
            return "success"

        def bad_task():
            raise ValueError("Task failed")

        results = executor.run_tasks([good_task, bad_task, good_task])

        assert results[0] == "success"
        assert isinstance(results[1], Exception)
        assert results[2] == "success"


# Import guard for scipy
try:
    from scipy import signal
except ImportError:
    signal = None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
