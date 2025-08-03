#!/usr/bin/env python
"""Hierarchical EEG Analysis Pipeline - TDD Implementation.

Architecture:
EEG → Normal/Abnormal → (if abnormal) → Epileptiform categorization
                     → (parallel) → Sleep staging
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for hierarchical EEG analysis pipeline."""
    enable_abnormality_screening: bool = True
    enable_epileptiform_detection: bool = True
    enable_sleep_staging: bool = True
    abnormal_threshold: float = 0.5
    confidence_threshold: float = 0.8
    parallel_execution: bool = True
    batch_size: int = 32
    enable_error_fallback: bool = True
    enable_feature_importance: bool = False
    checkpoint_dir: Path | None = None
    use_yasa_sleep_staging: bool = True  # Use real YASA implementation


@dataclass
class AnalysisResult:
    """Result from hierarchical EEG analysis."""
    abnormality_score: float
    is_abnormal: bool
    confidence: float
    triage_flag: str  # 'routine', 'review', 'urgent'
    epileptiform_events: list[dict[str, Any]] | None = None
    sleep_stage: str | None = None
    sleep_confidence: float | None = None
    processing_time_ms: float = 0
    processing_notes: list[str] = field(default_factory=list)
    feature_importance: dict[str, Any] | None = None
    has_errors: bool = False
    error_messages: list[str] = field(default_factory=list)
    batch_processed: bool = False


class AbnormalityScreener:
    """Binary abnormality screening using EEGPT features."""

    def __init__(
        self,
        model_path: Path | None = None,
        use_pretrained_features: bool = True,
        calibrated: bool = False
    ):
        """Initialize abnormality screener."""
        self.model_path = model_path
        self.use_pretrained_features = use_pretrained_features
        self.calibrated = calibrated

        # Mock implementation for TDD
        self._mock_scores: list[float] = []

    def screen(
        self,
        eeg: npt.NDArray[np.float64],
        return_features: bool = False
    ) -> float | tuple[float, npt.NDArray[np.float64]]:
        """Screen EEG for abnormality."""
        # Mock implementation for TDD
        # Check for spikes in the data (simple heuristic)
        max_amp = np.max(np.abs(eeg))

        # If we see high amplitude spikes, likely abnormal
        score = 0.8 + 0.2 * np.random.rand() if max_amp > 30 else 0.2 + 0.3 * np.random.rand()

        if self.calibrated:
            # Add some variance for calibration testing
            self._mock_scores.append(score)
            score = np.clip(score + 0.1 * np.random.randn(), 0, 1)

        if return_features:
            # Mock EEGPT features
            features = np.random.randn(768)
            return score, features

        return score


class EpileptiformDetector:
    """Detect epileptiform events in EEG."""

    def __init__(
        self,
        sensitivity: str = 'medium',
        min_spike_duration_ms: float = 20,
        max_spike_duration_ms: float = 70
    ):
        """Initialize epileptiform detector."""
        self.sensitivity = sensitivity
        self.min_spike_duration_ms = min_spike_duration_ms
        self.max_spike_duration_ms = max_spike_duration_ms

    def detect(self, eeg: npt.NDArray[np.float64]) -> list[dict[str, Any]]:
        """Detect epileptiform events."""
        events = []

        # Mock detection - look for high amplitude transients
        for ch in range(eeg.shape[0]):
            channel_data = eeg[ch]
            # Simple peak detection
            threshold = 3 * np.std(channel_data)
            peaks = np.where(np.abs(channel_data) > threshold)[0]

            if len(peaks) > 0:
                # Group nearby peaks
                for peak in peaks[:3]:  # Limit for mock
                    events.append({
                        'type': 'spike',
                        'channel': ch,
                        'time_ms': peak / 256 * 1000,  # 256Hz sampling
                        'amplitude': float(channel_data[peak]),
                        'duration_ms': 30.0
                    })

        # Check for spike-wave patterns (mock)
        if len(events) > 5:
            events.append({
                'type': 'spike_wave_complex',
                'channels': list(range(5)),
                'frequency_hz': 3.0,
                'duration_ms': 1000.0
            })

        # Check for polyspikes (mock)
        if len(events) > 10:
            events.append({
                'type': 'polyspike',
                'channel': 10,
                'spike_count': 4,
                'duration_ms': 120.0
            })

        return events


class SleepStager:
    """Sleep stage classification."""

    def __init__(self, use_yasa: bool = True):
        """Initialize sleep stager."""
        self.stages = ['W', 'N1', 'N2', 'N3', 'REM']
        self.use_yasa = use_yasa

        if use_yasa:
            try:
                from brain_go_brrr.services.yasa_adapter import HierarchicalPipelineYASAAdapter
                self.yasa_adapter = HierarchicalPipelineYASAAdapter()
                logger.info("Using YASA for sleep staging")
            except ImportError:
                logger.warning("YASA not available, using mock implementation")
                self.use_yasa = False
                self.yasa_adapter = None
        else:
            self.yasa_adapter = None

    def stage(self, eeg: npt.NDArray[np.float64]) -> tuple[str, float]:
        """Classify sleep stage."""
        if self.use_yasa and self.yasa_adapter:
            try:
                return self.yasa_adapter.stage(eeg)
            except Exception as e:
                logger.error(f"YASA staging failed: {e}, falling back to mock")

        # Mock implementation fallback
        # Check for alpha rhythm (8-12Hz) → Wake
        # Low amplitude → N1
        # Sleep spindles → N2
        # Delta waves → N3
        # Mixed frequency → REM

        # Simple heuristic based on amplitude
        mean_amp = np.mean(np.abs(eeg))

        if mean_amp > 20:
            return 'W', 0.85
        elif mean_amp > 15:
            return 'REM', 0.75
        elif mean_amp > 10:
            return 'N1', 0.80
        elif mean_amp > 5:
            return 'N2', 0.85
        else:
            return 'N3', 0.90


class ParallelExecutor:
    """Execute tasks in parallel."""

    def __init__(self, max_workers: int = 4):
        """Initialize parallel executor."""
        self.max_workers = max_workers

    async def run_parallel(self, tasks: list) -> list[Any]:
        """Run tasks in parallel using asyncio."""
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

    def run_tasks(self, tasks: list) -> list[Any]:
        """Run callable tasks in parallel using threads."""
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(task) for task in tasks]
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append(e)
        return results


class HierarchicalEEGAnalyzer:
    """Main hierarchical EEG analysis pipeline."""

    def __init__(self, config: PipelineConfig):
        """Initialize pipeline with configuration."""
        self.config = config
        self.screener = AbnormalityScreener(calibrated=True)
        self.epileptiform_detector = EpileptiformDetector()
        self.sleep_stager = SleepStager(use_yasa=config.use_yasa_sleep_staging)
        self.executor = ParallelExecutor()

        # State tracking
        self.samples_processed = 0
        self.running_statistics = {}

        # Create checkpoint directory if specified
        if config.checkpoint_dir:
            config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def analyze(self, eeg: npt.NDArray[np.float64]) -> AnalysisResult:
        """Analyze EEG data through hierarchical pipeline."""
        start_time = time.time()
        result = AnalysisResult(
            abnormality_score=0.5,
            is_abnormal=False,
            confidence=0.0,
            triage_flag='routine'
        )

        # Check for errors
        if np.any(np.isnan(eeg)):
            result.has_errors = True
            result.error_messages.append("NaN values detected")
            if self.config.enable_error_fallback:
                result.triage_flag = 'review'
                result.processing_notes.append("Error detected - forced to review")
                result.processing_time_ms = (time.time() - start_time) * 1000
                return result

        # Step 1: Abnormality screening
        if self.config.enable_abnormality_screening:
            if self.config.enable_feature_importance:
                score, features = self.screener.screen(eeg, return_features=True)
                result.feature_importance = {
                    'channel_contributions': np.random.rand(eeg.shape[0]).tolist(),
                    'temporal_regions': np.random.rand(10).tolist()
                }
            else:
                score = self.screener.screen(eeg)

            result.abnormality_score = score
            result.is_abnormal = score > self.config.abnormal_threshold

            # Calculate confidence (distance from threshold)
            result.confidence = abs(score - 0.5) * 2  # Scale to 0-1

            # Triage based on confidence and score
            if result.is_abnormal and result.confidence > 0.8:
                result.triage_flag = 'urgent'
            elif result.is_abnormal and result.confidence < 0.6:
                result.triage_flag = 'review'
            else:
                result.triage_flag = 'routine'

        # Parallel execution setup
        tasks = []

        # Step 2: Conditional epileptiform detection
        if self.config.enable_epileptiform_detection and result.is_abnormal:
            if self.config.parallel_execution:
                tasks.append(lambda: self.epileptiform_detector.detect(eeg))
            else:
                result.epileptiform_events = self.epileptiform_detector.detect(eeg)
        elif self.config.enable_epileptiform_detection and not result.is_abnormal:
            result.processing_notes.append("Skipped epileptiform detection for normal EEG")

        # Step 3: Parallel sleep staging
        if self.config.enable_sleep_staging and eeg.shape[1] >= 7680:  # 30s minimum
            if self.config.parallel_execution:
                tasks.append(lambda: self.sleep_stager.stage(eeg))
            else:
                stage, confidence = self.sleep_stager.stage(eeg)
                result.sleep_stage = stage
                result.sleep_confidence = confidence

        # Execute parallel tasks
        if tasks and self.config.parallel_execution:
            results = self.executor.run_tasks(tasks)

            # Process results - track which task is which
            task_idx = 0
            if self.config.enable_epileptiform_detection and result.is_abnormal:
                if not isinstance(results[task_idx], Exception):
                    result.epileptiform_events = results[task_idx]
                else:
                    result.has_errors = True
                    result.error_messages.append(str(results[task_idx]))
                task_idx += 1

            if self.config.enable_sleep_staging and eeg.shape[1] >= 7680:
                if not isinstance(results[task_idx], Exception):
                    stage, confidence = results[task_idx]
                    result.sleep_stage = stage
                    result.sleep_confidence = confidence
                    result.processing_notes.append("Sleep staging completed in parallel")
                else:
                    result.has_errors = True
                    result.error_messages.append(str(results[task_idx]))

        # Update statistics
        self.samples_processed += 1

        # Calculate processing time
        result.processing_time_ms = (time.time() - start_time) * 1000

        return result

    def analyze_batch(self, batch: list[npt.NDArray[np.float64]]) -> list[AnalysisResult]:
        """Analyze batch of EEG segments."""
        results = []
        for eeg in batch:
            result = self.analyze(eeg)
            result.batch_processed = True
            results.append(result)
        return results

    def save_checkpoint(self, name: str) -> Path:
        """Save pipeline state checkpoint."""
        if not self.config.checkpoint_dir:
            raise ValueError("Checkpoint directory not configured")

        checkpoint_path = self.config.checkpoint_dir / f"{name}.pkl"

        # Mock save
        import pickle
        state = {
            'samples_processed': self.samples_processed,
            'running_statistics': self.running_statistics
        }
        with checkpoint_path.open('wb') as f:
            pickle.dump(state, f)

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: Path):
        """Load pipeline state from checkpoint."""
        import pickle
        with checkpoint_path.open('rb') as f:
            state = pickle.load(f)

        self.samples_processed = state['samples_processed']
        self.running_statistics = state['running_statistics']
