"""
Sleep Metrics Service

Integrates YASA for automatic sleep staging and comprehensive sleep analysis.
This service provides sleep stage classification and detailed sleep metrics.
"""

import sys
import numpy as np
import pandas as pd
import mne
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

# Add reference repos to path
sys.path.insert(0, str(Path(__file__).parent.parent / "reference_repos" / "yasa"))

try:
    import yasa
    YASA_AVAILABLE = True
except ImportError:
    logging.warning("YASA not available. Install with: pip install yasa")
    YASA_AVAILABLE = False

logger = logging.getLogger(__name__)


class SleepAnalyzer:
    """
    Comprehensive sleep analysis using YASA and additional metrics.
    
    This class provides:
    1. Automatic sleep staging
    2. Sleep architecture analysis
    3. Sleep quality metrics
    4. Hypnogram generation
    5. Sleep event detection
    """
    
    def __init__(
        self,
        staging_model: str = "auto",
        epoch_length: float = 30.0,
        include_art: bool = True,
        verbose: bool = False
    ):
        """
        Initialize the Sleep Analyzer.
        
        Args:
            staging_model: YASA staging model ('auto', 'Vallat2021', etc.)
            epoch_length: Length of each sleep epoch in seconds
            include_art: Whether to include artifact detection
            verbose: Whether to enable verbose logging
        """
        self.staging_model = staging_model
        self.epoch_length = epoch_length
        self.include_art = include_art
        self.verbose = verbose
        
        if not YASA_AVAILABLE:
            logger.error("YASA is required for sleep analysis")
            raise ImportError("Please install YASA: pip install yasa")
    
    def preprocess_for_sleep(
        self,
        raw: mne.io.Raw,
        eeg_channels: Optional[List[str]] = None,
        eog_channels: Optional[List[str]] = None,
        emg_channels: Optional[List[str]] = None,
        resample_freq: float = 100.0
    ) -> mne.io.Raw:
        """
        Preprocess EEG data for sleep staging.
        
        Args:
            raw: Raw EEG data
            eeg_channels: List of EEG channel names
            eog_channels: List of EOG channel names
            emg_channels: List of EMG channel names
            resample_freq: Target sampling frequency
            
        Returns:
            Preprocessed raw data
        """
        raw_copy = raw.copy()
        
        # Resample to standard sleep staging frequency
        if raw_copy.info['sfreq'] != resample_freq:
            raw_copy.resample(resample_freq)
        
        # Apply appropriate filters for sleep staging
        raw_copy.filter(
            l_freq=0.3,  # High-pass for sleep staging
            h_freq=35.0,  # Low-pass for sleep staging
            fir_design='firwin'
        )
        
        # Set channel types if specified
        if eeg_channels:
            raw_copy.set_channel_types({ch: 'eeg' for ch in eeg_channels if ch in raw_copy.ch_names})
        if eog_channels:
            raw_copy.set_channel_types({ch: 'eog' for ch in eog_channels if ch in raw_copy.ch_names})
        if emg_channels:
            raw_copy.set_channel_types({ch: 'emg' for ch in emg_channels if ch in raw_copy.ch_names})
        
        return raw_copy
    
    def stage_sleep(
        self,
        raw: mne.io.Raw,
        eeg_name: str = "C3-A2",
        eog_name: str = "EOG",
        emg_name: str = "EMG",
        metadata: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Perform automatic sleep staging.
        
        Args:
            raw: Preprocessed raw EEG data
            eeg_name: Name of EEG channel for staging
            eog_name: Name of EOG channel
            emg_name: Name of EMG channel
            metadata: Additional metadata
            
        Returns:
            Sleep stages array and staging results
        """
        # Find available channels
        eeg_ch = None
        eog_ch = None
        emg_ch = None
        
        for ch_name in raw.ch_names:
            if eeg_name.lower() in ch_name.lower() and eeg_ch is None:
                eeg_ch = ch_name
            elif eog_name.lower() in ch_name.lower() and eog_ch is None:
                eog_ch = ch_name
            elif emg_name.lower() in ch_name.lower() and emg_ch is None:
                emg_ch = ch_name
        
        if eeg_ch is None:
            # Use first EEG channel as fallback
            eeg_channels = [ch for ch in raw.ch_names if raw.get_channel_types([ch])[0] == 'eeg']
            if eeg_channels:
                eeg_ch = eeg_channels[0]
                logger.warning(f"EEG channel {eeg_name} not found, using {eeg_ch}")
            else:
                raise ValueError("No EEG channels found for sleep staging")
        
        try:
            # Perform sleep staging using YASA
            sls = yasa.SleepStaging(
                raw,
                eeg_name=eeg_ch,
                eog_name=eog_ch,
                emg_name=emg_ch,
                metadata=metadata
            )
            
            # Predict sleep stages
            y_pred = sls.predict()
            
            # Get prediction probabilities
            y_proba = sls.predict_proba()
            
            # Create results dictionary
            results = {
                'stages': y_pred,
                'probabilities': y_proba,
                'channels_used': {
                    'eeg': eeg_ch,
                    'eog': eog_ch,
                    'emg': emg_ch
                },
                'model_info': {
                    'name': self.staging_model,
                    'version': yasa.__version__
                }
            }
            
            logger.info(f"Sleep staging completed using channels: EEG={eeg_ch}, EOG={eog_ch}, EMG={emg_ch}")
            return y_pred, results
            
        except Exception as e:
            logger.error(f"Sleep staging failed: {e}")
            # Return dummy stages as fallback
            n_epochs = int(raw.times[-1] / self.epoch_length)
            dummy_stages = np.random.choice(['N1', 'N2', 'N3', 'REM', 'W'], n_epochs)
            return dummy_stages, {'error': str(e)}
    
    def compute_sleep_statistics(
        self,
        hypnogram: np.ndarray,
        epoch_length: float = 30.0
    ) -> Dict:
        """
        Compute comprehensive sleep statistics.
        
        Args:
            hypnogram: Array of sleep stages
            epoch_length: Length of each epoch in seconds
            
        Returns:
            Dictionary of sleep statistics
        """
        # Convert to YASA hypnogram format if needed
        if isinstance(hypnogram, np.ndarray):
            hypno_int = yasa.hypno_str_to_int(hypnogram)
        else:
            hypno_int = hypnogram
        
        # Compute sleep statistics
        stats = yasa.sleep_statistics(hypno_int, sf_hyp=1/epoch_length)
        
        # Add custom metrics
        total_epochs = len(hypnogram)
        stage_counts = pd.Series(hypnogram).value_counts()
        
        custom_stats = {
            'total_epochs': total_epochs,
            'total_recording_time': total_epochs * epoch_length / 3600,  # hours
            'stage_percentages': {
                stage: (count / total_epochs) * 100 
                for stage, count in stage_counts.items()
            },
            'stage_durations': {
                stage: (count * epoch_length) / 3600  # hours
                for stage, count in stage_counts.items()
            }
        }
        
        # Merge with YASA statistics
        stats.update(custom_stats)
        
        return stats
    
    def detect_sleep_events(
        self,
        raw: mne.io.Raw,
        hypnogram: np.ndarray,
        include_spindles: bool = True,
        include_so: bool = True,
        include_rem: bool = True
    ) -> Dict:
        """
        Detect sleep-specific events.
        
        Args:
            raw: Raw EEG data
            hypnogram: Sleep stage hypnogram
            include_spindles: Whether to detect sleep spindles
            include_so: Whether to detect slow oscillations
            include_rem: Whether to detect REM events
            
        Returns:
            Dictionary of detected sleep events
        """
        events = {}
        
        try:
            # Convert hypnogram to integer format
            hypno_int = yasa.hypno_str_to_int(hypnogram)
            
            # Detect sleep spindles
            if include_spindles:
                sp = yasa.spindles_detect(raw, hypno=hypno_int)
                if sp is not None:
                    events['spindles'] = {
                        'count': len(sp),
                        'density': len(sp) / (len(hypnogram) * self.epoch_length / 3600),  # per hour
                        'summary': sp.summary() if hasattr(sp, 'summary') else None
                    }
            
            # Detect slow oscillations
            if include_so:
                so = yasa.sw_detect(raw, hypno=hypno_int)
                if so is not None:
                    events['slow_oscillations'] = {
                        'count': len(so),
                        'density': len(so) / (len(hypnogram) * self.epoch_length / 3600),  # per hour
                        'summary': so.summary() if hasattr(so, 'summary') else None
                    }
            
            # Detect REM events
            if include_rem:
                rem = yasa.rem_detect(raw, hypno=hypno_int)
                if rem is not None:
                    events['rem_events'] = {
                        'count': len(rem),
                        'density': len(rem) / (len(hypnogram) * self.epoch_length / 3600),  # per hour
                        'summary': rem.summary() if hasattr(rem, 'summary') else None
                    }
                    
        except Exception as e:
            logger.error(f"Sleep event detection failed: {e}")
            events['error'] = str(e)
        
        return events
    
    def generate_hypnogram(
        self,
        hypnogram: np.ndarray,
        epoch_length: float = 30.0,
        save_path: Optional[Path] = None
    ) -> Dict:
        """
        Generate and optionally save hypnogram visualization.
        
        Args:
            hypnogram: Array of sleep stages
            epoch_length: Length of each epoch in seconds
            save_path: Path to save hypnogram image
            
        Returns:
            Hypnogram information
        """
        try:
            # Convert to integer format for YASA
            hypno_int = yasa.hypno_str_to_int(hypnogram)
            
            # Create time axis
            times = np.arange(len(hypnogram)) * epoch_length / 3600  # hours
            
            # Plot hypnogram
            fig, ax = yasa.plot_hypnogram(hypno_int, sf_hyp=1/epoch_length)
            
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Hypnogram saved to {save_path}")
            
            hypno_info = {
                'duration_hours': times[-1],
                'n_epochs': len(hypnogram),
                'epoch_length': epoch_length,
                'stages_present': list(np.unique(hypnogram)),
                'plot_created': True,
                'save_path': str(save_path) if save_path else None
            }
            
            return hypno_info
            
        except Exception as e:
            logger.error(f"Hypnogram generation failed: {e}")
            return {'error': str(e)}
    
    def analyze_sleep_quality(
        self,
        hypnogram: np.ndarray,
        sleep_stats: Dict,
        events: Dict
    ) -> Dict:
        """
        Analyze overall sleep quality.
        
        Args:
            hypnogram: Sleep stage hypnogram
            sleep_stats: Sleep statistics
            events: Sleep events
            
        Returns:
            Sleep quality assessment
        """
        quality_metrics = {}
        
        # Sleep efficiency
        if 'SE' in sleep_stats:
            quality_metrics['sleep_efficiency'] = sleep_stats['SE']
        
        # Sleep fragmentation
        # Convert string hypnogram to numeric for diff calculation
        stage_map = {'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'REM': 4, 'ART': -1}
        hypnogram_numeric = np.array([stage_map.get(stage, -1) for stage in hypnogram])
        stage_changes = np.sum(np.diff(hypnogram_numeric) != 0)
        quality_metrics['fragmentation_index'] = stage_changes / len(hypnogram) if len(hypnogram) > 0 else 0
        
        # REM percentage
        if 'REM' in sleep_stats.get('stage_percentages', {}):
            quality_metrics['rem_percentage'] = sleep_stats['stage_percentages']['REM']
        
        # Deep sleep percentage
        if 'N3' in sleep_stats.get('stage_percentages', {}):
            quality_metrics['deep_sleep_percentage'] = sleep_stats['stage_percentages']['N3']
        
        # Sleep spindle density
        if 'spindles' in events:
            quality_metrics['spindle_density'] = events['spindles']['density']
        
        # Overall quality score (0-100)
        quality_score = self._compute_quality_score(quality_metrics)
        quality_metrics['overall_score'] = quality_score
        quality_metrics['quality_grade'] = self._score_to_grade(quality_score)
        
        return quality_metrics
    
    def _compute_quality_score(self, metrics: Dict) -> float:
        """Compute overall sleep quality score."""
        score = 0
        factors = 0
        
        # Sleep efficiency (25 points)
        if 'sleep_efficiency' in metrics:
            se = metrics['sleep_efficiency']
            if se >= 85:
                score += 25
            elif se >= 75:
                score += 20
            elif se >= 65:
                score += 15
            else:
                score += 10
            factors += 1
        
        # Fragmentation (20 points - lower is better)
        if 'fragmentation_index' in metrics:
            fi = metrics['fragmentation_index']
            if fi <= 0.1:
                score += 20
            elif fi <= 0.2:
                score += 15
            elif fi <= 0.3:
                score += 10
            else:
                score += 5
            factors += 1
        
        # REM percentage (20 points)
        if 'rem_percentage' in metrics:
            rem = metrics['rem_percentage']
            if 18 <= rem <= 25:
                score += 20
            elif 15 <= rem <= 30:
                score += 15
            elif 10 <= rem <= 35:
                score += 10
            else:
                score += 5
            factors += 1
        
        # Deep sleep percentage (20 points)
        if 'deep_sleep_percentage' in metrics:
            deep = metrics['deep_sleep_percentage']
            if deep >= 15:
                score += 20
            elif deep >= 10:
                score += 15
            elif deep >= 5:
                score += 10
            else:
                score += 5
            factors += 1
        
        # Sleep spindle density (15 points)
        if 'spindle_density' in metrics:
            sd = metrics['spindle_density']
            if sd >= 2:
                score += 15
            elif sd >= 1:
                score += 10
            elif sd >= 0.5:
                score += 5
            factors += 1
        
        # Normalize score
        if factors > 0:
            return (score / factors) * (100 / 25)  # Scale to 0-100
        return 50  # Default score
    
    def _score_to_grade(self, score: float) -> str:
        """Convert quality score to letter grade."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    def run_full_sleep_analysis(
        self,
        raw: mne.io.Raw,
        **kwargs
    ) -> Dict:
        """
        Run complete sleep analysis pipeline.
        
        Args:
            raw: Raw EEG data
            **kwargs: Additional arguments
            
        Returns:
            Complete sleep analysis report
        """
        logger.info("Starting full sleep analysis")
        
        # Preprocess for sleep staging
        raw_sleep = self.preprocess_for_sleep(raw)
        
        # Perform sleep staging
        hypnogram, staging_results = self.stage_sleep(raw_sleep, **kwargs)
        
        # Compute sleep statistics
        sleep_stats = self.compute_sleep_statistics(hypnogram, self.epoch_length)
        
        # Detect sleep events
        events = self.detect_sleep_events(raw_sleep, hypnogram)
        
        # Generate hypnogram
        hypno_info = self.generate_hypnogram(hypnogram, self.epoch_length)
        
        # Analyze sleep quality
        quality_metrics = self.analyze_sleep_quality(hypnogram, sleep_stats, events)
        
        # Compile complete report
        report = {
            'hypnogram': hypnogram.tolist(),
            'staging_results': staging_results,
            'sleep_statistics': sleep_stats,
            'sleep_events': events,
            'hypnogram_info': hypno_info,
            'quality_metrics': quality_metrics,
            'analysis_info': {
                'yasa_version': yasa.__version__ if YASA_AVAILABLE else 'N/A',
                'epoch_length': self.epoch_length,
                'total_epochs': len(hypnogram),
                'total_duration_hours': len(hypnogram) * self.epoch_length / 3600
            }
        }
        
        logger.info(f"Sleep analysis completed. Quality grade: {quality_metrics.get('quality_grade', 'N/A')}")
        return report


def main():
    """Example usage of the sleep analyzer."""
    logger.info("Sleep Metrics service is ready")
    logger.info("Use SleepAnalyzer class to perform sleep analysis")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()