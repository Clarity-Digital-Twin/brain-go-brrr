"""EEG Snippet Maker Service

Creates and manages EEG snippets for analysis, featuring extraction, processing,
and integration with EEGPT for comprehensive snippet analysis.
"""

import json
import logging
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from brain_go_brrr.utils import utc_now

# Add reference repos to path
sys.path.insert(0, str(Path(__file__).parent.parent / "reference_repos" / "EEGPT"))
sys.path.insert(0, str(Path(__file__).parent.parent / "reference_repos" / "tsfresh"))

try:
    import tsfresh
    from tsfresh import extract_features
    from tsfresh.utilities.dataframe_functions import impute
    TSFRESH_AVAILABLE = True
except ImportError:
    logging.warning("tsfresh not available. Install with: pip install tsfresh")
    TSFRESH_AVAILABLE = False

logger = logging.getLogger(__name__)


class EEGSnippetMaker:
    """EEG Snippet creation and analysis service.
    
    This class provides:
    1. Snippet extraction from continuous EEG
    2. Event-based snippet creation
    3. Feature extraction from snippets
    4. EEGPT-based snippet analysis
    5. Snippet classification and annotation
    """

    def __init__(
        self,
        snippet_length: float = 10.0,
        overlap: float = 0.5,
        min_snippet_length: float = 1.0,
        max_snippets_per_file: int = 1000,
        feature_extraction: bool = True
    ):
        """Initialize the EEG Snippet Maker.
        
        Args:
            snippet_length: Default snippet length in seconds
            overlap: Overlap between snippets (0-1)
            min_snippet_length: Minimum snippet length in seconds
            max_snippets_per_file: Maximum snippets to extract per file
            feature_extraction: Whether to extract features from snippets
        """
        self.snippet_length = snippet_length
        self.overlap = overlap
        self.min_snippet_length = min_snippet_length
        self.max_snippets_per_file = max_snippets_per_file
        self.feature_extraction = feature_extraction

        # Initialize feature extraction
        if feature_extraction and not TSFRESH_AVAILABLE:
            logger.warning("tsfresh not available - feature extraction disabled")
            self.feature_extraction = False

    def extract_fixed_snippets(
        self,
        raw: mne.io.Raw,
        snippet_length: float | None = None,
        overlap: float | None = None,
        start_time: float = 0.0,
        end_time: float | None = None,
        channel_selection: list[str] | None = None
    ) -> list[dict]:
        """Extract fixed-length snippets from continuous EEG.
        
        Args:
            raw: Raw EEG data
            snippet_length: Length of each snippet in seconds
            overlap: Overlap between snippets (0-1)
            start_time: Start time for extraction
            end_time: End time for extraction
            channel_selection: List of channels to include
            
        Returns:
            List of snippet dictionaries
        """
        snippet_length = snippet_length or self.snippet_length
        overlap = overlap or self.overlap
        end_time = end_time or raw.times[-1]

        # Calculate step size
        step_size = snippet_length * (1 - overlap)

        # Select channels
        if channel_selection:
            picks = mne.pick_channels(raw.ch_names, channel_selection)
        else:
            picks = mne.pick_types(raw.info, eeg=True, exclude='bads')

        snippets = []
        current_time = start_time
        snippet_id = 0

        while current_time + snippet_length <= end_time and len(snippets) < self.max_snippets_per_file:
            # Extract snippet data
            start_sample = int(current_time * raw.info['sfreq'])
            end_sample = int((current_time + snippet_length) * raw.info['sfreq'])

            snippet_data = raw.get_data(picks=picks, start=start_sample, stop=end_sample)

            # Create snippet dictionary
            snippet = {
                'id': snippet_id,
                'start_time': current_time,
                'end_time': current_time + snippet_length,
                'duration': snippet_length,
                'data': snippet_data,
                'channels': [raw.ch_names[i] for i in picks],
                'sampling_rate': raw.info['sfreq'],
                'n_samples': snippet_data.shape[1],
                'n_channels': snippet_data.shape[0],
                'extraction_method': 'fixed_length',
                'timestamp': utc_now().isoformat()
            }

            snippets.append(snippet)
            current_time += step_size
            snippet_id += 1

        logger.info(f"Extracted {len(snippets)} fixed-length snippets")
        return snippets

    def extract_event_snippets(
        self,
        raw: mne.io.Raw,
        events: np.ndarray,
        event_id: dict[str, int],
        tmin: float = -1.0,
        tmax: float = 2.0,
        channel_selection: list[str] | None = None,
        reject_criteria: dict | None = None
    ) -> list[dict]:
        """Extract event-based snippets from EEG data.
        
        Args:
            raw: Raw EEG data
            events: Events array
            event_id: Event ID mapping
            tmin: Start time relative to event
            tmax: End time relative to event
            channel_selection: List of channels to include
            reject_criteria: Rejection criteria for bad snippets
            
        Returns:
            List of event-based snippet dictionaries
        """
        # Create epochs around events
        epochs = mne.Epochs(
            raw, events, event_id, tmin=tmin, tmax=tmax,
            baseline=None, reject=reject_criteria,
            preload=True, verbose=False
        )

        # Select channels
        if channel_selection:
            epochs.pick_channels(channel_selection)

        snippets = []
        for i, epoch in enumerate(epochs):
            if len(snippets) >= self.max_snippets_per_file:
                break

            # Get event information
            event_sample = events[i, 0]
            event_time = event_sample / raw.info['sfreq']
            event_type = events[i, 2]

            # Find event name
            event_name = None
            for name, id_val in event_id.items():
                if id_val == event_type:
                    event_name = name
                    break

            snippet = {
                'id': i,
                'event_name': event_name,
                'event_type': event_type,
                'event_time': event_time,
                'start_time': event_time + tmin,
                'end_time': event_time + tmax,
                'duration': tmax - tmin,
                'data': epoch,
                'channels': epochs.ch_names,
                'sampling_rate': epochs.info['sfreq'],
                'n_samples': epoch.shape[1],
                'n_channels': epoch.shape[0],
                'extraction_method': 'event_based',
                'timestamp': utc_now().isoformat()
            }

            snippets.append(snippet)

        logger.info(f"Extracted {len(snippets)} event-based snippets")
        return snippets

    def extract_anomaly_snippets(
        self,
        raw: mne.io.Raw,
        anomaly_scores: np.ndarray,
        score_threshold: float = 0.8,
        snippet_length: float = 5.0,
        channel_selection: list[str] | None = None
    ) -> list[dict]:
        """Extract snippets around detected anomalies.
        
        Args:
            raw: Raw EEG data
            anomaly_scores: Array of anomaly scores
            score_threshold: Threshold for anomaly detection
            snippet_length: Length of anomaly snippets
            channel_selection: List of channels to include
            
        Returns:
            List of anomaly snippet dictionaries
        """
        # Find anomaly peaks
        anomaly_indices = np.where(anomaly_scores > score_threshold)[0]

        if len(anomaly_indices) == 0:
            logger.info("No anomalies detected above threshold")
            return []

        # Select channels
        if channel_selection:
            picks = mne.pick_channels(raw.ch_names, channel_selection)
        else:
            picks = mne.pick_types(raw.info, eeg=True, exclude='bads')

        snippets = []
        snippet_id = 0

        for idx in anomaly_indices:
            if len(snippets) >= self.max_snippets_per_file:
                break

            # Calculate snippet boundaries
            anomaly_time = idx / raw.info['sfreq']
            start_time = max(0, anomaly_time - snippet_length / 2)
            end_time = min(raw.times[-1], anomaly_time + snippet_length / 2)

            # Extract snippet data
            start_sample = int(start_time * raw.info['sfreq'])
            end_sample = int(end_time * raw.info['sfreq'])

            snippet_data = raw.get_data(picks=picks, start=start_sample, stop=end_sample)

            snippet = {
                'id': snippet_id,
                'anomaly_time': anomaly_time,
                'anomaly_score': float(anomaly_scores[idx]),
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'data': snippet_data,
                'channels': [raw.ch_names[i] for i in picks],
                'sampling_rate': raw.info['sfreq'],
                'n_samples': snippet_data.shape[1],
                'n_channels': snippet_data.shape[0],
                'extraction_method': 'anomaly_based',
                'timestamp': utc_now().isoformat()
            }

            snippets.append(snippet)
            snippet_id += 1

        logger.info(f"Extracted {len(snippets)} anomaly-based snippets")
        return snippets

    def extract_features_from_snippet(
        self,
        snippet: dict,
        feature_settings: dict | None = None
    ) -> dict:
        """Extract time-series features from a snippet using tsfresh.
        
        Args:
            snippet: Snippet dictionary
            feature_settings: tsfresh feature settings
            
        Returns:
            Dictionary of extracted features
        """
        if not TSFRESH_AVAILABLE:
            logger.warning("tsfresh not available - skipping feature extraction")
            return {}

        try:
            # Prepare data for tsfresh
            data = snippet['data']
            n_channels, n_samples = data.shape

            # Create DataFrame for tsfresh
            df_list = []
            for ch_idx, ch_name in enumerate(snippet['channels']):
                for sample_idx in range(n_samples):
                    df_list.append({
                        'id': snippet['id'],
                        'channel': ch_name,
                        'time': sample_idx / snippet['sampling_rate'],
                        'value': data[ch_idx, sample_idx]
                    })

            df = pd.DataFrame(df_list)

            # Extract features
            if feature_settings is None:
                # Use default feature settings
                features = extract_features(
                    df, column_id='id', column_sort='time',
                    column_value='value', column_kind='channel'
                )
            else:
                features = extract_features(
                    df, column_id='id', column_sort='time',
                    column_value='value', column_kind='channel',
                    default_fc_parameters=feature_settings
                )

            # Impute missing values
            impute(features)

            # Convert to dictionary
            features_dict = features.iloc[0].to_dict()

            return features_dict

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return {}

    def analyze_snippet_with_eegpt(
        self,
        snippet: dict,
        model_path: Path | None = None
    ) -> dict:
        """Analyze snippet using EEGPT model.
        
        Args:
            snippet: Snippet dictionary
            model_path: Path to EEGPT model
            
        Returns:
            EEGPT analysis results
        """
        # TODO: Implement actual EEGPT analysis
        # This is a placeholder for EEGPT integration

        try:
            # Prepare data for EEGPT
            data = snippet['data']

            # Dummy EEGPT analysis (replace with actual implementation)
            analysis = {
                'abnormality_score': np.random.random(),
                'confidence': np.random.random(),
                'predicted_class': np.random.choice(['normal', 'abnormal']),
                'feature_importance': np.random.random(data.shape[0]).tolist(),
                'model_version': 'eegpt-v1.0',
                'processing_time': 0.1
            }

            return analysis

        except Exception as e:
            logger.error(f"EEGPT analysis failed: {e}")
            return {'error': str(e)}

    def classify_snippet(
        self,
        snippet: dict,
        features: dict,
        eegpt_results: dict
    ) -> dict:
        """Classify snippet based on features and EEGPT results.
        
        Args:
            snippet: Snippet dictionary
            features: Extracted features
            eegpt_results: EEGPT analysis results
            
        Returns:
            Classification results
        """
        classification = {
            'snippet_id': snippet['id'],
            'primary_class': 'unknown',
            'confidence': 0.0,
            'secondary_classes': [],
            'quality_score': 0.0
        }

        # Use EEGPT results for classification
        if 'predicted_class' in eegpt_results:
            classification['primary_class'] = eegpt_results['predicted_class']
            classification['confidence'] = eegpt_results.get('confidence', 0.0)

        # Compute quality score based on various factors
        quality_factors = []

        # Signal quality
        if snippet['data'].std() > 0:
            snr = np.mean(snippet['data']) / np.std(snippet['data'])
            quality_factors.append(min(abs(snr) / 10, 1.0))

        # Feature completeness
        if features:
            completeness = len([v for v in features.values() if not np.isnan(v)]) / len(features)
            quality_factors.append(completeness)

        # EEGPT confidence
        if 'confidence' in eegpt_results:
            quality_factors.append(eegpt_results['confidence'])

        classification['quality_score'] = np.mean(quality_factors) if quality_factors else 0.0

        return classification

    def create_snippet_report(
        self,
        snippets: list[dict],
        include_features: bool = True,
        include_eegpt: bool = True
    ) -> dict:
        """Create comprehensive report for all snippets.
        
        Args:
            snippets: List of snippet dictionaries
            include_features: Whether to include feature extraction
            include_eegpt: Whether to include EEGPT analysis
            
        Returns:
            Comprehensive snippet report
        """
        logger.info(f"Creating report for {len(snippets)} snippets")

        processed_snippets = []
        feature_summary = {}
        classification_summary = {}

        for snippet in snippets:
            # Extract features
            features = {}
            if include_features and self.feature_extraction:
                features = self.extract_features_from_snippet(snippet)

            # EEGPT analysis
            eegpt_results = {}
            if include_eegpt:
                eegpt_results = self.analyze_snippet_with_eegpt(snippet)

            # Classify snippet
            classification = self.classify_snippet(snippet, features, eegpt_results)

            # Store processed snippet
            processed_snippet = {
                'snippet_info': {
                    'id': snippet['id'],
                    'start_time': snippet['start_time'],
                    'end_time': snippet['end_time'],
                    'duration': snippet['duration'],
                    'channels': snippet['channels'],
                    'extraction_method': snippet['extraction_method']
                },
                'features': features,
                'eegpt_analysis': eegpt_results,
                'classification': classification
            }

            processed_snippets.append(processed_snippet)

            # Update summaries
            class_name = classification['primary_class']
            if class_name not in classification_summary:
                classification_summary[class_name] = 0
            classification_summary[class_name] += 1

        # Create final report
        report = {
            'summary': {
                'total_snippets': len(snippets),
                'extraction_methods': list(set(s['extraction_method'] for s in snippets)),
                'total_duration': sum(s['duration'] for s in snippets),
                'average_duration': np.mean([s['duration'] for s in snippets]),
                'classification_distribution': classification_summary,
                'average_quality_score': np.mean([s['classification']['quality_score'] for s in processed_snippets])
            },
            'snippets': processed_snippets,
            'processing_info': {
                'features_extracted': include_features and self.feature_extraction,
                'eegpt_analysis': include_eegpt,
                'tsfresh_available': TSFRESH_AVAILABLE,
                'timestamp': utc_now().isoformat()
            }
        }

        logger.info(f"Snippet report created with {len(processed_snippets)} processed snippets")
        return report

    def save_snippets(
        self,
        snippets: list[dict],
        output_dir: Path,
        format: str = 'json'
    ) -> list[Path]:
        """Save snippets to files.
        
        Args:
            snippets: List of snippet dictionaries
            output_dir: Output directory
            format: Save format ('json', 'npz', 'csv')
            
        Returns:
            List of saved file paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_files = []

        for snippet in snippets:
            filename = f"snippet_{snippet['id']:04d}_{snippet['extraction_method']}"

            if format == 'json':
                filepath = output_dir / f"{filename}.json"
                # Convert numpy arrays to lists for JSON serialization
                snippet_copy = snippet.copy()
                snippet_copy['data'] = snippet_copy['data'].tolist()

                with open(filepath, 'w') as f:
                    json.dump(snippet_copy, f, indent=2)

            elif format == 'npz':
                filepath = output_dir / f"{filename}.npz"
                np.savez(filepath, **snippet)

            elif format == 'csv':
                filepath = output_dir / f"{filename}.csv"
                data = snippet['data']
                df = pd.DataFrame(data.T, columns=snippet['channels'])
                df.to_csv(filepath, index=False)

            saved_files.append(filepath)

        logger.info(f"Saved {len(saved_files)} snippets to {output_dir}")
        return saved_files


def main():
    """Example usage of the snippet maker."""
    logger.info("Snippet Maker service is ready")
    logger.info("Use EEGSnippetMaker class to create and analyze EEG snippets")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
