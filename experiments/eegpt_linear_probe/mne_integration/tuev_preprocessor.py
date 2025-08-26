"""
MNE+Autoreject preprocessing for TUEV dataset.
Extends TUABPreprocessor to handle TUEV's 23-channel format.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    import mne

import numpy as np

from .preprocessor import TUABPreprocessor

logger = logging.getLogger(__name__)


class TUEVPreprocessor(TUABPreprocessor):
    """MNE+Autoreject preprocessing for TUEV dataset.
    
    Key differences from TUAB:
    1. 23 input channels (not 20)
    2. Different channel naming convention
    3. Multi-class labels from .lab files
    """
    
    # TUEV uses TCP montage with 23 channels
    # These are the channels present in TUEV
    TUEV_CHANNELS = [
        'FP1', 'FP2', 'FPZ',
        'F7', 'F3', 'FZ', 'F4', 'F8',
        'A1', 'T3', 'C3', 'CZ', 'C4', 'T4', 'A2',
        'T5', 'P3', 'PZ', 'P4', 'T6',
        'O1', 'OZ', 'O2'
    ]
    
    # Map TUEV channels to standard 20 (dropping A1, A2, FPZ)
    # Also handle old naming (T3→T7, T4→T8, T5→P7, T6→P8)
    CHANNEL_MAPPING = {
        'T3': 'T7',
        'T4': 'T8',
        'T5': 'P7',
        'T6': 'P8',
        'A1': None,  # Drop reference channels
        'A2': None,
        'FPZ': None  # Drop extra midline channel
    }
    
    # Final 20 channels we want (standard 10-20 without FPZ)
    STANDARD_CHANNELS = [
        'Fp1', 'Fp2',
        'F7', 'F3', 'Fz', 'F4', 'F8',
        'T7', 'C3', 'Cz', 'C4', 'T8',
        'P7', 'P3', 'Pz', 'P4', 'P8',
        'O1', 'O2', 'Oz'
    ]
    
    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize TUEV preprocessor.
        
        Args:
            config: Optional configuration dict
        """
        super().__init__(config)
        logger.info("Initialized TUEVPreprocessor for 23→20 channel mapping")
    
    def _apply_channel_mapping(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Apply TUEV channel mapping from 23 to 20 channels.
        
        Handles:
        - Old to modern naming (T3→T7, etc.)
        - Dropping reference channels (A1, A2)
        - Dropping extra midline channel (FPZ)
        - Case normalization
        
        Args:
            raw: Raw MNE object with 23 channels
            
        Returns:
            Raw object with 20 standard channels
        """
        import re
        
        # First normalize channel names (TUEV uses uppercase)
        rename_dict = {}
        channels_to_drop = []
        
        for ch_name in raw.ch_names:
            # Clean channel name
            clean_name = re.sub(r'^EEG\s+', '', ch_name, flags=re.IGNORECASE)
            clean_name = re.sub(r'-REF$', '', clean_name, flags=re.IGNORECASE)
            clean_name = clean_name.strip().upper()
            
            # Check if this needs mapping
            if clean_name in self.CHANNEL_MAPPING:
                new_name = self.CHANNEL_MAPPING[clean_name]
                if new_name is None:
                    # Mark for dropping
                    channels_to_drop.append(ch_name)
                else:
                    # Map to new name
                    rename_dict[ch_name] = new_name
            else:
                # Standardize case (FP1 -> Fp1, FZ -> Fz, etc.)
                if clean_name == 'FP1':
                    rename_dict[ch_name] = 'Fp1'
                elif clean_name == 'FP2':
                    rename_dict[ch_name] = 'Fp2'
                elif clean_name == 'FZ':
                    rename_dict[ch_name] = 'Fz'
                elif clean_name == 'CZ':
                    rename_dict[ch_name] = 'Cz'
                elif clean_name == 'PZ':
                    rename_dict[ch_name] = 'Pz'
                elif clean_name == 'OZ':
                    rename_dict[ch_name] = 'Oz'
                elif clean_name == 'FPZ':
                    # Drop FPZ
                    channels_to_drop.append(ch_name)
                else:
                    # Keep channel with proper case
                    for std_name in self.STANDARD_CHANNELS:
                        if clean_name == std_name.upper():
                            rename_dict[ch_name] = std_name
                            break
        
        # Apply renaming
        if rename_dict:
            logger.info(f"Renaming {len(rename_dict)} channels")
            raw.rename_channels(rename_dict)
        
        # Drop unwanted channels
        if channels_to_drop:
            logger.info(f"Dropping {len(channels_to_drop)} channels: {channels_to_drop}")
            raw.drop_channels(channels_to_drop)
        
        # Select and reorder to standard 20 channels
        available_standard = [ch for ch in self.STANDARD_CHANNELS if ch in raw.ch_names]
        missing_channels = [ch for ch in self.STANDARD_CHANNELS if ch not in raw.ch_names]
        
        if missing_channels:
            logger.warning(f"Missing standard channels: {missing_channels}")
            if len(available_standard) < 19:  # Minimum requirement
                raise ValueError(
                    f"Too few standard channels ({len(available_standard)}/20). Need at least 19."
                )
        
        # Pick and reorder channels
        raw.pick(available_standard)
        logger.info(f"Selected {len(raw.ch_names)} standard channels from TUEV's 23")
        
        return raw
    
    def process_raw_with_annotations(
        self, 
        edf_path: Path, 
        annotations: List[Dict[str, float | str]]
    ) -> tuple[mne.Epochs, dict[str, int]]:
        """Process raw EDF file with event annotations.
        
        Args:
            edf_path: Path to EDF file
            annotations: List of dicts with 'start', 'end', 'label' keys
            
        Returns:
            Tuple of (clean_epochs, info_dict)
        """
        logger.info(f"Processing {edf_path} with {len(annotations)} annotations")
        
        # Load and preprocess raw data
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
        logger.info(f"Loaded {len(raw.ch_names)} channels, {raw.info['sfreq']} Hz")
        
        # Apply TUEV-specific channel mapping (23→20)
        raw = self._apply_channel_mapping(raw)
        
        # Apply standard preprocessing (filtering, resampling, etc.)
        if raw.info['sfreq'] != self.sampling_rate:
            logger.info(f"Resampling from {raw.info['sfreq']} to {self.sampling_rate} Hz")
            raw.resample(self.sampling_rate, npad='auto')
        
        # Apply MNE preprocessing
        raw = self._apply_mne_preprocessing(raw)
        
        # Create epochs from annotations
        events = []
        event_id = {}
        
        for i, ann in enumerate(annotations):
            # Convert time to samples
            start_sample = int(ann['start'] * raw.info['sfreq'])
            events.append([start_sample, 0, i])  # Use index as event code
            event_id[f"{ann['label']}_{i}"] = i
        
        # Create epochs array
        events_array = mne.events_from_annotations(raw, verbose=False)[0] if events else []
        
        if len(events) > 0:
            events_array = np.array(events)
            
            # Create epochs
            epochs = mne.Epochs(
                raw,
                events_array,
                event_id=event_id,
                tmin=0,
                tmax=self.window_duration,
                baseline=None,
                preload=True,
                verbose=False
            )
        else:
            # Fallback to fixed-length epochs if no annotations
            epochs = self._create_epochs(raw)
        
        n_epochs_before = len(epochs)
        
        # Apply Autoreject
        epochs_clean = self._apply_autoreject(epochs)
        n_epochs_after = len(epochs_clean)
        
        # Create info dict
        info = {
            'n_epochs_before': n_epochs_before,
            'n_epochs_after': n_epochs_after,
            'n_rejected': n_epochs_before - n_epochs_after,
        }
        
        return epochs_clean, info