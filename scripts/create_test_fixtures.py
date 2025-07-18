"""Create minimal EEG test fixtures from TUAB dataset.

This script extracts 30-second segments from TUAB EDF files to create
lightweight test fixtures that maintain realism without bloating the repo.
"""

import json
from pathlib import Path
import mne
import numpy as np

# Suppress MNE verbose output
mne.set_log_level('ERROR')

def create_fixture(input_path: Path, output_path: Path, duration: float = 30.0):
    """Extract first N seconds of EDF and save as fixture.
    
    Args:
        input_path: Path to full EDF file
        output_path: Path to save fixture
        duration: Duration in seconds to extract
    """
    # Read EDF with preload=False to save memory
    raw = mne.io.read_raw_edf(input_path, preload=False)
    
    # Crop to first 30 seconds
    raw.crop(tmin=0, tmax=duration)
    
    # Now load the cropped data
    raw.load_data()
    
    # Save as FIF format (more efficient than EDF for fixtures)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw.save(output_path, overwrite=True)
    
    print(f"Created fixture: {output_path.name}")
    print(f"  - Channels: {len(raw.ch_names)}")
    print(f"  - Duration: {raw.times[-1]:.1f}s")
    print(f"  - Size: {output_path.stat().st_size / 1024:.1f} KB")
    
    return raw

def main():
    """Create test fixtures from TUAB dataset."""
    # Define source files
    fixtures = [
        {
            "source": "data/datasets/external/tuh_eeg_abnormal/v3.0.1/edf/eval/normal/01_tcp_ar/aaaaacad_s003_t000.edf",
            "output": "tests/fixtures/eeg/tuab_normal_001.fif",
            "label": 0  # normal
        },
        {
            "source": "data/datasets/external/tuh_eeg_abnormal/v3.0.1/edf/eval/normal/01_tcp_ar/aaaaaomx_s001_t000.edf", 
            "output": "tests/fixtures/eeg/tuab_normal_002.fif",
            "label": 0  # normal
        },
        {
            "source": "data/datasets/external/tuh_eeg_abnormal/v3.0.1/edf/eval/abnormal/01_tcp_ar/aaaaaijh_s002_t000.edf",
            "output": "tests/fixtures/eeg/tuab_abnormal_001.fif",
            "label": 1  # abnormal
        },
        {
            "source": "data/datasets/external/tuh_eeg_abnormal/v3.0.1/edf/eval/abnormal/01_tcp_ar/aaaaahie_s005_t001.edf",
            "output": "tests/fixtures/eeg/tuab_abnormal_002.fif", 
            "label": 1  # abnormal
        }
    ]
    
    labels = {}
    
    for fixture in fixtures:
        source_path = Path(fixture["source"])
        output_path = Path(fixture["output"])
        
        if not source_path.exists():
            print(f"WARNING: Source file not found: {source_path}")
            continue
            
        # Create fixture
        raw = create_fixture(source_path, output_path)
        
        # Store label
        labels[output_path.name] = {
            "label": fixture["label"],
            "label_name": "normal" if fixture["label"] == 0 else "abnormal",
            "channels": len(raw.ch_names),
            "sfreq": raw.info['sfreq'],
            "duration": float(raw.times[-1])
        }
        
        print()
    
    # Save labels file
    labels_path = Path("tests/fixtures/eeg/labels.json")
    with open(labels_path, 'w') as f:
        json.dump(labels, f, indent=2)
    
    print(f"Created labels file: {labels_path}")
    print(f"Total fixtures created: {len(labels)}")

if __name__ == "__main__":
    main()