#!/usr/bin/env python3
"""Test script to generate sample markdown report."""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.brain_go_brrr.visualization.markdown_report import MarkdownReportGenerator
from src.brain_go_brrr.utils import utc_now


def main():
    """Generate a sample markdown report."""
    # Create sample QC results
    sample_results = {
        'quality_metrics': {
            'bad_channels': ['T3', 'O2'],
            'bad_channel_ratio': 0.105,
            'abnormality_score': 0.72,
            'quality_grade': 'FAIR',
            'artifact_segments': [
                {'start': 10.5, 'end': 12.3, 'type': 'muscle', 'severity': 0.85},
                {'start': 45.2, 'end': 47.8, 'type': 'eye_blink', 'severity': 0.7},
                {'start': 120.0, 'end': 125.5, 'type': 'electrode_pop', 'severity': 0.95},
            ],
            'channel_positions': {
                'Fp1': (-0.3, 0.8), 'Fp2': (0.3, 0.8),
                'F3': (-0.5, 0.5), 'F4': (0.5, 0.5),
                'C3': (-0.5, 0), 'C4': (0.5, 0),
                'P3': (-0.5, -0.5), 'P4': (0.5, -0.5),
                'O1': (-0.3, -0.8), 'O2': (0.3, -0.8),
                'T3': (-0.8, 0), 'T4': (0.8, 0),
            }
        },
        'processing_info': {
            'file_name': 'sample_eeg.edf',
            'duration_seconds': 600,
            'sampling_rate': 256,
            'timestamp': utc_now().isoformat()
        }
    }
    
    # Generate markdown report
    generator = MarkdownReportGenerator()
    markdown = generator.generate_report(sample_results)
    
    # Print to console
    print(markdown)
    
    # Save to file
    output_dir = Path(__file__).parent.parent / "outputs" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "sample_report.md"
    generator.save_report(sample_results, output_path)
    print(f"\n✅ Markdown report saved to: {output_path}")


if __name__ == "__main__":
    main()