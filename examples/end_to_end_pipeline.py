"""
End-to-End EEG Processing Pipeline

This example demonstrates how to use all the services together:
1. Load EEG data using pyEDFlib and MNE
2. Quality control with autoreject and EEGPT
3. Sleep analysis with YASA
4. Snippet extraction and analysis
5. Feature extraction with tsfresh
6. Generate comprehensive reports

This is the "one afternoon" implementation suggested in the pragmatic starter stack.
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import mne
import json
from datetime import datetime, timezone

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add reference repos to path
sys.path.insert(0, str(project_root / "reference_repos" / "pyEDFlib"))
sys.path.insert(0, str(project_root / "reference_repos" / "mne-python"))

# Import our services
from services.qc_flagger import EEGQualityController
from services.sleep_metrics import SleepAnalyzer
from services.snippet_maker import EEGSnippetMaker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EEGPipeline:
    """
    Complete EEG processing pipeline integrating all services.
    """
    
    def __init__(
        self,
        output_dir: Path = Path("outputs"),
        enable_qc: bool = True,
        enable_sleep: bool = True,
        enable_snippets: bool = True,
        save_reports: bool = True
    ):
        """
        Initialize the EEG processing pipeline.
        
        Args:
            output_dir: Directory to save outputs
            enable_qc: Enable quality control
            enable_sleep: Enable sleep analysis
            enable_snippets: Enable snippet extraction
            save_reports: Save detailed reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize services
        self.qc_controller = EEGQualityController() if enable_qc else None
        self.sleep_analyzer = SleepAnalyzer() if enable_sleep else None
        self.snippet_maker = EEGSnippetMaker() if enable_snippets else None
        
        self.save_reports = save_reports
        
        logger.info("EEG Pipeline initialized")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"QC enabled: {enable_qc}")
        logger.info(f"Sleep analysis enabled: {enable_sleep}")
        logger.info(f"Snippet extraction enabled: {enable_snippets}")
    
    def load_eeg_data(
        self,
        file_path: Path,
        loader: str = "mne"
    ) -> mne.io.Raw:
        """
        Load EEG data using specified loader.
        
        Args:
            file_path: Path to EEG file
            loader: Loader to use ('mne', 'pyedflib')
            
        Returns:
            Raw EEG data
        """
        logger.info(f"Loading EEG data from {file_path}")
        
        if loader == "mne":
            # Use MNE for loading
            if file_path.suffix.lower() == '.edf':
                raw = mne.io.read_raw_edf(file_path, preload=True)
            elif file_path.suffix.lower() == '.bdf':
                raw = mne.io.read_raw_bdf(file_path, preload=True)
            else:
                # Try generic MNE loader
                raw = mne.io.read_raw(file_path, preload=True)
        
        elif loader == "pyedflib":
            # Use pyEDFlib for loading (requires implementation)
            try:
                import pyedflib
                # Convert pyEDFlib data to MNE format
                # This is a simplified conversion - full implementation needed
                with pyedflib.EdfReader(str(file_path)) as f:
                    n_channels = f.signals_in_file
                    signal_labels = f.getSignalLabels()
                    sfreq = f.getSampleFrequency(0)
                    
                    # Read all signals
                    signals = []
                    for i in range(n_channels):
                        signals.append(f.readSignal(i))
                    
                    # Create MNE info
                    info = mne.create_info(
                        ch_names=signal_labels,
                        sfreq=sfreq,
                        ch_types='eeg'
                    )
                    
                    # Create Raw object
                    raw = mne.io.RawArray(np.array(signals), info)
                    
            except ImportError:
                logger.warning("pyEDFlib not available, falling back to MNE")
                raw = mne.io.read_raw_edf(file_path, preload=True)
        
        else:
            raise ValueError(f"Unknown loader: {loader}")
        
        logger.info(f"Loaded EEG data: {raw.info['nchan']} channels, {raw.info['sfreq']} Hz, {raw.times[-1]:.1f}s")
        return raw
    
    def run_quality_control(
        self,
        raw: mne.io.Raw
    ) -> dict:
        """
        Run quality control pipeline.
        
        Args:
            raw: Raw EEG data
            
        Returns:
            QC report
        """
        if self.qc_controller is None:
            logger.warning("QC controller not initialized")
            return {}
        
        logger.info("Running quality control analysis...")
        
        # Run full QC pipeline
        qc_report = self.qc_controller.run_full_qc_pipeline(raw)
        
        # Save QC report
        if self.save_reports:
            qc_path = self.output_dir / "qc_report.json"
            with open(qc_path, 'w') as f:
                json.dump(qc_report, f, indent=2, default=str)
            logger.info(f"QC report saved to {qc_path}")
        
        return qc_report
    
    def run_sleep_analysis(
        self,
        raw: mne.io.Raw
    ) -> dict:
        """
        Run sleep analysis pipeline.
        
        Args:
            raw: Raw EEG data
            
        Returns:
            Sleep analysis report
        """
        if self.sleep_analyzer is None:
            logger.warning("Sleep analyzer not initialized")
            return {}
        
        logger.info("Running sleep analysis...")
        
        # Run full sleep analysis
        sleep_report = self.sleep_analyzer.run_full_sleep_analysis(raw)
        
        # Save sleep report
        if self.save_reports:
            sleep_path = self.output_dir / "sleep_report.json"
            with open(sleep_path, 'w') as f:
                json.dump(sleep_report, f, indent=2, default=str)
            logger.info(f"Sleep report saved to {sleep_path}")
        
        return sleep_report
    
    def run_snippet_extraction(
        self,
        raw: mne.io.Raw,
        anomaly_scores: np.ndarray = None
    ) -> dict:
        """
        Run snippet extraction and analysis.
        
        Args:
            raw: Raw EEG data
            anomaly_scores: Optional anomaly scores for anomaly-based extraction
            
        Returns:
            Snippet analysis report
        """
        if self.snippet_maker is None:
            logger.warning("Snippet maker not initialized")
            return {}
        
        logger.info("Running snippet extraction...")
        
        # Extract fixed-length snippets
        fixed_snippets = self.snippet_maker.extract_fixed_snippets(raw)
        
        # Extract anomaly-based snippets if scores provided
        anomaly_snippets = []
        if anomaly_scores is not None:
            anomaly_snippets = self.snippet_maker.extract_anomaly_snippets(
                raw, anomaly_scores
            )
        
        # Combine all snippets
        all_snippets = fixed_snippets + anomaly_snippets
        
        # Create comprehensive report
        snippet_report = self.snippet_maker.create_snippet_report(
            all_snippets,
            include_features=True,
            include_eegpt=True
        )
        
        # Save snippets
        if self.save_reports:
            snippet_path = self.output_dir / "snippet_report.json"
            with open(snippet_path, 'w') as f:
                # Remove data arrays for JSON serialization
                report_copy = snippet_report.copy()
                for snippet in report_copy['snippets']:
                    snippet['snippet_info'].pop('data', None)
                json.dump(report_copy, f, indent=2, default=str)
            logger.info(f"Snippet report saved to {snippet_path}")
            
            # Save individual snippets
            snippets_dir = self.output_dir / "snippets"
            self.snippet_maker.save_snippets(all_snippets, snippets_dir)
        
        return snippet_report
    
    def generate_comprehensive_report(
        self,
        raw: mne.io.Raw,
        qc_report: dict,
        sleep_report: dict,
        snippet_report: dict
    ) -> dict:
        """
        Generate comprehensive analysis report.
        
        Args:
            raw: Raw EEG data
            qc_report: Quality control report
            sleep_report: Sleep analysis report
            snippet_report: Snippet analysis report
            
        Returns:
            Comprehensive report
        """
        logger.info("Generating comprehensive report...")
        
        # Create comprehensive report
        report = {
            'metadata': {
                'analysis_timestamp': datetime.now(timezone.utc).isoformat(),
                'pipeline_version': '1.0.0',
                'data_info': {
                    'n_channels': raw.info['nchan'],
                    'sampling_rate': raw.info['sfreq'],
                    'duration_seconds': raw.times[-1],
                    'channel_names': raw.ch_names
                }
            },
            'quality_control': qc_report,
            'sleep_analysis': sleep_report,
            'snippet_analysis': snippet_report,
            'summary': {
                'overall_quality': qc_report.get('quality_metrics', {}).get('quality_grade', 'N/A'),
                'sleep_quality': sleep_report.get('quality_metrics', {}).get('quality_grade', 'N/A'),
                'total_snippets': snippet_report.get('summary', {}).get('total_snippets', 0),
                'processing_success': True
            }
        }
        
        # Save comprehensive report
        if self.save_reports:
            report_path = self.output_dir / "comprehensive_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Comprehensive report saved to {report_path}")
        
        return report
    
    def process_eeg_file(
        self,
        file_path: Path,
        session_id: str = None
    ) -> dict:
        """
        Process a single EEG file through the complete pipeline.
        
        Args:
            file_path: Path to EEG file
            session_id: Optional session identifier
            
        Returns:
            Complete processing report
        """
        if session_id is None:
            session_id = f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting EEG processing for {file_path}")
        logger.info(f"Session ID: {session_id}")
        
        # Create session output directory
        session_dir = self.output_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Temporarily update output directory
        original_output_dir = self.output_dir
        self.output_dir = session_dir
        
        try:
            # Load EEG data
            raw = self.load_eeg_data(file_path)
            
            # Run quality control
            qc_report = self.run_quality_control(raw)
            
            # Run sleep analysis
            sleep_report = self.run_sleep_analysis(raw)
            
            # Generate anomaly scores for snippet extraction
            # This is a placeholder - would use actual EEGPT model
            anomaly_scores = np.random.random(int(raw.times[-1] * raw.info['sfreq']))
            
            # Run snippet extraction
            snippet_report = self.run_snippet_extraction(raw, anomaly_scores)
            
            # Generate comprehensive report
            comprehensive_report = self.generate_comprehensive_report(
                raw, qc_report, sleep_report, snippet_report
            )
            
            logger.info(f"EEG processing completed successfully for {file_path}")
            logger.info(f"Results saved to {session_dir}")
            
            return comprehensive_report
            
        except Exception as e:
            logger.error(f"EEG processing failed: {e}")
            error_report = {
                'error': str(e),
                'session_id': session_id,
                'file_path': str(file_path),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Save error report
            error_path = session_dir / "error_report.json"
            with open(error_path, 'w') as f:
                json.dump(error_report, f, indent=2)
            
            return error_report
        
        finally:
            # Restore original output directory
            self.output_dir = original_output_dir


def main():
    """
    Example usage of the complete EEG processing pipeline.
    """
    logger.info("Starting EEG Pipeline Example")
    
    # Initialize pipeline
    pipeline = EEGPipeline(
        output_dir=Path("outputs"),
        enable_qc=True,
        enable_sleep=True,
        enable_snippets=True,
        save_reports=True
    )
    
    # Example: Process a sample EEG file
    # Replace with actual EEG file path
    sample_file = Path("data/sample_eeg.edf")
    
    if sample_file.exists():
        # Process the file
        report = pipeline.process_eeg_file(sample_file)
        
        # Print summary
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        print(f"Overall Quality: {report.get('summary', {}).get('overall_quality', 'N/A')}")
        print(f"Sleep Quality: {report.get('summary', {}).get('sleep_quality', 'N/A')}")
        print(f"Total Snippets: {report.get('summary', {}).get('total_snippets', 0)}")
        print(f"Processing Success: {report.get('summary', {}).get('processing_success', False)}")
        print("="*60)
        
    else:
        logger.warning(f"Sample file {sample_file} not found")
        logger.info("To test the pipeline:")
        logger.info("1. Place an EEG file (EDF/BDF) in the data/ directory")
        logger.info("2. Update the sample_file path in this script")
        logger.info("3. Run the script again")
        
        # Show available demo
        logger.info("\nFor now, running a demo with synthetic data...")
        
        # Create synthetic EEG data for demonstration
        info = mne.create_info(
            ch_names=['Fp1', 'Fp2', 'C3', 'C4', 'O1', 'O2'],
            sfreq=250,
            ch_types='eeg'
        )
        
        # Generate 5 minutes of synthetic data
        n_samples = 5 * 60 * 250
        data = np.random.randn(6, n_samples) * 20e-6  # 20 µV amplitude
        
        # Create Raw object
        raw = mne.io.RawArray(data, info)
        
        # Run individual components
        logger.info("Running quality control...")
        qc_report = pipeline.run_quality_control(raw)
        
        logger.info("Running sleep analysis...")
        sleep_report = pipeline.run_sleep_analysis(raw)
        
        logger.info("Running snippet extraction...")
        snippet_report = pipeline.run_snippet_extraction(raw)
        
        logger.info("Demo completed successfully!")


if __name__ == "__main__":
    main()