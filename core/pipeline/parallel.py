"""Simple parallel pipeline for EEG analysis.

Runs EEGPT and YASA independently and returns both results.
No fusion, no complexity - just both engines running side by side.
"""

import logging
from pathlib import Path
from typing import Any

import mne
import numpy as np

from core.features import EEGPTFeatureExtractor
from core.sleep import SleepAnalyzer

logger = logging.getLogger(__name__)


class ParallelEEGPipeline:
    """Run EEGPT and YASA in parallel, return both outputs."""

    def __init__(self, eegpt_model_path: Path | None = None, device: str = "cpu"):
        """Initialize parallel pipeline.

        Args:
            eegpt_model_path: Path to EEGPT model checkpoint
            device: Device for EEGPT inference
        """
        # Initialize both services
        self.eegpt_extractor = EEGPTFeatureExtractor(model_path=eegpt_model_path, device=device)
        self.sleep_analyzer = SleepAnalyzer()

        logger.info("Initialized parallel EEG pipeline")

    def process(self, raw: mne.io.Raw) -> dict[str, Any]:
        """Process EEG data through both pipelines.

        Args:
            raw: Raw EEG data

        Returns:
            Dictionary with both EEGPT and YASA results
        """
        results = {}

        # Run EEGPT feature extraction
        logger.info("Extracting EEGPT features...")
        try:
            eegpt_result = self.eegpt_extractor.extract_embeddings_with_metadata(raw)
            results["eegpt"] = {
                "embeddings": eegpt_result["embeddings"],
                "window_times": eegpt_result["window_times"],
                "embedding_shape": eegpt_result["embeddings"].shape,
                "status": "success",
            }
            logger.info(f"EEGPT extracted {eegpt_result['embeddings'].shape[0]} embeddings")
        except Exception as e:
            logger.error(f"EEGPT extraction failed: {e}")
            results["eegpt"] = {"status": "failed", "error": str(e)}

        # Run YASA sleep analysis
        logger.info("Running YASA sleep analysis...")
        try:
            # Get hypnogram with confidence scores
            hypnogram, confidence = self.sleep_analyzer.stage_sleep(
                raw, return_proba=True, apply_smoothing=True
            )

            # Calculate sleep metrics
            sleep_stats = self.sleep_analyzer.compute_sleep_statistics(hypnogram)

            results["yasa"] = {
                "hypnogram": hypnogram.tolist() if isinstance(hypnogram, np.ndarray) else hypnogram,
                "confidence": confidence,
                "sleep_stats": sleep_stats,
                "n_epochs": len(hypnogram),
                "status": "success",
            }
            logger.info(f"YASA staged {len(hypnogram)} epochs")
        except Exception as e:
            logger.error(f"YASA sleep analysis failed: {e}")
            results["yasa"] = {"status": "failed", "error": str(e)}

        # Add metadata
        results["metadata"] = {
            "duration_seconds": raw.times[-1],
            "sampling_rate": raw.info["sfreq"],
            "n_channels": len(raw.ch_names),
            "channel_names": raw.ch_names,
        }

        return results

    def process_file(self, edf_path: Path) -> dict[str, Any]:
        """Process an EDF file through both pipelines.

        Args:
            edf_path: Path to EDF file

        Returns:
            Dictionary with both EEGPT and YASA results
        """
        logger.info(f"Processing file: {edf_path}")

        # Load EDF
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

        # Process through both pipelines
        return self.process(raw)


def main() -> None:
    """Example usage of parallel pipeline."""
    import json

    # Example with Sleep-EDF data
    edf_path = Path("data/datasets/external/sleep-edf/sleep-cassette/SC4001E0-PSG.edf")

    if not edf_path.exists():
        logger.error(f"EDF file not found: {edf_path}")
        return

    # Create pipeline
    pipeline = ParallelEEGPipeline()

    # Process file
    results = pipeline.process_file(edf_path)

    # Print summary
    print("\n=== PARALLEL PIPELINE RESULTS ===")

    if results["eegpt"]["status"] == "success":
        print("\nEEGPT Features:")
        print(f"  - Embeddings shape: {results['eegpt']['embedding_shape']}")
        print(f"  - Number of windows: {len(results['eegpt']['window_times'])}")
    else:
        print(f"\nEEGPT Failed: {results['eegpt']['error']}")

    if results["yasa"]["status"] == "success":
        print("\nYASA Sleep Analysis:")
        print(f"  - Number of epochs: {results['yasa']['n_epochs']}")
        print(f"  - Sleep efficiency: {results['yasa']['sleep_stats'].get('SE', 'N/A')}%")
        print(f"  - Sleep stages present: {set(results['yasa']['hypnogram'])}")
    else:
        print(f"\nYASA Failed: {results['yasa']['error']}")

    print("\nMetadata:")
    print(f"  - Duration: {results['metadata']['duration_seconds']:.1f} seconds")
    print(f"  - Channels: {results['metadata']['n_channels']}")

    # Save results
    output_path = Path("parallel_pipeline_results.json")
    with output_path.open("w") as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {
            "eegpt": {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in results["eegpt"].items()
            },
            "yasa": {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in results["yasa"].items()
            },
            "metadata": results["metadata"],
        }
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
