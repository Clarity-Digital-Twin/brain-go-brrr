"""PDF Report Generation for EEG Quality Control.

Generates professional PDF reports with:
- Warning banners for abnormal EEGs
- Electrode heat-maps showing bad channels
- Visualization of worst artifact examples
"""

import io
import logging
from typing import Any

import matplotlib

from brain_go_brrr.utils import utc_now

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


def generate_qc_report(qc_results: dict[str, Any], eeg_data: npt.NDArray | None = None) -> bytes:
    """Generate a QC report from results.

    Args:
        qc_results: QC analysis results
        eeg_data: Optional EEG data for artifact visualization

    Returns:
        PDF bytes
    """
    generator = PDFReportGenerator()
    return generator.generate_report(qc_results, eeg_data)


class PDFReportGenerator:
    """Generate professional PDF reports for EEG QC analysis."""

    def __init__(self) -> None:
        """Initialize PDF report generator."""
        self.figsize = (8.5, 11)  # Letter size
        self.dpi = 100

    def generate_report(
        self, results: dict[str, Any], eeg_data: npt.NDArray | None = None
    ) -> bytes:
        """Generate complete PDF report.

        Args:
            results: QC analysis results
            eeg_data: Optional EEG data for artifact visualization

        Returns:
            PDF as bytes
        """
        pdf_buffer = io.BytesIO()

        with PdfPages(
            pdf_buffer,
            metadata={
                "Title": "EEG Quality Control Report",
                "Author": "Brain-Go-Brrr",
                "Subject": "EEG Analysis Results",
                "Creator": "Brain-Go-Brrr v0.1.0",
                "CreationDate": utc_now(),
            },
        ) as pdf:
            # Create main report page
            fig = self._create_main_page(results)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # Add artifact examples if available
            if eeg_data is not None and "quality_metrics" in results:
                artifacts = results["quality_metrics"].get("artifact_segments", [])
                if artifacts:
                    fig = self._create_artifact_page(eeg_data, artifacts, results)
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)

        pdf_buffer.seek(0)
        return pdf_buffer.read()

    def _create_main_page(self, results: dict[str, Any]) -> Figure:
        """Create main report page with summary and visualizations."""
        fig: Figure = plt.figure(figsize=self.figsize)

        # Get quality metrics
        quality_metrics = results.get("quality_metrics", {})
        processing_info = results.get("processing_info", {})

        # Determine abnormality and triage flag
        abnormality_score = quality_metrics.get("abnormality_score", 0)
        quality_grade = quality_metrics.get("quality_grade", "UNKNOWN")

        # Determine triage flag based on score
        if abnormality_score > 0.8 or quality_grade == "POOR":
            flag = "URGENT"
            banner_text = "WARNING: URGENT - Expedite read"
        elif abnormality_score > 0.6 or quality_grade == "FAIR":
            flag = "EXPEDITE"
            banner_text = "EXPEDITE - Priority review recommended"
        elif abnormality_score > 0.4:
            flag = "ROUTINE"
            banner_text = "ROUTINE - Standard workflow"
        else:
            flag = "NORMAL"
            banner_text = "NORMAL - Good Quality EEG"

        # Add warning banner at top
        if flag != "NORMAL":
            self._add_warning_banner(fig, flag, banner_text)
        else:
            self._add_normal_banner(fig, banner_text)

        # Add report header
        self._add_header(fig, processing_info)

        # Add summary statistics
        self._add_summary_stats(fig, quality_metrics, processing_info)

        # Add electrode heatmap if channel positions available
        if "channel_positions" in quality_metrics:
            self._add_electrode_heatmap(
                fig,
                quality_metrics["channel_positions"],
                quality_metrics.get("bad_channels", []),
            )

        return fig

    def _add_warning_banner(self, fig: Figure, flag: str, text: str) -> None:
        """Add colored warning banner based on triage flag."""
        color = get_banner_color(flag)

        # Add banner as colored rectangle at top
        ax = fig.add_axes((0, 0.92, 1, 0.08))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Add colored background
        rect = patches.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="none")
        ax.add_patch(rect)

        # Add warning text
        ax.text(
            0.5,
            0.5,
            text,
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            color="white",
        )

        ax.axis("off")

    def _add_normal_banner(self, fig: Figure, text: str) -> None:
        """Add green banner for normal EEG."""
        ax = fig.add_axes((0, 0.92, 1, 0.08))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Add green background
        rect = patches.Rectangle((0, 0), 1, 1, facecolor="green", edgecolor="none")
        ax.add_patch(rect)

        # Add text
        ax.text(
            0.5,
            0.5,
            text,
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            color="white",
        )

        ax.axis("off")

    def _add_header(self, fig: Figure, processing_info: dict[str, Any]) -> None:
        """Add report header with file info."""
        ax = fig.add_axes((0.1, 0.82, 0.8, 0.08))
        ax.axis("off")

        # Title
        ax.text(
            0.5,
            0.7,
            "EEG Quality Control Report",
            ha="center",
            fontsize=18,
            fontweight="bold",
        )

        # File info
        filename = processing_info.get("file_name", "Unknown")
        timestamp = processing_info.get("timestamp", utc_now().isoformat())

        ax.text(0.5, 0.3, f"File: {filename}", ha="center", fontsize=10)
        ax.text(0.5, 0.1, f"Generated: {timestamp}", ha="center", fontsize=8)

    def _add_summary_stats(
        self,
        fig: Figure,
        quality_metrics: dict[str, Any],
        processing_info: dict[str, Any],
    ) -> None:
        """Add summary statistics section."""
        ax = fig.add_axes((0.1, 0.6, 0.8, 0.2))
        ax.axis("off")

        # Extract metrics
        bad_channels = quality_metrics.get("bad_channels", [])
        bad_ratio = quality_metrics.get("bad_channel_ratio", 0)
        abnormality = quality_metrics.get("abnormality_score", 0)
        quality_grade = quality_metrics.get("quality_grade", "UNKNOWN")

        # Format text
        stats_text = [
            f"Quality Grade: {quality_grade}",
            f"Bad Channels: {len(bad_channels)} ({bad_ratio * 100:.1f}%)",
            f"Abnormality Score: {abnormality:.2f}",
            f"Duration: {processing_info.get('duration_seconds', 0):.1f}s",
            f"Sampling Rate: {processing_info.get('sampling_rate', 0)}Hz",
        ]

        # Add bad channel names if any
        if bad_channels:
            stats_text.append(f"Bad channels: {', '.join(bad_channels[:10])}")
            if len(bad_channels) > 10:
                stats_text.append(f"    ... and {len(bad_channels) - 10} more")

        # Display stats
        y_pos = 0.9
        for stat in stats_text:
            ax.text(0.1, y_pos, stat, fontsize=11, verticalalignment="top")
            y_pos -= 0.15

    def _add_electrode_heatmap(
        self,
        fig: Figure,
        channel_positions: dict[str, tuple[float, float]],
        bad_channels: list[str],
    ) -> None:
        """Add electrode heatmap visualization."""
        ax = fig.add_axes((0.2, 0.15, 0.6, 0.4))

        # Normalize positions if needed
        positions = normalize_electrode_positions(channel_positions)

        # Create head circle
        head = patches.Circle((0, 0), 1, fill=False, linewidth=2)
        ax.add_patch(head)

        # Add nose
        nose = patches.Wedge((0, 1), 0.1, 70, 110, fill=False, linewidth=2)
        ax.add_patch(nose)

        # Plot electrodes
        for channel, (x, y) in positions.items():
            if channel in bad_channels:
                # Bad channels in red
                ax.plot(
                    x,
                    y,
                    "ro",
                    markersize=12,
                    markerfacecolor="red",
                    markeredgecolor="darkred",
                    markeredgewidth=2,
                )
                ax.text(x, y - 0.15, channel, ha="center", fontsize=8, color="red")
            else:
                # Good channels in green
                ax.plot(
                    x,
                    y,
                    "go",
                    markersize=10,
                    markerfacecolor="lightgreen",
                    markeredgecolor="darkgreen",
                    markeredgewidth=1,
                )
                ax.text(x, y - 0.15, channel, ha="center", fontsize=6)

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Electrode Quality Map", fontsize=14, pad=20)

    def _create_artifact_page(
        self,
        eeg_data: npt.NDArray,
        artifacts: list[dict[str, Any]],
        results: dict[str, Any],
    ) -> Figure:
        """Create page showing worst artifact examples."""
        # Sort artifacts by severity
        sorted_artifacts = sorted(artifacts, key=lambda x: x["severity"], reverse=True)[:5]

        # Create artifact visualizations
        artifact_fig = create_artifact_examples(
            eeg_data,
            sorted_artifacts,
            results.get("processing_info", {}).get("sampling_rate", 256),
        )

        # Handle case where no artifacts to visualize
        if artifact_fig is None:
            # Create empty figure with message
            artifact_fig = plt.figure(figsize=self.figsize)
            ax = artifact_fig.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "No artifacts to display",
                ha="center",
                va="center",
                fontsize=14,
            )
            ax.axis("off")

        return artifact_fig


def create_electrode_heatmap(
    channel_positions: dict[str, tuple[float, float]], bad_channels: list[str]
) -> Figure:
    """Create electrode heatmap figure.

    Args:
        channel_positions: Dictionary of channel positions
        bad_channels: List of bad channel names

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Normalize positions
    positions = normalize_electrode_positions(channel_positions)

    # Create head outline
    head = patches.Circle((0, 0), 1, fill=False, linewidth=2)
    ax.add_patch(head)

    # Add nose
    nose = patches.Wedge((0, 1), 0.1, 70, 110, fill=False, linewidth=2)
    ax.add_patch(nose)

    # Plot electrodes
    for channel, (x, y) in positions.items():
        if channel in bad_channels:
            color = "red"
            size = 12
        else:
            color = "green"
            size = 10

        ax.plot(x, y, "o", color=color, markersize=size)
        ax.text(x, y - 0.1, channel, ha="center", fontsize=8)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")

    return fig


def create_artifact_examples(
    eeg_data: npt.NDArray, artifacts: list[dict[str, Any]], sampling_rate: int
) -> Figure | None:
    """Create visualization of artifact examples.

    Args:
        eeg_data: EEG data array
        artifacts: List of artifact segments
        sampling_rate: Sampling rate in Hz

    Returns:
        Matplotlib figure or None if no artifacts
    """
    if not artifacts:
        return None

    n_artifacts = min(len(artifacts), 5)
    fig, axes = plt.subplots(n_artifacts, 1, figsize=(10, 2 * n_artifacts))

    if n_artifacts == 1:
        axes = [axes]

    for i, (ax, artifact) in enumerate(zip(axes, artifacts[:n_artifacts], strict=False)):
        # Extract segment
        start_sample = int(artifact["start"] * sampling_rate)
        end_sample = int(artifact["end"] * sampling_rate)

        # Ensure valid range
        start_sample = max(0, start_sample)
        end_sample = min(eeg_data.shape[1], end_sample)

        if start_sample < end_sample:
            segment = eeg_data[:, start_sample:end_sample]
            time = np.arange(segment.shape[1]) / sampling_rate

            # Plot first few channels
            n_channels_to_plot = min(5, segment.shape[0])
            for ch in range(n_channels_to_plot):
                ax.plot(time, segment[ch] * 1e6 + ch * 100, label=f"Ch{ch + 1}")

            ax.set_ylabel("Amplitude (μV)")
            ax.set_title(
                f"Artifact {i + 1}: {artifact['type']} (severity: {artifact['severity']:.2f})"
            )

            if i == n_artifacts - 1:
                ax.set_xlabel("Time (s)")

    plt.tight_layout()
    return fig


def normalize_electrode_positions(
    positions: dict[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    """Normalize electrode positions to -1 to 1 range.

    Args:
        positions: Raw position coordinates

    Returns:
        Normalized positions
    """
    if not positions:
        return {}

    # Extract all coordinates
    x_coords = [pos[0] for pos in positions.values()]
    y_coords = [pos[1] for pos in positions.values()]

    # Find ranges
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Avoid division by zero
    x_range = x_max - x_min if x_max != x_min else 1
    y_range = y_max - y_min if y_max != y_min else 1

    # Normalize to -1 to 1
    normalized = {}
    for channel, (x, y) in positions.items():
        nx = 2 * (x - x_min) / x_range - 1
        ny = 2 * (y - y_min) / y_range - 1
        normalized[channel] = (nx, ny)

    return normalized


def get_banner_color(flag: str) -> str:
    """Get banner color based on triage flag.

    Args:
        flag: Triage flag (URGENT, EXPEDITE, ROUTINE, NORMAL)

    Returns:
        Color string or RGB tuple
    """
    color_map = {
        "URGENT": "red",
        "EXPEDITE": "orange",
        "ROUTINE": "yellow",
        "NORMAL": "green",
    }
    return color_map.get(flag.upper(), "gray")


def severity_to_color(severity: float) -> str:
    """Map severity score to color.

    Args:
        severity: Severity score (0-1)

    Returns:
        Color string
    """
    if severity >= 0.8:
        return "red"
    elif severity >= 0.5:
        return "orange"
    elif severity >= 0.3:
        return "yellow"
    else:
        return "green"
