"""Markdown Report Generation for EEG Quality Control.

Generates markdown reports that are easy to read and version control friendly.
"""

import logging
from pathlib import Path
from typing import Any

from brain_go_brrr.utils import utc_now

logger = logging.getLogger(__name__)


def generate_markdown_report(qc_results: dict[str, Any]) -> str:
    """Generate a markdown report from QC results.

    Args:
        qc_results: QC analysis results

    Returns:
        Markdown formatted report
    """
    generator = MarkdownReportGenerator()
    return generator.generate_report(qc_results)


def convert_results_to_markdown(qc_results: dict[str, Any]) -> str:
    """Convert QC results to markdown format.

    Args:
        qc_results: QC analysis results

    Returns:
        Markdown formatted report
    """
    return generate_markdown_report(qc_results)


class MarkdownReportGenerator:
    """Generate markdown reports for EEG QC analysis."""

    def generate_report(self, results: dict[str, Any]) -> str:
        """Generate complete markdown report.

        Args:
            results: QC analysis results

        Returns:
            Markdown formatted report
        """
        sections = []

        # Add header
        sections.append(self._create_header(results))

        # Add warning banner if needed
        warning = self._create_warning_banner(results)
        if warning:
            sections.append(warning)

        # Add file information
        sections.append(self._create_file_info(results))

        # Add summary statistics
        sections.append(self._create_summary_stats(results))

        # Add channel quality section
        sections.append(self._create_channel_quality(results))

        # Add electrode map
        sections.append(self._create_electrode_map(results))

        # Add artifact summary
        sections.append(self._create_artifact_summary(results))

        # Add footer
        sections.append(self._create_footer())

        return "\n\n".join(sections)

    def save_report(self, results: dict[str, Any], output_path: Path) -> None:
        """Save markdown report to file.

        Args:
            results: QC analysis results
            output_path: Path to save the report
        """
        markdown = self.generate_report(results)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown)
        logger.info(f"Markdown report saved to {output_path}")

    def _create_header(self, results: dict[str, Any]) -> str:  # noqa: ARG002
        """Create report header."""
        return "# EEG Quality Control Report"

    def _create_warning_banner(self, results: dict[str, Any]) -> str | None:
        """Create warning banner for abnormal EEGs."""
        quality_metrics = results.get('quality_metrics', {})
        abnormality_score = quality_metrics.get('abnormality_score', 0)
        quality_grade = quality_metrics.get('quality_grade', 'UNKNOWN')

        if abnormality_score > 0.8 or quality_grade == 'POOR':
            emoji = get_triage_emoji("URGENT")
            return f"> {emoji} **URGENT - Expedite read**\n> \n> This EEG shows significant abnormalities and requires immediate review."
        elif abnormality_score > 0.6 or quality_grade == 'FAIR':
            emoji = get_triage_emoji("EXPEDITE")
            return f"> {emoji} **EXPEDITE - Priority review recommended**\n> \n> This EEG shows moderate abnormalities."
        elif abnormality_score > 0.4:
            emoji = get_triage_emoji("ROUTINE")
            return f"> {emoji} **ROUTINE - Standard workflow**\n> \n> This EEG shows minor irregularities."
        else:
            emoji = get_triage_emoji("NORMAL")
            return f"> {emoji} **NORMAL - Good quality EEG**\n> \n> No significant abnormalities detected."

    def _create_file_info(self, results: dict[str, Any]) -> str:
        """Create file information section."""
        processing_info = results.get('processing_info', {})

        lines = ["## File Information"]
        lines.append("")
        lines.append(f"- **File**: {processing_info.get('file_name', 'Unknown')}")
        lines.append(f"- **Duration**: {processing_info.get('duration_seconds', 0):.1f} seconds")
        lines.append(f"- **Sampling Rate**: {processing_info.get('sampling_rate', 0)} Hz")
        lines.append(f"- **Timestamp**: {processing_info.get('timestamp', utc_now().isoformat())}")

        return "\n".join(lines)

    def _create_summary_stats(self, results: dict[str, Any]) -> str:
        """Create summary statistics section."""
        quality_metrics = results.get('quality_metrics', {})

        lines = ["## Summary Statistics"]
        lines.append("")

        bad_channels = quality_metrics.get('bad_channels', [])
        bad_ratio = quality_metrics.get('bad_channel_ratio', 0)
        abnormality = quality_metrics.get('abnormality_score', 0)
        quality_grade = quality_metrics.get('quality_grade', 'UNKNOWN')

        lines.append(f"- **Quality Grade**: {quality_grade}")
        lines.append(f"- **Bad Channels**: {len(bad_channels)} ({bad_ratio*100:.1f}%)")
        lines.append(f"- **Abnormality Score**: {abnormality:.2f}")

        if bad_channels:
            lines.append(f"- **Bad Channel List**: {', '.join(bad_channels)}")

        return "\n".join(lines)

    def _create_channel_quality(self, results: dict[str, Any]) -> str:
        """Create channel quality table."""
        quality_metrics = results.get('quality_metrics', {})
        bad_channels = set(quality_metrics.get('bad_channels', []))
        channel_positions = quality_metrics.get('channel_positions', {})

        lines = ["## Channel Quality"]
        lines.append("")
        lines.append("| Channel | Status |")
        lines.append("|---------|--------|")

        # Get all channels from positions or use standard set
        if channel_positions:
            channels = sorted(channel_positions.keys())
        else:
            # Use a standard set of channels
            channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
                       'O1', 'O2', 'T3', 'T4', 'F7', 'F8', 'T5', 'T6', 'Fz', 'Cz', 'Pz']

        for channel in channels:
            if channel in bad_channels:
                lines.append(f"| {channel} | ❌ Bad |")
            else:
                lines.append(f"| {channel} | ✅ Good |")

        return "\n".join(lines)

    def _create_electrode_map(self, results: dict[str, Any]) -> str:
        """Create ASCII electrode map."""
        quality_metrics = results.get('quality_metrics', {})
        bad_channels = set(quality_metrics.get('bad_channels', []))

        lines = ["## Electrode Map"]
        lines.append("")
        lines.append("```")
        lines.append("      Fp1---Fp2")
        lines.append("       |     |")
        lines.append("    F7-F3---F4-F8")
        lines.append("     | |     | |")
        lines.append("   T3--C3---C4--T4")
        lines.append("     | |     | |")
        lines.append("    T5-P3---P4-T6")
        lines.append("       |     |")
        lines.append("      O1----O2")
        lines.append("")
        lines.append("Legend: ❌ = Bad Channel")

        # Add bad channel indicators
        if bad_channels:
            lines.append("Bad channels marked above: " + ", ".join(f"{ch}(❌)" for ch in sorted(bad_channels)))

        lines.append("```")

        return "\n".join(lines)

    def _create_artifact_summary(self, results: dict[str, Any]) -> str:
        """Create artifact summary section."""
        quality_metrics = results.get('quality_metrics', {})
        artifacts = quality_metrics.get('artifact_segments', [])

        lines = ["## Detected Artifacts"]
        lines.append("")

        if not artifacts:
            lines.append("No significant artifacts detected.")
        else:
            # Sort by severity
            sorted_artifacts = sorted(artifacts, key=lambda x: x['severity'], reverse=True)

            lines.append("| Time (s) | Type | Severity |")
            lines.append("|----------|------|----------|")

            for artifact in sorted_artifacts[:10]:  # Show top 10
                start = artifact['start']
                end = artifact['end']
                artifact_type = artifact['type']
                severity = artifact['severity']

                lines.append(f"| {start:.1f}-{end:.1f} | {artifact_type} | {severity:.2f} |")

            if len(artifacts) > 10:
                lines.append(f"\n*... and {len(artifacts) - 10} more artifacts*")

        return "\n".join(lines)

    def _create_footer(self) -> str:
        """Create report footer."""
        lines = ["---"]
        lines.append(f"*Generated on {utc_now().strftime('%Y-%m-%d %H:%M:%S UTC')} by Brain-Go-Brrr*")
        return "\n".join(lines)


def get_triage_emoji(flag: str) -> str:
    """Get emoji for triage flag.

    Args:
        flag: Triage flag (URGENT, EXPEDITE, ROUTINE, NORMAL)

    Returns:
        Appropriate emoji
    """
    emoji_map = {
        'URGENT': '🚨',
        'EXPEDITE': '⚠️',
        'ROUTINE': '📋',
        'NORMAL': '✅'
    }
    return emoji_map.get(flag.upper(), '📄')
