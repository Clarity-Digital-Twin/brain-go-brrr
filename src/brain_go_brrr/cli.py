"""Command-line interface for Brain Go Brrr."""

from pathlib import Path

import typer
from rich.console import Console

from brain_go_brrr.core.config import Config
from brain_go_brrr.core.logger import get_logger

app = typer.Typer(
    name="brain-go-brrr",
    help="Brain Go Brrr: EEG signal processing and neural representation learning",
    rich_markup_mode="rich",
)

console = Console()
logger = get_logger(__name__)


@app.command()
def train(
    config_file: Path | None = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
    data_path: Path | None = typer.Option(None, "--data", "-d", help="Path to training data"),
    output_dir: Path | None = typer.Option(None, "--output", "-o", help="Output directory"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode"),
) -> None:
    """Train an EEGPT model."""
    console.print("[green]Starting EEGPT training...[/green]")

    # Load configuration
    config = Config(debug=debug)
    if config_file:
        logger.info(f"Loading config from: {config_file}")
    if data_path:
        config.data_dir = data_path
    if output_dir:
        config.output_dir = output_dir

    logger.info("Training configuration loaded: EEGPT")
    logger.info(f"Data directory: {config.data_dir}")
    logger.info(f"Output directory: {config.output_dir}")

    # TODO: Implement training logic
    console.print("[yellow]Training logic not implemented yet[/yellow]")


@app.command()
def preprocess(
    data_path: Path = typer.Argument(..., help="Path to raw EEG data"),
    output_path: Path = typer.Argument(..., help="Path to save preprocessed data"),
    config_file: Path | None = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
) -> None:
    """Preprocess EEG data."""
    console.print("[green]Starting EEG preprocessing...[/green]")

    if config_file:
        logger.info(f"Loading config from: {config_file}")
    logger.info(f"Preprocessing {data_path} -> {output_path}")

    # TODO: Implement preprocessing logic
    console.print("[yellow]Preprocessing logic not implemented yet[/yellow]")


@app.command()
def evaluate(
    model_path: Path = typer.Argument(..., help="Path to trained model"),
    data_path: Path = typer.Argument(..., help="Path to evaluation data"),
    output_path: Path | None = typer.Option(
        None, "--output", "-o", help="Path to save evaluation results"
    ),
) -> None:
    """Evaluate a trained model."""
    console.print("[green]Starting model evaluation...[/green]")

    logger.info(f"Evaluating model: {model_path}")
    logger.info(f"Evaluation data: {data_path}")
    if output_path:
        logger.info(f"Results will be saved to: {output_path}")

    # TODO: Implement evaluation logic
    console.print("[yellow]Evaluation logic not implemented yet[/yellow]")


@app.command()
def serve(
    model_path: Path = typer.Argument(..., help="Path to trained model"),
    host: str = typer.Option("127.0.0.1", help="Host address"),
    port: int = typer.Option(8000, help="Port number"),
) -> None:
    """Serve a trained model via REST API."""
    console.print(f"[green]Starting model server on {host}:{port}[/green]")
    logger.info(f"Loading model from: {model_path}")

    # TODO: Implement serving logic
    console.print("[yellow]Serving logic not implemented yet[/yellow]")


@app.command()
def stream(
    edf_path: Path = typer.Argument(..., help="Path to EDF file to stream"),
    window_size: float = typer.Option(4.0, "--window-size", "-w", help="Window size in seconds"),
    overlap: float = typer.Option(0.0, "--overlap", "-o", help="Overlap fraction (0.0 to 1.0)"),
    output_format: str = typer.Option("json", "--format", "-f", help="Output format (json, csv)"),
    max_windows: int = typer.Option(
        0, "--max-windows", "-n", help="Maximum windows to process (0=all)"
    ),
) -> None:
    """Stream EDF file and extract features in real-time."""
    import json

    from brain_go_brrr.core.config import ModelConfig
    from brain_go_brrr.data.edf_streaming import EDFStreamer
    from brain_go_brrr.models.eegpt_model import EEGPTModel

    console.print(f"[green]Starting EDF streaming from {edf_path}[/green]")

    if not edf_path.exists():
        console.print(f"[red]Error: File not found: {edf_path}[/red]")
        raise typer.Exit(1)

    # Initialize model
    model_config = ModelConfig(device="cpu")
    model = EEGPTModel(config=model_config, auto_load=False)

    # Use mock model for now
    from brain_go_brrr.models.eegpt_architecture import create_eegpt_model

    model.encoder = create_eegpt_model(checkpoint_path=None)
    model.encoder.to(model.device)
    model.is_loaded = True

    # Stream and process
    try:
        with EDFStreamer(edf_path) as streamer:
            info = streamer.get_info()
            console.print(
                f"Duration: {info['duration']:.1f}s, Channels: {info['n_channels']}, SR: {info['sampling_rate']}Hz"
            )

            # Get channel names once
            ch_names = list(streamer._raw.ch_names) if streamer._raw else []

            for window_count, (data_window, start_time) in enumerate(
                streamer.process_in_windows(window_size, overlap), 1
            ):
                # Extract features
                features = model.extract_features(data_window, ch_names)

                # Output result
                result = {
                    "window": window_count,
                    "start_time": float(start_time),
                    "end_time": float(start_time + window_size),
                    "feature_shape": list(features.shape),
                    "feature_mean": float(features.mean()),
                    "feature_std": float(features.std()),
                }

                if output_format == "json":
                    print(json.dumps(result))
                else:
                    console.print(
                        f"Window {window_count}: {start_time:.1f}s - {start_time + window_size:.1f}s"
                    )

                # Check if we've reached the limit
                if max_windows > 0 and window_count >= max_windows:
                    break
    except KeyboardInterrupt:
        # Re-raise to ensure test sees the exception
        logger.info("Streaming interrupted by user")
        raise


@app.command()
def version() -> None:
    """Show version information."""
    from brain_go_brrr import __version__

    console.print(f"Brain Go Brrr version: [bold blue]{__version__}[/bold blue]")


if __name__ == "__main__":
    app()
