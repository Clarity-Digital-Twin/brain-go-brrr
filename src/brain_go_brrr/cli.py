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
    host: str = typer.Option("0.0.0.0", help="Host address"),
    port: int = typer.Option(8000, help="Port number"),
) -> None:
    """Serve a trained model via REST API."""
    console.print(f"[green]Starting model server on {host}:{port}[/green]")
    logger.info(f"Loading model from: {model_path}")

    # TODO: Implement serving logic
    console.print("[yellow]Serving logic not implemented yet[/yellow]")


@app.command()
def version() -> None:
    """Show version information."""
    from brain_go_brrr import __version__

    console.print(f"Brain Go Brrr version: [bold blue]{__version__}[/bold blue]")


if __name__ == "__main__":
    app()
