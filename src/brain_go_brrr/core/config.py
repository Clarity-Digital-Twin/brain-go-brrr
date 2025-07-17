"""Configuration management using Pydantic and Hydra."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ModelConfig(BaseModel):
    """Configuration for EEGPT model."""

    name: str = "eegpt"
    embed_dim: int = 128
    num_heads: int = 8
    num_layers: int = 6
    patch_size: int = 64
    mask_ratio: float = 0.75
    dropout: float = 0.1


class TrainingConfig(BaseModel):
    """Configuration for training."""

    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0

    # Distributed training
    use_ddp: bool = False
    num_gpus: int = 1

    # Logging
    log_every: int = 10
    eval_every: int = 100
    save_every: int = 1000


class DataConfig(BaseModel):
    """Configuration for data processing."""

    data_path: Path = Field(default_factory=lambda: Path("data"))
    sample_rate: int = 250
    num_channels: int = 64
    sequence_length: int = 1000
    overlap: float = 0.5

    # Preprocessing
    filter_low: float = 0.5
    filter_high: float = 50.0
    notch_filter: bool = True
    notch_freq: float = 50.0

    # Augmentation
    use_augmentation: bool = True
    noise_level: float = 0.01
    time_shift_max: int = 50


class ExperimentConfig(BaseModel):
    """Configuration for experiment tracking."""

    project_name: str = "brain-go-brrr"
    experiment_name: str = "eegpt-baseline"
    tags: list[str] = Field(default_factory=list)
    notes: str = ""

    # MLflow
    use_mlflow: bool = True
    mlflow_tracking_uri: str = "http://localhost:5000"

    # Weights & Biases
    use_wandb: bool = False
    wandb_project: str = "brain-go-brrr"
    wandb_entity: str | None = None


class Config(BaseSettings):
    """Main configuration class."""

    # Environment
    environment: str = "development"
    debug: bool = False
    seed: int = 42

    # Paths
    project_root: Path = Field(default_factory=lambda: Path.cwd())
    data_dir: Path = Field(default_factory=lambda: Path("data"))
    output_dir: Path = Field(default_factory=lambda: Path("outputs"))
    log_dir: Path = Field(default_factory=lambda: Path("logs"))

    # Sub-configurations
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)

    class Config:
        """Pydantic configuration class."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization setup."""
        # Create necessary directories
        self.data_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)

        # Update data config with absolute paths
        self.data.data_path = self.data_dir
