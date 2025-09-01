"""Configuration management using Pydantic and Hydra."""

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseModel):
    """Configuration for EEGPT model."""

    # EEGPT Model
    model_path: Path = Field(
        default_factory=lambda: Path(
            "data/models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"
        ),
        description="Path to EEGPT pretrained checkpoint",
    )
    device: str = Field(default="auto", description="Device for model inference (auto, cpu, cuda)")
    batch_size: int = Field(default=8, description="Batch size for inference")

    # Model parameters (from EEGPT paper)
    sampling_rate: int = Field(default=256, description="Target sampling rate in Hz")
    window_duration: float = Field(default=4.0, description="Window duration in seconds")
    patch_size: int = Field(default=64, description="Patch size in samples")
    n_summary_tokens: int = Field(default=4, description="Number of summary tokens (S=4)")
    embed_dim: int = Field(default=512, description="Embedding dimension")

    # Streaming configuration
    streaming_threshold: float = Field(
        default=120.0, description="Duration threshold for streaming (seconds)"
    )
    window_overlap: float = Field(default=0.5, description="Window overlap ratio for streaming")

    @property
    def window_samples(self) -> int:
        """Calculate window size in samples."""
        samples = self.window_duration * self.sampling_rate
        if not samples.is_integer():
            raise ValueError("Window duration must result in integer samples")
        return int(samples)

    @field_validator("model_path")
    @classmethod
    def validate_model_path(cls, v: Path) -> Path:
        """Validate that model path exists."""
        if not v.exists():
            raise ValueError(f"EEGPT model checkpoint not found: {v}")
        return v


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

    @property
    def sleep_edf_version(self) -> str:
        """Get Sleep-EDF version from env or default."""
        return os.environ.get("BGB_SLEEP_EDF_VERSION", "sleep-edf-database-expanded-1.0.0")

    @property
    def sleep_edf_root(self) -> Path:
        """Get Sleep-EDF root directory with env override."""
        # Check explicit override first
        override = os.environ.get("BGB_SLEEP_EDF_DIR")
        if override:
            return Path(override)

        # Use data_path (which already exists in this class!)
        base = self.data_path / "datasets" / "sleep-edf" / self.sleep_edf_version
        if base.exists():
            return base

        # Legacy fallback (temporary)
        legacy = self.data_path / "datasets" / "external" / "sleep-edf"
        return legacy

    @property
    def sleep_edf_cassette_dir(self) -> Path:
        """Get Sleep-EDF cassette directory."""
        return self.sleep_edf_root / "sleep-cassette"

    def get_sleep_edf_psg_file(self, explicit: str | None = None) -> Path | None:
        """Get a PSG file deterministically.

        Args:
            explicit: Optional explicit file path to use

        Returns:
            Path to PSG file or None if not found
        """
        if explicit or os.environ.get("BGB_SLEEP_EDF_FILE"):
            path_str = explicit or os.environ.get("BGB_SLEEP_EDF_FILE", "")
            if path_str:
                p = Path(path_str)
                return p if p.exists() else None

        # Get first file sorted (deterministic)
        if not self.sleep_edf_cassette_dir.exists():
            return None

        files = sorted(self.sleep_edf_cassette_dir.glob("*-PSG.edf"))
        # Filter out macOS resource forks
        files = [f for f in files if not f.name.startswith("._")]
        return files[0] if files else None

    @property
    def tuab_root(self) -> Path:
        """Get TUAB dataset root directory with env override."""
        override = os.environ.get("BGB_TUAB_DIR")
        if override:
            return Path(override)

        # Standard location under data root
        return self.data_path / "datasets" / "tuab"

    @property
    def tuev_root(self) -> Path:
        """Get TUEV dataset root directory with env override."""
        override = os.environ.get("BGB_TUEV_DIR")
        if override:
            return Path(override)

        # Standard location under data root
        return self.data_path / "datasets" / "tuev"

    @property
    def tuab_version(self) -> str:
        """Get TUAB version from env or default."""
        return os.environ.get("BGB_TUAB_VERSION", "")

    @property
    def tuev_version(self) -> str:
        """Get TUEV version from env or default."""
        return os.environ.get("BGB_TUEV_VERSION", "")

    def get_tuab_sample_file(
        self, split: str = "train", label: str = "abnormal", protocol: str = "01_tcp_ar"
    ) -> Path | None:
        """Get a TUAB EDF file deterministically.

        Args:
            split: Dataset split (train/eval/test)
            label: Label type (normal/abnormal)
            protocol: Protocol name (default: 01_tcp_ar)

        Returns:
            Path to EDF file or None if not found
        """
        # Check for explicit override
        explicit = os.environ.get("BGB_TUAB_FILE")
        if explicit:
            p = Path(explicit)
            return p if p.exists() else None

        # Construct path to TUAB data
        # Note: TUAB doesn't have version subdirectory in actual structure
        if self.tuab_version:
            base = self.tuab_root / self.tuab_version / "edf" / split / label / protocol
        else:
            base = self.tuab_root / "edf" / split / label / protocol
        if not base.exists():
            return None

        # Get first file sorted (deterministic)
        files = sorted(base.glob("*.edf"))
        # Filter out macOS resource forks
        files = [f for f in files if not f.name.startswith("._")]
        return files[0] if files else None

    def get_tuev_sample_file(self, event_type: str = "bckg") -> Path | None:
        """Get a TUEV EDF file deterministically.

        Args:
            event_type: Event type directory (e.g., 'bckg', 'gped', 'pled')

        Returns:
            Path to EDF file or None if not found
        """
        # Check for explicit override
        explicit = os.environ.get("BGB_TUEV_FILE")
        if explicit:
            p = Path(explicit)
            return p if p.exists() else None

        # Construct path to TUEV data
        # Note: TUEV doesn't have version subdirectory in actual structure
        if self.tuev_version:
            base = self.tuev_root / self.tuev_version / "edf" / event_type
        else:
            # Try looking in eval subdirectory first (common TUEV structure)
            base = self.tuev_root / "edf" / "eval"
            if base.exists():
                # Find first directory with matching event type
                for subdir in sorted(base.iterdir()):
                    if subdir.is_dir():
                        event_files = sorted(subdir.glob(f"{event_type}*.edf"))
                        if event_files:
                            return event_files[0]
            # Fallback to direct path
            base = self.tuev_root / "edf" / event_type
        if not base.exists():
            return None

        # Get first file sorted (deterministic)
        files = sorted(base.glob("*.edf"))
        # Filter out macOS resource forks
        files = [f for f in files if not f.name.startswith("._")]
        return files[0] if files else None


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

    # Proper pydantic-settings v2 configuration
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization setup."""
        # Create necessary directories
        self.data_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)

        # Update data config with absolute paths
        self.data.data_path = self.data_dir
