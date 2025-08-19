"""API Settings configuration with environment variable support."""

from pydantic_settings import BaseSettings


class APISettings(BaseSettings):
    """Settings for the Brain-Go-Brrr API.

    Environment variables can override defaults:
    - BGBR_CACHE_TTL_SECONDS
    - BGBR_API_TITLE
    - BGBR_API_VERSION
    - BGBR_MAX_FILE_SIZE_MB
    - BGBR_MAX_CONCURRENT_REQUESTS
    """

    # Cache configuration
    cache_ttl_seconds: int = 3600  # 1 hour default

    # API configuration
    api_title: str = "Brain-Go-Brrr API"
    api_version: str = "1.0.0"

    # Processing limits
    max_file_size_mb: int = 2048  # 2GB max file size
    max_concurrent_requests: int = 50

    class Config:
        """Pydantic config."""

        env_prefix = "BGBR_"  # All env vars start with BGBR_
        case_sensitive = False


# Singleton instance - will read from environment if available
settings = APISettings()
