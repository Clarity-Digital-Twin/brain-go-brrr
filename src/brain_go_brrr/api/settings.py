"""API Settings configuration."""

from pydantic import BaseModel


class APISettings(BaseModel):
    """Settings for the Brain-Go-Brrr API."""

    # Cache configuration
    cache_ttl_seconds: int = 3600  # 1 hour default

    # API configuration
    api_title: str = "Brain-Go-Brrr API"
    api_version: str = "1.0.0"

    # Processing limits
    max_file_size_mb: int = 2048  # 2GB max file size
    max_concurrent_requests: int = 50


# Singleton instance
settings = APISettings()
