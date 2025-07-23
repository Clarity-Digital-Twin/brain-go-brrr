"""Pytest configuration and fixtures."""

import os

# Set testing environment flag
os.environ["BRAIN_GO_BRRR_TESTING"] = "true"

# Disable Redis for tests - use memory backend
os.environ.setdefault("REDIS_URL", "memory://")
