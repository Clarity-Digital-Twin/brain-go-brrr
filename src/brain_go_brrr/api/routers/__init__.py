"""API routers for Brain-Go-Brrr application.

Note: Routers are not imported here to avoid heavy dependencies (torch, etc.)
Import routers directly where needed:
    from brain_go_brrr.api.routers import health
    from brain_go_brrr.api.routers import eegpt  # Only when torch is needed
"""

__all__ = ["cache", "eegpt", "health", "jobs", "qc", "queue", "resources", "sleep"]
