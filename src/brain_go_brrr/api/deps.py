"""FastAPI dependency injection for Clean Architecture.

REAL DI. NO DEFAULTS. NO FALLBACKS IN ROUTES.
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Protocol

from fastapi import Depends

from brain_go_brrr.application.factories import create_quality_controller
from brain_go_brrr.application.factories_types import QualityControllerPort


class _NoopQC(Protocol):
    """Minimal protocol surface for QC controller."""

    def run_quality_check(self, raw: Any) -> dict[str, Any]: ...
    def run_full_qc_pipeline(self, raw: Any) -> dict[str, Any]: ...  # Alias for backward compat
    def validate_input(self, raw: Any) -> bool: ...


class _NoopQualityController:
    """Safe fallback so routes that error-early don't fail on import."""

    def run_quality_check(self, raw: Any) -> dict[str, Any]:  # pragma: no cover  # noqa: ARG002
        raise RuntimeError(
            "QC model not configured. Set EEGPT_CKPT_PATH or configure app startup DI."
        )

    def run_full_qc_pipeline(self, raw: Any) -> dict[str, Any]:  # pragma: no cover
        """Alias for backward compatibility with existing routes."""
        return self.run_quality_check(raw)

    def validate_input(self, raw: Any) -> bool:  # pragma: no cover  # noqa: ARG002
        return True  # Let it fail in run_quality_check if actually called


@lru_cache
def get_qc_controller() -> QualityControllerPort | _NoopQC:
    """Get QC controller - real or noop based on env config."""
    ckpt = os.getenv("EEGPT_CKPT_PATH", "data/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt")

    if not ckpt or not Path(ckpt).exists():
        # Don't blow up unless we actually try to use it
        return _NoopQualityController()

    return create_quality_controller(  # type: ignore[return-value]
        model_path=ckpt,
        device=os.getenv("EEGPT_DEVICE", "cpu"),
        enable_logging=True,
        enable_autoreject=True,
    )


# Cache control via DI - clean separation of test/prod behavior
class CacheMode(str, Enum):
    """Cache behavior modes for clean DI control.

    Modes:
    - AUTO: Normal production behavior - use cache when available
    - BYPASS: Skip all cache operations (default for tests)
    - FORCE: Enable cache operations (for cache-specific tests)

    Note: FORCE and AUTO behave identically (both enable cache).
    The distinction allows tests to explicitly request cache usage
    vs relying on default production behavior.
    """

    AUTO = "auto"  # Normal production behavior
    BYPASS = "bypass"  # Skip cache (for most tests)
    FORCE = "force"  # Always use cache (for cache tests)


def get_cache_mode() -> CacheMode:
    """Get cache mode - bypass in tests by default, auto in production.

    This keeps ALL test logic out of production code paths.
    Tests can override this dependency to control caching precisely.
    """
    # Only check for test environment here, in the dependency
    # Production code never needs to know about test environment
    if os.getenv("PYTEST_CURRENT_TEST"):
        return CacheMode.BYPASS
    return CacheMode.AUTO


# Type aliases for dependency injection - NO DEFAULTS IN ROUTES
QCController = Annotated[QualityControllerPort | _NoopQC, Depends(get_qc_controller)]


__all__ = ["QCController", "get_qc_controller", "CacheMode", "get_cache_mode"]
