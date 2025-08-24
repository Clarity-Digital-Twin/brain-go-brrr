"""Async base class with explicit entry points for sync/async contexts.

This module provides a base class for services that need to work in both
sync and async contexts with explicit entry points instead of runtime detection.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, TypeVar

T = TypeVar('T')


class AsyncCapable(ABC):
    """Base class for services that can run in both sync and async contexts.

    Provides explicit entry points:
    - launch() for synchronous contexts
    - launch_async() for async contexts

    This avoids the "event loop already running" problem by being explicit
    about the execution context.
    """

    @abstractmethod
    async def _execute_async(self, *args: Any, **kwargs: Any) -> Any:
        """Internal async implementation to be overridden by subclasses."""
        ...

    def launch(self, *args: Any, **kwargs: Any) -> Any:
        """Synchronous entry point - creates new event loop if needed.

        Use this when calling from synchronous code (scripts, tests, etc).
        """
        try:
            # Try to get existing event loop
            asyncio.get_running_loop()
            # If we get here, loop is running - this is an error
            raise RuntimeError("launch() called from async context. Use launch_async() instead.")
        except RuntimeError:
            # No loop running, create one
            return asyncio.run(self._execute_async(*args, **kwargs))

    async def launch_async(self, *args: Any, **kwargs: Any) -> Any:
        """Asynchronous entry point - uses existing event loop.

        Use this when calling from async code (FastAPI endpoints, async tests).
        """
        return await self._execute_async(*args, **kwargs)


class AsyncAnalyzer(AsyncCapable):
    """Base class for analyzers that may perform async operations."""

    def supports(self, data: Any) -> bool:  # noqa: ARG002
        """Check if this analyzer supports the given data.

        Args:
            data: Input data to check (e.g., MNE Raw object)

        Returns:
            True if this analyzer can process the data
        """
        # Default implementation - override in subclasses
        return True

    @abstractmethod
    async def _execute_async(self, data: Any) -> dict[str, Any]:
        """Perform the analysis asynchronously.

        Args:
            data: Input data to analyze

        Returns:
            Analysis results dictionary
        """
        ...

    def analyze(self, data: Any) -> dict[str, Any]:
        """Synchronous analyze method for backward compatibility."""
        return self.launch(data)

    async def analyze_async(self, data: Any) -> dict[str, Any]:
        """Async analyze method for async contexts."""
        return await self.launch_async(data)
