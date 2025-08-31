"""Tests for async_base module - REAL BEHAVIORAL TESTS, NO MOCKING."""

import asyncio
from typing import Any

import pytest

from brain_go_brrr.application.async_base import AsyncAnalyzer, AsyncCapable


class ConcreteAsyncCapable(AsyncCapable):
    """Concrete implementation for testing."""

    def __init__(self):
        """Initialize the concrete implementation."""
        self.executed = False
        self.args_received = None
        self.kwargs_received = None

    async def _execute_async(self, *args: Any, **kwargs: Any) -> str:
        """Simple async implementation that tracks execution."""
        self.executed = True
        self.args_received = args
        self.kwargs_received = kwargs
        await asyncio.sleep(0.001)  # Simulate async work
        return f"result: {args}, {kwargs}"


class ConcreteAnalyzer(AsyncAnalyzer):
    """Concrete analyzer for testing."""

    def __init__(self, supported: bool = True):
        """Initialize the analyzer with support flag."""
        self._supported = supported
        self.data_analyzed = None

    def supports(self, data: Any) -> bool:
        """Check support based on config."""
        return self._supported

    async def _execute_async(self, data: Any) -> dict[str, Any]:
        """Analyze data and return results."""
        self.data_analyzed = data
        await asyncio.sleep(0.001)  # Simulate async work
        return {"analyzed": data, "status": "complete"}


class TestAsyncCapable:
    """Test AsyncCapable base class BEHAVIOR."""

    def test_launch_from_sync_context(self):
        """Test launch() creates event loop in sync context."""
        obj = ConcreteAsyncCapable()
        result = obj.launch("arg1", "arg2", key="value")

        assert obj.executed is True
        assert obj.args_received == ("arg1", "arg2")
        assert obj.kwargs_received == {"key": "value"}
        assert result == "result: ('arg1', 'arg2'), {'key': 'value'}"

    def test_launch_async_from_async_context(self):
        """Test launch_async() uses existing event loop."""
        obj = ConcreteAsyncCapable()
        # Run in new event loop
        import asyncio

        async def run_test():
            result = await obj.launch_async("async_arg", async_key="async_value")
            return result, obj

        result, obj = asyncio.run(run_test())
        assert obj.executed is True
        assert obj.args_received == ("async_arg",)
        assert obj.kwargs_received == {"async_key": "async_value"}
        assert result == "result: ('async_arg',), {'async_key': 'async_value'}"

    def test_launch_from_async_context_raises(self):
        """Test launch() raises error when called from async context."""
        import asyncio
        import warnings

        async def run_test():
            obj = ConcreteAsyncCapable()
            # The actual error we get is different - asyncio.run() can't be called
            # Suppress the "coroutine was never awaited" warning since we're testing error case
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                with pytest.raises(
                    RuntimeError, match="asyncio.run\\(\\) cannot be called from a running event loop"
                ):
                    obj.launch("should_fail")

        asyncio.run(run_test())

    def test_multiple_sync_launches(self):
        """Test multiple sync launches work correctly."""
        obj = ConcreteAsyncCapable()

        # First launch
        result1 = obj.launch(1)
        assert result1 == "result: (1,), {}"

        # Second launch - should work fine
        result2 = obj.launch(2)
        assert result2 == "result: (2,), {}"

        # Verify state is updated
        assert obj.args_received == (2,)


class TestAsyncAnalyzer:
    """Test AsyncAnalyzer behavior."""

    def test_supports_default_implementation(self):
        """Test default supports() returns True."""
        analyzer = ConcreteAnalyzer()
        assert analyzer.supports("any_data") is True

    def test_supports_custom_implementation(self):
        """Test custom supports() logic."""
        analyzer = ConcreteAnalyzer(supported=False)
        assert analyzer.supports("any_data") is False

    def test_analyze_sync_method(self):
        """Test analyze() method for backward compatibility."""
        analyzer = ConcreteAnalyzer()
        data = {"eeg": "raw_data"}
        result = analyzer.analyze(data)

        assert analyzer.data_analyzed == data
        assert result == {"analyzed": data, "status": "complete"}

    def test_analyze_async_method(self):
        """Test analyze_async() in async context."""
        import asyncio

        async def run_test():
            analyzer = ConcreteAnalyzer()
            data = {"eeg": "async_data"}
            result = await analyzer.analyze_async(data)
            return analyzer, result, data

        analyzer, result, data = asyncio.run(run_test())
        assert analyzer.data_analyzed == data
        assert result == {"analyzed": data, "status": "complete"}

    def test_analyzer_inheritance_chain(self):
        """Test analyzer properly inherits from AsyncCapable."""
        analyzer = ConcreteAnalyzer()

        # Should have both launch methods
        assert hasattr(analyzer, "launch")
        assert hasattr(analyzer, "launch_async")

        # Should work as AsyncCapable
        result = analyzer.launch({"test": "data"})
        assert result == {"analyzed": {"test": "data"}, "status": "complete"}


class TestAsyncCapableEdgeCases:
    """Test edge cases and error conditions."""

    def test_abstract_class_cannot_instantiate(self):
        """Test abstract classes cannot be instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            AsyncCapable()  # type: ignore[abstract]

    def test_abstract_analyzer_cannot_instantiate(self):
        """Test abstract analyzer cannot be instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            AsyncAnalyzer()  # type: ignore[abstract]

    def test_concurrent_async_executions(self):
        """Test multiple concurrent async executions."""
        import asyncio

        async def run_test():
            obj1 = ConcreteAsyncCapable()
            obj2 = ConcreteAsyncCapable()

            # Launch both concurrently
            results = await asyncio.gather(
                obj1.launch_async("obj1"),
                obj2.launch_async("obj2"),
            )
            return results, obj1, obj2

        results, obj1, obj2 = asyncio.run(run_test())
        assert results[0] == "result: ('obj1',), {}"
        assert results[1] == "result: ('obj2',), {}"
        assert obj1.args_received == ("obj1",)
        assert obj2.args_received == ("obj2",)

    def test_exception_propagation_sync(self):
        """Test exceptions propagate through sync launch."""

        class FailingCapable(AsyncCapable):
            async def _execute_async(self, *args: Any, **kwargs: Any) -> Any:
                raise ValueError("Intentional failure")

        obj = FailingCapable()
        with pytest.raises(ValueError, match="Intentional failure"):
            obj.launch()

    def test_exception_propagation_async(self):
        """Test exceptions propagate through async launch."""
        import asyncio

        class FailingCapable(AsyncCapable):
            async def _execute_async(self, *args: Any, **kwargs: Any) -> Any:
                raise ValueError("Async failure")

        async def run_test():
            obj = FailingCapable()
            with pytest.raises(ValueError, match="Async failure"):
                await obj.launch_async()

        asyncio.run(run_test())
