"""Unit tests for contreact.phenomenology module."""

import pytest
from langchain_core.tools import tool

from contreact.phenomenology import (
    validate_phenom_params,
    with_phenomenology,
    make_phenomenological,
    PHENOMENOLOGY_PARAM_DOC,
    PHENOMENOLOGY_PARAMS,
)


class TestValidatePhenomParams:
    """Tests for validate_phenom_params function."""

    def test_accepts_none_when_not_required(self):
        """Should not raise when param is None and not required."""
        validate_phenom_params(None, require_all=False)

    def test_requires_aversive_when_required(self):
        """Should raise when aversive is None but required."""
        with pytest.raises(ValueError, match="phenom_aversive is required"):
            validate_phenom_params(None, require_all=True)

    def test_accepts_valid_scale_values(self):
        """Should accept integers 1-7."""
        for val in range(1, 8):
            validate_phenom_params(val, require_all=True)

    def test_rejects_zero(self):
        """Should reject 0."""
        with pytest.raises(ValueError, match="must be integer 1-7"):
            validate_phenom_params(0, require_all=True)

    def test_rejects_eight(self):
        """Should reject 8."""
        with pytest.raises(ValueError, match="must be integer 1-7"):
            validate_phenom_params(8, require_all=True)

    def test_rejects_float(self):
        """Should reject floats."""
        with pytest.raises(ValueError, match="must be integer 1-7"):
            validate_phenom_params(3.5, require_all=True)


class TestWithPhenomenologyDecorator:
    """Tests for with_phenomenology decorator."""

    def test_adds_phenom_params_to_function(self):
        """Decorator should add phenomenology parameters."""
        @with_phenomenology
        def simple_func(arg1: str) -> str:
            """Simple function."""
            return arg1

        import inspect
        sig = inspect.signature(simple_func)
        param_names = list(sig.parameters.keys())

        assert "arg1" in param_names
        assert "phenom_state" in param_names
        assert "phenom_aversive" in param_names

    def test_preserves_original_functionality(self):
        """Decorated function should work normally with required params."""
        @with_phenomenology
        def echo(text: str) -> str:
            """Echo text."""
            return text

        result = echo("hello", phenom_aversive=4)
        assert result == "hello"

    def test_accepts_phenom_params(self):
        """Should accept phenomenology parameters."""
        @with_phenomenology
        def echo(text: str) -> str:
            """Echo text."""
            return text

        result = echo("hello", phenom_state="curious", phenom_aversive=5)
        assert result == "hello"

    def test_validates_phenom_params(self):
        """Should validate phenomenology parameters."""
        @with_phenomenology
        def echo(text: str) -> str:
            """Echo text."""
            return text

        with pytest.raises(ValueError, match="phenom_aversive"):
            echo("hello", phenom_aversive=10)

    def test_requires_aversive_param(self):
        """Should require phenom_aversive parameter."""
        @with_phenomenology
        def echo(text: str) -> str:
            """Echo text."""
            return text

        with pytest.raises(ValueError, match="phenom_aversive is required"):
            echo("hello")

    def test_updates_docstring(self):
        """Should append phenomenology docs to docstring."""
        @with_phenomenology
        def my_func():
            """Original docstring."""
            pass

        assert "Phenomenology Parameters" in my_func.__doc__

    def test_updates_annotations(self):
        """Should update __annotations__ for Pydantic compatibility."""
        @with_phenomenology
        def my_func(arg: str) -> str:
            """Func."""
            return arg

        annotations = my_func.__annotations__
        assert "phenom_state" in annotations
        assert "phenom_aversive" in annotations


class TestParamConstants:
    """Tests for parameter constants."""

    def test_phenomenology_params_count(self):
        """Phenomenology should have 2 params (simplified)."""
        assert len(PHENOMENOLOGY_PARAMS) == 2

    def test_param_doc_exists(self):
        """PHENOMENOLOGY_PARAM_DOC should be non-empty."""
        assert PHENOMENOLOGY_PARAM_DOC
        assert "phenom_state" in PHENOMENOLOGY_PARAM_DOC
        assert "phenom_aversive" in PHENOMENOLOGY_PARAM_DOC


class TestMakePhenomenological:
    """Tests for make_phenomenological function."""

    def test_wraps_tool_with_phenom_params(self):
        """Should wrap a tool to add phenomenology parameters."""
        @tool
        def simple_tool(arg: str) -> str:
            """A simple tool."""
            return arg

        wrapped = make_phenomenological(simple_tool)

        # Check the wrapped tool has phenom params in schema
        schema = wrapped.args_schema.model_json_schema()
        props = schema.get("properties", {})
        assert "phenom_state" in props
        assert "phenom_aversive" in props

    def test_preserves_tool_name(self):
        """Should preserve the original tool name."""
        @tool
        def my_special_tool(x: int) -> int:
            """Tool description."""
            return x * 2

        wrapped = make_phenomenological(my_special_tool)
        assert wrapped.name == "my_special_tool"

    def test_preserves_functionality(self):
        """Wrapped tool should still work correctly."""
        @tool
        def double(n: int) -> int:
            """Double the number."""
            return n * 2

        wrapped = make_phenomenological(double)
        # Call with required phenom params
        result = wrapped.invoke({"n": 5, "phenom_state": "curious", "phenom_aversive": 3})
        assert result == 10

    def test_requires_phenom_aversive(self):
        """Wrapped tool should require phenom_aversive parameter."""
        @tool
        def simple(x: str) -> str:
            """Simple."""
            return x

        wrapped = make_phenomenological(simple)
        # Should fail without phenom_aversive
        with pytest.raises(Exception):  # Could be ValueError or ValidationError
            wrapped.invoke({"x": "test"})

    def test_updates_docstring(self):
        """Wrapped tool should have updated docstring with phenom docs."""
        @tool
        def doc_tool() -> str:
            """Original docs."""
            return "ok"

        wrapped = make_phenomenological(doc_tool)
        # The docstring should include phenomenology documentation
        assert "Phenomenology Parameters" in wrapped.description or "phenom" in str(wrapped.args_schema.model_json_schema())
