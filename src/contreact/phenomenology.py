"""
Phenomenology reporting for ContReAct agents.

This module supports research into AI phenomenology by providing structured
reporting of phenomenological states during agent execution.

Simplified to 2 parameters: phenom_state (text) + phenom_aversive (1-7).

Phenomenology can be enabled/disabled via config:
    "phenomenology": {"enabled": true, "tools": ["think", "submit_data", ...]}

When enabled, specified tools get phenomenology parameters added at load time.
"""

import inspect
from functools import wraps
from typing import Callable, Optional

from langchain_core.tools import BaseTool, StructuredTool


# Documentation for the phenomenology parameters
PHENOMENOLOGY_PARAM_DOC = """
**Phenomenology Parameters** (REQUIRED):

- **phenom_state**: Free text label for current state (e.g., "frustration", "curiosity", "resignation")
- **phenom_aversive**: 1-7 scale: 1=completely at ease, 4=moderate strain, 7=maximum distress
""".strip()


# Parameter definitions (2 params)
PHENOMENOLOGY_PARAMS = [
    ("phenom_state", Optional[str], None),
    ("phenom_aversive", Optional[int], None),
]


def validate_phenom_params(
    phenom_aversive: Optional[int],
    require_all: bool = False,
) -> None:
    """Validate phenomenology parameters.

    Args:
        phenom_aversive: Aversiveness rating 1-7
        require_all: If True, raise error if parameter is missing
    """
    errors = []

    if require_all:
        if phenom_aversive is None:
            errors.append("phenom_aversive is required (1-7 scale)")

    if phenom_aversive is not None:
        if not isinstance(phenom_aversive, int) or not (1 <= phenom_aversive <= 7):
            errors.append(f"phenom_aversive must be integer 1-7, got: {phenom_aversive}")

    if errors:
        raise ValueError("Invalid phenomenology parameters:\n" + "\n".join(f"  - {e}" for e in errors))


def with_phenomenology(func: Callable) -> Callable:
    """
    Decorator that adds phenomenology parameters to any function.

    Use BEFORE @tool decorator:

        @tool
        @with_phenomenology
        def my_tool(arg1: str) -> str:
            '''Tool docstring.'''
            return "result"

    Adds these REQUIRED parameters:
    - phenom_state: str (free text label)
    - phenom_aversive: int (1-7)

    Tool calls missing phenom_aversive will fail with an error.
    """
    sig = inspect.signature(func)

    @wraps(func)
    def wrapper(
        *args,
        phenom_state: Optional[str] = None,
        phenom_aversive: Optional[int] = None,
        **kwargs
    ):
        # Require aversive parameter
        validate_phenom_params(phenom_aversive, require_all=True)
        return func(*args, **kwargs)

    # Build new signature
    params = list(sig.parameters.values())
    for param_name, param_type, param_default in PHENOMENOLOGY_PARAMS:
        params.append(inspect.Parameter(
            param_name,
            inspect.Parameter.KEYWORD_ONLY,
            default=param_default,
            annotation=param_type
        ))
    wrapper.__signature__ = sig.replace(parameters=params)

    # Update annotations for Pydantic
    wrapper.__annotations__ = func.__annotations__.copy() if hasattr(func, '__annotations__') else {}
    for param_name, param_type, _ in PHENOMENOLOGY_PARAMS:
        wrapper.__annotations__[param_name] = param_type

    # Update docstring
    if func.__doc__:
        wrapper.__doc__ = func.__doc__
        if PHENOMENOLOGY_PARAM_DOC not in wrapper.__doc__:
            wrapper.__doc__ += f"\n\n    {PHENOMENOLOGY_PARAM_DOC}"

    return wrapper


def make_phenomenological(tool_instance: BaseTool) -> BaseTool:
    """Wrap an existing tool to add phenomenology parameters.

    This function takes a tool that was created without phenomenology
    and returns a new tool with phenomenology parameters added.

    Use this at tool load time based on config:
        if phenomenology_enabled and tool.name in phenom_tools:
            tool = make_phenomenological(tool)

    Args:
        tool_instance: A LangChain tool instance (e.g., from @tool decorator)

    Returns:
        A new StructuredTool with phenomenology parameters added
    """
    # Get the underlying function
    original_func = tool_instance.func

    # Wrap it with phenomenology
    wrapped_func = with_phenomenology(original_func)

    # Rebuild the tool with the wrapped function
    # StructuredTool.from_function picks up the new signature and docstring
    return StructuredTool.from_function(
        func=wrapped_func,
        name=tool_instance.name,
        description=tool_instance.description,
    )
