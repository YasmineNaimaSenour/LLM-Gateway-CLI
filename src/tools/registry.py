"""In-repo tool registry: an explicit, small set of tools the CLI can
expose to a model via `--tools name[,name...]`.

Tool authors declare parameters either as a Pydantic model (preferred —
ergonomic, and Pydantic already gives us `.model_json_schema()` for free)
or as a raw JSON Schema dict (reuses the same schema/model_builder pipeline
already built for structured outputs, so there's exactly one JSON-Schema
subset and one validation path in the whole codebase).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Type, Union
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from ..core.errors import FormatError
from ..core.types import ToolSpec
from ..structured.model_builder import build_model
from ..structured.schema import check_supported_subset

ParametersSpec = Union[Dict[str, Any], Type[BaseModel]]


@dataclass
class RegisteredTool:
    """A tool as known to the runtime: its wire-facing spec, the Pydantic
    model used to validate incoming call arguments, and the Python
    callable that actually does the work."""

    spec: ToolSpec
    model: Type[BaseModel]
    func: Callable[..., Any]


_REGISTRY: Dict[str, RegisteredTool] = {}


def register(name: str, description: str, parameters: ParametersSpec) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: register a function as a tool available to `--tools`."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if isinstance(parameters, type) and issubclass(parameters, BaseModel):
            model = parameters
            schema = model.model_json_schema()
        else:
            schema = parameters
            if schema.get("type") != "object":
                raise FormatError(f"Tool {name!r}: parameters schema must have \"type\": \"object\".")
            check_supported_subset(schema)
            model = build_model(schema, model_name=f"{name.title()}Args")

        _REGISTRY[name] = RegisteredTool(
            spec=ToolSpec(name=name, description=description, parameters=schema),
            model=model,
            func=func,
        )
        return func

    return decorator


def get_tools(names: List[str]) -> List[RegisteredTool]:
    """Resolve tool names to registered tools.

    Raises FormatError for any unknown name. This is a startup-time,
    caller/config check (the CLI validates `--tools` before ever calling a
    provider) — distinct from a model calling an unregistered tool
    mid-conversation, which the ToolExecutor handles as a recoverable,
    model-facing error instead.
    """
    missing = [n for n in names if n not in _REGISTRY]
    if missing:
        available = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise FormatError(f"Unknown tool(s): {missing}. Available tools: {available}.")
    return [_REGISTRY[n] for n in names]


# ---------------------------------------------------------------------------
# example tools
# ---------------------------------------------------------------------------


class _CalculatorArgs(BaseModel):
    a: float = Field(description="First operand.")
    b: float = Field(description="Second operand.")
    operation: str = Field(description="One of: add, subtract, multiply, divide.")


@register("calculator", "Perform a basic arithmetic operation on two numbers.", _CalculatorArgs)
def _calculator(a: float, b: float, operation: str) -> float:
    if operation == "add":
        return a + b
    if operation == "subtract":
        return a - b
    if operation == "multiply":
        return a * b
    if operation == "divide":
        if b == 0:
            raise ValueError("Division by zero.")
        return a / b
    raise ValueError(f"Unknown operation: {operation!r}. Use add, subtract, multiply, or divide.")


class _CurrentTimeArgs(BaseModel):
    timezone: str = Field(default="UTC", description="IANA timezone name, e.g. 'Asia/Tokyo'.")


@register("current_time", "Get the current date and time in a given IANA timezone.", _CurrentTimeArgs)
def _current_time(timezone: str = "UTC") -> str:
    try:
        tz = ZoneInfo(timezone)
    except Exception as exc:
        raise ValueError(f"Unknown timezone: {timezone!r}") from exc
    return datetime.now(tz).isoformat()