"""In-repo tool registry and execution for M2 tool calling.

Not a plugin system: tools are Python functions registered here via
`register()`. Anything more dynamic (discovery, MCP-sourced tools) is out
of scope for this milestone.

Public surface: `register`/`get_tools`/`tool_names` operate on the default
registry; `ToolRegistry` and `RegisteredTool` are the instance-level and
record types; `ToolExecutor` runs a model's ToolCalls against a fixed tool
set. See registry.py for the isolation story (audit #8).
"""

from .executor import ToolExecutor
from .registry import (
    DEFAULT_REGISTRY,
    ParametersSpec,
    RegisteredTool,
    ToolRegistry,
    get_tools,
    register,
    tool_names,
)

__all__ = [
    "DEFAULT_REGISTRY",
    "ParametersSpec",
    "RegisteredTool",
    "ToolExecutor",
    "ToolRegistry",
    "get_tools",
    "register",
    "tool_names",
]
