"""Provider-agnostic runtime types for tool calling.

These are shared vocabulary between providers (which translate to/from
their own wire format), the tool registry/executor (which define and run
tools), and the orchestrator (which loops over both). They deliberately do
NOT live in providers/base.py: providers consume these types, they don't
own them, and the tool registry/executor need them without depending on
the provider package at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ToolSpec:
    """Declares a callable tool: name, description, and a JSON Schema
    (object type) describing its parameters. This is the wire-agnostic
    contract every provider adapter translates into its own tool-definition
    shape."""

    name: str
    description: str
    parameters: Dict[str, Any]


@dataclass
class ToolCall:
    """A single tool invocation requested by the model. `arguments` is
    always an already-parsed dict by the time gateway-core sees it — each
    provider adapter is responsible for getting it into that shape (e.g.
    json.loads-ing a JSON-encoded argument string)."""

    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ToolResult:
    """The result of executing a ToolCall, to be sent back to the model as
    a tool-role message. `is_error` marks tool-domain failures (unknown
    tool, bad arguments, an exception inside the tool) that are meant to be
    seen and reacted to by the model, not raised as gateway errors."""

    tool_call_id: str
    content: str
    is_error: bool = False