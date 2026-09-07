"""Executes ToolCalls against a fixed set of registered tools.

Tool-domain failures — an unknown tool name in a call, arguments that fail
validation, an exception raised by the tool's own function — are treated
as conversational, recoverable events: they become an error `ToolResult`
fed back to the model, never an exception that aborts the run. A
`GatewayError` means the *runtime* failed to do its job; a tool error means
the *thing the tool touched* failed, which is squarely inside what the
model is reasoning about and should be able to see and react to.

This is deliberately not the case for asking `--tools` for a name that was
never registered at all — that is validated up front by
`registry.get_tools()` (see there) and raises before any provider call is
made, since there's no way for the model to recover from a CLI
misconfiguration mid-conversation.
"""

from __future__ import annotations

import json
from typing import Dict, List

from pydantic import ValidationError

from ..core.types import ToolCall, ToolResult
from .registry import RegisteredTool


class ToolExecutor:
    def __init__(self, tools: List[RegisteredTool]):
        self._tools: Dict[str, RegisteredTool] = {t.spec.name: t for t in tools}

    def execute(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(tool_call_id=call.id, content=f"No such tool: {call.name!r}.", is_error=True)

        try:
            args = tool.model.model_validate(call.arguments)
        except ValidationError as exc:
            return ToolResult(tool_call_id=call.id, content=f"Invalid arguments: {exc}", is_error=True)

        try:
            result = tool.func(**args.model_dump())
        except Exception as exc:  # tool-domain failure: recoverable, not a gateway error
            return ToolResult(tool_call_id=call.id, content=f"Tool error: {exc}", is_error=True)

        content = result if isinstance(result, str) else json.dumps(result, default=str)
        return ToolResult(tool_call_id=call.id, content=content, is_error=False)