"""The gateway-core runtime: the single entry point both the `chat` and
`structured` CLI commands call into. This is what "the CLI is only an
interface" means in code — `run_turn()` owns the tool-call loop and the
structured-output coercion; `cli.py` only does argument parsing and I/O.

Tool-bearing turns are always non-streaming (see providers/base.py's
`chat_stream()` docstring for why). Schema enforcement, when requested,
applies only to the final, tool-free turn — the tool loop and the schema
coercion are different control-flow shapes (a bounded loop vs. a bounded
retry) that compose here rather than sharing one abstraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .errors import ToolLoopError
from .types import ToolSpec
from ..providers.base import BaseProvider, ChatMessage
from ..structured.extractor import DEFAULT_MAX_RETRIES, coerce_to_schema
from ..tools.executor import ToolExecutor

DEFAULT_MAX_TOOL_ITERATIONS = 8


@dataclass
class OrchestrationResult:
    text: str
    data: Optional[dict]
    tokens_out: int
    tool_call_count: int
    tool_iterations: int
    attempts: int
    # Complete transcript for the turn (caller's input + every assistant/tool
    # message added during it, including the final answer). Feed it back as
    # `messages` for the next turn to continue the conversation.
    messages: List[ChatMessage]


def run_turn(
    provider: BaseProvider,
    messages: List[ChatMessage],
    *,
    tools: Optional[List[ToolSpec]] = None,
    tool_executor: Optional[ToolExecutor] = None,
    response_schema: Optional[dict] = None,
    max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> OrchestrationResult:
    """Run one full turn: an optional tool-calling loop, then an optional
    schema-coercion step on the final, tool-free answer.

    `messages` is never mutated: the turn runs on an internal copy, and the
    complete transcript (the caller's messages plus every assistant/tool
    message appended during the turn, including the final answer) is
    returned as `OrchestrationResult.messages`. Reuse that list — not the
    one you passed in — to continue the conversation in a follow-up turn.
    """
    if tools and tool_executor is None:
        raise ValueError("tool_executor is required when tools are provided")

    # Copy-on-entry: work on our own list so a caller reusing their list
    # across turns can't get doubled-up history. A shallow copy suffices —
    # ChatMessage instances are treated as immutable value objects.
    messages = list(messages)

    tokens_out = 0
    tool_call_count = 0
    tool_iterations = 0
    response = None

    if tools:
        for _ in range(max_tool_iterations):
            # Snapshot per call: the provider sees the conversation exactly as
            # it was at call time, even though `messages` keeps growing after.
            response = provider.chat(list(messages), temperature=temperature, max_tokens=max_tokens, tools=tools)
            if not response.tool_calls:
                break  # tool-free final turn — fall through to schema handling below

            tokens_out += response.tokens_out
            tool_iterations += 1
            messages.append(
                ChatMessage(role="assistant", content=response.text or None, tool_calls=response.tool_calls)
            )
            for call in response.tool_calls:
                tool_call_count += 1
                result = tool_executor.execute(call)
                messages.append(
                    ChatMessage(role="tool", content=result.content, tool_call_id=result.tool_call_id, name=call.name)
                )
        else:
            raise ToolLoopError(
                f"Model kept calling tools past the limit of {max_tool_iterations} iteration(s) "
                "without producing a final answer.",
                provider=provider.name,
            )

    if response_schema is not None:
        extraction = coerce_to_schema(
            provider,
            messages,
            response_schema,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            initial_response=response,
        )
        # Record the final answer so the returned transcript is complete;
        # otherwise a caller continuing the conversation would silently
        # lose the model's last word.
        messages.append(ChatMessage(role="assistant", content=extraction.raw_text or None))
        return OrchestrationResult(
            text=extraction.raw_text,
            data=extraction.data,
            tokens_out=tokens_out + extraction.tokens_out,
            tool_call_count=tool_call_count,
            tool_iterations=tool_iterations,
            attempts=extraction.attempts,
            messages=messages,
        )

    if response is None:
        response = provider.chat(list(messages), temperature=temperature, max_tokens=max_tokens)
    tokens_out += response.tokens_out
    messages.append(ChatMessage(role="assistant", content=response.text or None))

    return OrchestrationResult(
        text=response.text,
        data=None,
        tokens_out=tokens_out,
        tool_call_count=tool_call_count,
        tool_iterations=tool_iterations,
        attempts=1,
        messages=messages,
    )