"""Structured-output coercion: messages + JSON Schema -> validated dict.

`coerce_to_schema()` is the shared primitive: given an in-progress
conversation and a schema, it validates the model's answer against the
schema and retries (feeding the validation error back to the model) until
it succeeds or `max_retries` is exhausted. Both the standalone `extract()`
(text -> JSON, used by the `structured` CLI command) and the orchestrator's
chat+`--schema` path build on this one implementation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterator, List, Optional

from pydantic import ValidationError

from ..core.errors import ExtractionError
from ..providers.base import BaseProvider, ChatMessage, ChatResponse
from .model_builder import build_model

DEFAULT_MAX_RETRIES = 2

_SYSTEM_PROMPT_TEMPLATE = """You are a precise data-extraction engine.

Extract structured data from the text the user provides and respond with
ONLY a single JSON object that matches the following JSON Schema exactly.
Do not include prose, explanations, or markdown code fences — output raw
JSON and nothing else. Omit no required field. Use `null` for optional
fields you cannot find in the text.

JSON Schema:
{schema_json}"""

_CHAT_SCHEMA_SYSTEM_TEMPLATE = """Once you are ready to give your final answer (after using any tools, if
needed), respond with ONLY a single JSON object that matches the following
JSON Schema exactly. Do not include prose, explanations, or markdown code
fences in your final answer — output raw JSON and nothing else.

JSON Schema:
{schema_json}"""

_RETRY_TEMPLATE = """Your previous response was not valid for the schema.

What you must produce (reminder):
{schema_summary}

Your previous response:
{previous}

Validation error:
{error}

Reply again with ONLY a corrected JSON object matching the schema. No prose, no code fences."""

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE) 


def _json_object_summary(schema: dict, path: str = "$") -> str:
    """A compact, model-readable one-line-per-field summary of an object schema.

    Used in corrective retry messages (audit #10): by the time a retry
    happens, the full schema from the original system prompt may have
    scrolled out of the model's effective attention window, so each retry
    re-states just what the final object must contain — required fields,
    types, enum values — instead of relying on "your last answer was wrong"
    alone.
    """
    lines: List[str] = []

    def _describe(sub: dict) -> str:
        if "enum" in sub:
            return "one of " + ", ".join(json.dumps(v) for v in sub["enum"])
        t = sub.get("type")
        if isinstance(t, list):
            return " or ".join(t)
        return t or "any"

    properties: dict = schema.get("properties", {})
    required = set(schema.get("required", []))
    for name, sub in properties.items():
        if not isinstance(sub, dict):
            continue
        marker = "required" if name in required else "optional"
        lines.append(f"- {path}.{name} ({marker}): {_describe(sub)}")
        if sub.get("type") == "object" and isinstance(sub.get("properties"), dict):
            lines.append(_json_object_summary(sub, path=f"{path}.{name}"))
        if sub.get("type") == "array" and isinstance(sub.get("items"), dict):
            lines.append(f"  ({path}.{name} items: {_describe(sub['items'])})")
    return "\n".join(lines)


@dataclass
class ExtractionResult:
    """Everything the CLI (or another caller) needs from a successful extraction."""

    data: dict
    tokens_out: int
    attempts: int
    raw_text: str
    # Provider-reported prompt tokens for the call that produced the accepted
    # answer (audit #11); None when the provider doesn't report usage, in
    # which case callers fall back to client-side counting.
    tokens_in: Optional[int] = None


def schema_instruction_message(schema: dict) -> ChatMessage:
    """Build the system message that instructs a *chat* conversation to give
    a schema-conforming final answer. Distinct wording from the extraction
    prompt above: this one is for a general chat turn (optionally after a
    tool-calling loop), not for extracting from a fixed block of text."""
    return ChatMessage(
        role="system",
        content=_CHAT_SCHEMA_SYSTEM_TEMPLATE.format(schema_json=json.dumps(schema, indent=2)),
    )


def coerce_to_schema(
    provider: BaseProvider,
    messages: List[ChatMessage],
    schema: dict,
    *,
    temperature: float = 0.0,
    max_tokens: int = 512,
    max_retries: int = DEFAULT_MAX_RETRIES,
    model_name: str = "ExtractedData",
    initial_response: Optional[ChatResponse] = None,
) -> ExtractionResult:
    """Validate-and-retry loop shared by `extract()` and the chat+schema path.

    `schema` must already have passed `schema.load_and_validate_schema()` —
    this function does not re-check schema validity, only model output.

    Unlike the orchestrator's `run_turn()` (which copies), this function
    mutates the `messages` list it is given: each failed attempt appends the
    invalid assistant response plus a corrective user message so the retry
    call has the failure in context. Callers pass their own working copy if
    they need the input list untouched.

    If `initial_response` is given, it is used as the first attempt instead
    of issuing a fresh `provider.chat()` call (used when the caller already
    has a tool-free response in hand, e.g. at the end of a tool-calling
    loop) — every retry beyond that still calls the provider fresh.

    Raises `ExtractionError` if the model never produces schema-valid JSON
    within `max_retries` retries (`GatewayError`s raised by the provider
    itself, e.g. rate limits, propagate unchanged so they keep their
    original classification).
    """
    model = build_model(schema, model_name=model_name)
    # Built once: every corrective retry message carries a concise reminder
    # of WHAT to produce (required fields, types, enum values), so the retry
    # prompt stands on its own even when early context (audit #10) has
    # pushed the full schema out of the model's effective attention window.
    schema_summary = _json_object_summary(schema)

    last_error: Optional[str] = None
    last_raw = ""
    last_tokens_in: Optional[int] = None
    total_attempts = max_retries + 1
    response = initial_response

    for attempt in range(1, total_attempts + 1):
        if response is None:
            # Snapshot per call: the provider sees the conversation exactly as
            # it was at call time, even though retry appends keep mutating
            # `messages` between attempts.
            response = provider.chat(
                list(messages), temperature=temperature, max_tokens=max_tokens, response_schema=schema
            )
        last_raw = response.text
        last_tokens_in = response.tokens_in  # provider-billed prompt count (audit #11), if reported

        parsed = _extract_json_value(response.text)
        if parsed is None:
            last_error = "Response did not contain a parsable JSON object."
        elif not isinstance(parsed, dict):
            last_error = f"Response was valid JSON but not a JSON object (got {type(parsed).__name__})."
        else:
            try:
                instance = model.model_validate(parsed)
            except ValidationError as exc:
                last_error = str(exc)
            else:
                return ExtractionResult(
                    data=instance.model_dump(by_alias=True),
                    tokens_out=response.tokens_out,
                    attempts=attempt,
                    raw_text=response.text,
                    tokens_in=last_tokens_in,
                )

        if attempt < total_attempts:
            messages.append(ChatMessage(role="assistant", content=response.text))
            messages.append(
                ChatMessage(
                    role="user",
                    content=_RETRY_TEMPLATE.format(
                        schema_summary=schema_summary, previous=response.text, error=last_error
                    ),
                )
            )
            response = None  # force a fresh call on the next attempt

    raise ExtractionError(
        f"Model output did not satisfy the schema after {total_attempts} attempt(s): {last_error}\n"
        f"Last raw response: {last_raw!r}",
        provider=provider.name,
    )


def extract(
    provider: BaseProvider,
    text: str,
    schema: dict,
    *,
    temperature: float = 0.0,
    max_tokens: int = 512,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> ExtractionResult:
    """Extract structured data from a fixed block of text (the `structured`
    CLI command). Thin wrapper: builds the extraction-specific system/user
    messages and delegates to `coerce_to_schema()`. The provider-billed
    `tokens_in` (audit #11) rides on the returned ExtractionResult — the
    provider's count already covers the messages built here.
    """
    schema_json = json.dumps(schema, indent=2)
    messages: List[ChatMessage] = [
        ChatMessage(role="system", content=_SYSTEM_PROMPT_TEMPLATE.format(schema_json=schema_json)),
        ChatMessage(role="user", content=text),
    ]
    return coerce_to_schema(
        provider, messages, schema, temperature=temperature, max_tokens=max_tokens, max_retries=max_retries
    )


def _extract_json_value(text: str) -> Any:
    """Best-effort extraction of a JSON value from a raw model response.

    Tries, in order: the whole response as-is, a ```json ... ``` fenced
    block, then the first balanced {...} span found by brace-depth counting
    (audit #9). The depth counter handles nested objects correctly and is
    immune to braces inside string literals or prose sitting between two
    separate JSON objects — failure modes of the old first-{-to-last-}
    slice. Returns None if nothing parses.
    """
    text = text.strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fence_match = _FENCE_RE.search(text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    for candidate in _find_balanced_json_objects(text):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue  # a balanced but non-JSON span (e.g. prose braces) — try the next

    return None


def _find_balanced_json_objects(text: str) -> Iterator[str]:
    """Yield each balanced `{...}` span in `text`, in order.

    Walks the text counting brace depth (audit #9): a span is yielded where
    the depth first returns to 0, then scanning continues after it — so a
    non-JSON balanced span in prose (e.g. `{curly}`) is skipped rather than
    aborting the search, and prose braces between two JSON objects cannot
    glue them into one invalid span. String literals are honored — braces
    inside `"..."` never affect the count, so `"a": "}"` cannot end a span
    early. A balanced-but-non-JSON span simply fails `json.loads` in the
    caller and the walk continues with the next candidate.
    """
    in_string = False
    escaped = False
    depth = 0
    start: Optional[int] = None
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    yield text[start : i + 1]
                    start = None