"""Structured-output extraction: text + Pydantic model -> validated dict."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, List, Optional

from pydantic import ValidationError

from ..core.errors import ExtractionError
from ..providers.base import BaseProvider, ChatMessage
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

_RETRY_TEMPLATE = """Your previous response was not valid for the schema.

Your previous response:
{previous}

Validation error:
{error}

Reply again with ONLY a corrected JSON object matching the schema. No prose, no code fences."""

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE) 


@dataclass
class ExtractionResult:
    """Everything the CLI (or another caller) needs from a successful extraction."""

    data: dict
    tokens_out: int
    attempts: int
    raw_text: str


def extract(
    provider: BaseProvider,
    text: str,
    schema: dict,
    *,
    temperature: float = 0.0,
    max_tokens: int = 512,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> ExtractionResult:
    """Run the full extraction pipeline against an already-validated schema.

    `schema` must already have passed `schema.load_and_validate_schema()` —
    this function does not re-check schema validity, only model output.

    Raises `ExtractionError` if the model never produces schema-valid JSON
    within `max_retries` retries (`GatewayError`s raised by the provider
    itself, e.g. rate limits, propagate unchanged so they keep their
    original classification).
    """
    model = build_model(schema, model_name="ExtractedData")
    schema_json = json.dumps(schema, indent=2)

    messages: List[ChatMessage] = [
        ChatMessage(role="system", content=_SYSTEM_PROMPT_TEMPLATE.format(schema_json=schema_json)),
        ChatMessage(role="user", content=text),
    ]

    last_error: Optional[str] = None
    last_raw = ""
    total_attempts = max_retries + 1

    for attempt in range(1, total_attempts + 1):
        response = provider.chat(messages, temperature=temperature, max_tokens=max_tokens)
        last_raw = response.text

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
                )

        if attempt < total_attempts:
            messages.append(ChatMessage(role="assistant", content=response.text))
            messages.append(
                ChatMessage(role="user", content=_RETRY_TEMPLATE.format(previous=response.text, error=last_error))
            )

    raise ExtractionError(
        f"Model output did not satisfy the schema after {total_attempts} attempt(s): {last_error}\n"
        f"Last raw response: {last_raw!r}",
        provider=provider.name,
    )


def _extract_json_value(text: str) -> Any:
    """Best-effort extraction of a JSON value from a raw model response.

    Tries, in order: the whole response as-is, a ```json ... ``` fenced
    block, then the largest {...} span in the text. Returns None if nothing
    parses.
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

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    return None
