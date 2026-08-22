"""Convert a validated JSON Schema (subset) into a Pydantic model."""

from __future__ import annotations

import keyword
import re
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

from pydantic import BaseModel, ConfigDict, Field, create_model

_SCALAR_TYPES: Dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}

_NON_IDENTIFIER_RE = re.compile(r"[^0-9a-zA-Z_]")


def build_model(schema: Dict[str, Any], model_name: str = "ExtractedData") -> Type[BaseModel]:
    """Build a Pydantic model for an already-validated, already-supported object schema."""
    return _object_to_model(schema, model_name)


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _object_to_model(schema: Dict[str, Any], model_name: str) -> Type[BaseModel]:
    properties: Dict[str, Any] = schema.get("properties", {})
    required = set(schema.get("required", []))
    additional = schema.get("additionalProperties", True)

    fields: Dict[str, Tuple[Any, Any]] = {}
    for prop_name, prop_schema in properties.items():
        py_type, field_kwargs = _type_and_kwargs(prop_schema, _child_name(model_name, prop_name))
        is_required = prop_name in required

        field_name, alias = _safe_identifier(prop_name) # use an alias to avoid conflicts with reserved keywords
        if alias is not None:
            field_kwargs["alias"] = alias

        if not is_required:
            py_type = Optional[py_type]
            field_kwargs.setdefault("default", None)

        fields[field_name] = (py_type, Field(**field_kwargs))

    config = ConfigDict(
        extra="forbid" if additional is False else "ignore",
        populate_by_name=True,
    )
    return create_model(_safe_class_name(model_name), __config__=config, **fields)  # type: ignore[call-overload]


def _type_and_kwargs(prop_schema: Dict[str, Any], child_name: str) -> Tuple[Any, Dict[str, Any]]:
    kwargs: Dict[str, Any] = {}
    if "description" in prop_schema:
        kwargs["description"] = prop_schema["description"]

    raw_type = prop_schema.get("type")
    schema_type = raw_type
    nullable = False
    if isinstance(raw_type, list):
        non_null = [t for t in raw_type if t != "null"]
        schema_type = non_null[0] if non_null else None
        nullable = "null" in raw_type

    if "enum" in prop_schema:
        py_type: Any = Literal[tuple(prop_schema["enum"])]  # type: ignore[valid-type]
    elif schema_type == "object":
        py_type = _object_to_model(prop_schema, child_name)
    elif schema_type == "array":
        item_type, _item_kwargs = _type_and_kwargs(prop_schema["items"], f"{child_name}Item")
        py_type = List[item_type]
        if "minItems" in prop_schema:
            kwargs["min_length"] = prop_schema["minItems"]
        if "maxItems" in prop_schema:
            kwargs["max_length"] = prop_schema["maxItems"]
    elif schema_type in _SCALAR_TYPES:
        py_type = _SCALAR_TYPES[schema_type]
        if schema_type in ("integer", "number"):
            if "minimum" in prop_schema:
                kwargs["ge"] = prop_schema["minimum"]
            if "maximum" in prop_schema:
                kwargs["le"] = prop_schema["maximum"]
            if "exclusiveMinimum" in prop_schema:
                kwargs["gt"] = prop_schema["exclusiveMinimum"]
            if "exclusiveMaximum" in prop_schema:
                kwargs["lt"] = prop_schema["exclusiveMaximum"]
        if schema_type == "string":
            if "minLength" in prop_schema:
                kwargs["min_length"] = prop_schema["minLength"]
            if "maxLength" in prop_schema:
                kwargs["max_length"] = prop_schema["maxLength"]
            if "pattern" in prop_schema:
                kwargs["pattern"] = prop_schema["pattern"]
    else:
        # schema.check_supported_subset() should already have rejected this;
        # this only guards against calling build_model() on an unvalidated schema.
        raise ValueError(f"Unsupported or missing 'type' in schema fragment: {prop_schema!r}")

    if nullable:
        py_type = Optional[py_type]

    return py_type, kwargs


def _child_name(parent: str, prop: str) -> str:
    return f"{_safe_class_name(parent)}_{_safe_class_name(prop)}"


def _safe_class_name(name: str) -> str:
    cleaned = _NON_IDENTIFIER_RE.sub("_", str(name))
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"M_{cleaned}"
    return cleaned[:1].upper() + cleaned[1:]


def _safe_identifier(name: str) -> Tuple[str, Optional[str]]:
    """Return (python_field_name, alias_or_None).

    `alias` is set (to the original property name) whenever sanitization
    changed the name, so `model_dump(by_alias=True)` round-trips back to the
    exact property names from the source JSON Schema — including names with
    spaces, hyphens, or that collide with Python keywords.
    """
    cleaned = _NON_IDENTIFIER_RE.sub("_", name)
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"f_{cleaned}"
    if keyword.iskeyword(cleaned):
        cleaned = f"{cleaned}_"
    return cleaned, (name if cleaned != name else None)
