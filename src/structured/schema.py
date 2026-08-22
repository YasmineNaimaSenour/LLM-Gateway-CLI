"""JSON Schema loading + validation for the structured-extraction pipeline.

Two different things can be wrong with a `--schema` file, and callers need
to be able to tell them apart

1. It isn't valid JSON Schema at all (bad JSON, bad meta-schema, garbage
   keyword types, ...) -> raises SchemaError
2. It IS valid JSON Schema, but it uses a feature outside the subset this
   gateway currently knows how to turn into a Pydantic model -> raises UnsupportedSchemaError

The root schema must be `"type": "object"` , structured extraction produces
one JSON object per call.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union

import jsonschema
from jsonschema.validators import validator_for

from ..core.errors import SchemaError, UnsupportedSchemaError

_SUPPORTED_SCALAR_TYPES = {"string", "integer", "number", "boolean"}
_SUPPORTED_TYPES = _SUPPORTED_SCALAR_TYPES | {"object", "array"}

# Keywords that signal a JSON Schema feature we deliberately don't support yet.
# Checked at every level of the schema (root, properties, array items, ...).
_UNSUPPORTED_KEYWORDS = (
    "$ref",
    "$defs",
    "definitions",
    "oneOf",
    "anyOf",
    "allOf",
    "not",
    "if",
    "then",
    "else",
    "const",
    "multipleOf",
    "patternProperties",
    "propertyNames",
    "dependentRequired",
    "dependentSchemas",
    "unevaluatedProperties",
    "unevaluatedItems",
    "contains",
    "prefixItems",
    "additionalItems",
)


def load_schema_file(path: Union[str, Path]) -> Dict[str, Any]:
    """Read and JSON-parse a schema file. Raises SchemaError on any problem."""
    p = Path(path)
    if not p.exists():
        raise SchemaError(f"Schema file not found: {path}")

    try:
        raw = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise SchemaError(f"Could not read schema file {path}: {exc}", cause=exc) from exc

    try:
        schema = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SchemaError(f"Schema file {path} is not valid JSON: {exc}", cause=exc) from exc

    if not isinstance(schema, dict):
        raise SchemaError(
            f"Schema file {path} must contain a JSON object at the top level, "
            f"got {type(schema).__name__}."
        )
    return schema


def validate_json_schema_document(schema: Dict[str, Any]) -> None:
    """Meta-validate: is this a structurally valid JSON Schema document at all?"""
    try:
        validator_cls = validator_for(schema)
        validator_cls.check_schema(schema)
    except jsonschema.exceptions.SchemaError as exc:
        raise SchemaError(f"Not a valid JSON Schema: {exc.message}", cause=exc) from exc
    except Exception as exc:  # pragma: no cover - defensive: malformed input jsonschema itself chokes on
        raise SchemaError(f"Could not parse JSON Schema: {exc}", cause=exc) from exc


def check_supported_subset(schema: Any, *, path: str = "$") -> None:
    """Walk `schema` and raise UnsupportedSchemaError for anything outside our subset.    """
    if not isinstance(schema, dict):
        # JSON Schema permits `true`/`false` as a whole schema; we don't support that.
        raise UnsupportedSchemaError(f"{path}: boolean JSON Schemas (`true`/`false`) are not supported.")

    _reject_unsupported_keywords(schema, path)
    _check_enum(schema, path)

    schema_type, nullable = _resolve_type(schema, path)

    if schema_type is None:
        if "enum" not in schema:
            raise UnsupportedSchemaError(
                f"{path}: every schema needs an explicit 'type' (or an 'enum') in this subset."
            )
        return  # a bare enum with no declared type is fine

    if schema_type not in _SUPPORTED_TYPES:
        raise UnsupportedSchemaError(f"{path}: unsupported 'type': {schema_type!r}.")

    if schema_type == "object":
        _check_object(schema, path)
    elif schema_type == "array":
        _check_array(schema, path)
    # scalars: nothing further beyond the keyword allow-list already enforced above


def load_and_validate_schema(path: Union[str, Path]) -> Dict[str, Any]:
    """Full pipeline step: load -> meta-validate -> enforce supported subset.

    Returns the raw schema dict, ready for `model_builder.build_model()`.
    """
    schema = load_schema_file(path)
    validate_json_schema_document(schema)

    if schema.get("type") != "object":
        raise UnsupportedSchemaError(
            "Root schema must have \"type\": \"object\" — structured extraction "
            "produces one JSON object per call."
        )

    check_supported_subset(schema)
    return schema


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _reject_unsupported_keywords(schema: Dict[str, Any], path: str) -> None:
    present = [k for k in _UNSUPPORTED_KEYWORDS if k in schema]
    if present:
        raise UnsupportedSchemaError(
            f"{path}: uses keyword(s) {present} which are outside the supported JSON Schema subset."
        )


def _check_enum(schema: Dict[str, Any], path: str) -> None:
    if "enum" not in schema:
        return
    values = schema["enum"]
    if not isinstance(values, list) or not values:
        raise UnsupportedSchemaError(f"{path}: 'enum' must be a non-empty list.")
    for v in values:
        if v is not None and not isinstance(v, (str, int, float, bool)):
            raise UnsupportedSchemaError(f"{path}: 'enum' values must be strings, numbers, booleans, or null.")


def _resolve_type(schema: Dict[str, Any], path: str):
    t = schema.get("type")
    if t is None:
        return None, False
    if isinstance(t, list):
        non_null = [x for x in t if x != "null"]
        if len(non_null) != 1 or len(t) != len(non_null) + (1 if "null" in t else 0):
            raise UnsupportedSchemaError(
                f'{path}: union types are only supported as [<type>, "null"], got {t!r}.'
            )
        return non_null[0], "null" in t
    if not isinstance(t, str):
        raise UnsupportedSchemaError(f'{path}: \'type\' must be a string or [<type>, "null"].')
    return t, False


def _check_object(schema: Dict[str, Any], path: str) -> None:
    properties = schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        raise UnsupportedSchemaError(
            f"{path}: object schemas must declare a non-empty 'properties' map in this subset."
        )

    required: List[Any] = schema.get("required", [])
    if not isinstance(required, list) or not all(isinstance(r, str) for r in required):
        raise UnsupportedSchemaError(f"{path}: 'required' must be a list of property-name strings.")
    unknown_required = [r for r in required if r not in properties]
    if unknown_required:
        raise UnsupportedSchemaError(f"{path}: 'required' references unknown properties: {unknown_required}.")

    additional = schema.get("additionalProperties", True)
    if not isinstance(additional, bool):
        raise UnsupportedSchemaError(
            f"{path}: 'additionalProperties' must be a boolean in this subset "
            "(a schema-valued additionalProperties is unsupported)."
        )

    for name, sub_schema in properties.items():
        check_supported_subset(sub_schema, path=f"{path}.properties.{name}")


def _check_array(schema: Dict[str, Any], path: str) -> None:
    items = schema.get("items")
    if items is None:
        raise UnsupportedSchemaError(f"{path}: array schemas must declare 'items' in this subset.")
    if isinstance(items, list):
        raise UnsupportedSchemaError(f"{path}: tuple-style 'items' (a list of schemas) is not supported.")

    for kw in ("minItems", "maxItems"):
        if kw in schema and not isinstance(schema[kw], int):
            raise UnsupportedSchemaError(f"{path}: '{kw}' must be an integer.")

    check_supported_subset(items, path=f"{path}.items")
