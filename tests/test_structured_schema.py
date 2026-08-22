import json

import pytest

from src.core.errors import SchemaError, UnsupportedSchemaError
from src.structured.schema import (
    check_supported_subset,
    load_and_validate_schema,
    load_schema_file,
    validate_json_schema_document,
)


def _write_schema(tmp_path, schema, name="schema.json"):
    path = tmp_path / name
    path.write_text(json.dumps(schema), encoding="utf-8")
    return path


# -- load_schema_file -----------------------------------------------------


def test_load_schema_file_missing_raises_schema_error(tmp_path):
    with pytest.raises(SchemaError):
        load_schema_file(tmp_path / "does_not_exist.json")


def test_load_schema_file_bad_json_raises_schema_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(SchemaError):
        load_schema_file(path)


def test_load_schema_file_non_object_top_level_raises_schema_error(tmp_path):
    path = tmp_path / "array_root.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(SchemaError):
        load_schema_file(path)


# -- validate_json_schema_document (invalid JSON Schema, distinct from "unsupported") --


def test_invalid_meta_schema_raises_schema_error():
    # `required` must be an array of strings per the JSON Schema meta-schema, not a string.
    schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": "a"}
    with pytest.raises(SchemaError):
        validate_json_schema_document(schema)


def test_property_type_not_a_string_raises_schema_error():
    schema = {"type": "object", "properties": {"a": {"type": 123}}}
    with pytest.raises(SchemaError):
        validate_json_schema_document(schema)


def test_structurally_valid_schema_passes_meta_validation():
    schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
    validate_json_schema_document(schema)  # should not raise


# -- check_supported_subset (valid JSON Schema, but outside our subset) --


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object", "properties": {"a": {"$ref": "#/$defs/Foo"}}},
        {"type": "object", "properties": {"a": {"oneOf": [{"type": "string"}, {"type": "integer"}]}}},
        {"type": "object", "properties": {"a": {"anyOf": [{"type": "string"}, {"type": "integer"}]}}},
        {"type": "object", "properties": {"a": {"allOf": [{"type": "string"}]}}},
        {"type": "object", "properties": {"a": {"type": "number", "multipleOf": 2}}},
        {"type": "object", "properties": {"a": {"const": "fixed"}}},
        {"type": "object", "patternProperties": {"^S_": {"type": "string"}}, "properties": {"a": {"type": "string"}}},
        # tuple-style `items` (a list of schemas) is only meta-schema-valid under
        # older drafts (e.g. draft-07) — declare it explicitly so this exercises
        # our subset check rather than tripping the meta-schema check first.
        {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {"items": {"type": "array", "items": [{"type": "string"}, {"type": "integer"}]}},
        },
        {"type": "object", "properties": {"a": {"type": "array"}}},  # array missing 'items'
        {"type": "object", "additionalProperties": {"type": "string"}, "properties": {"a": {"type": "string"}}},
        {"type": "object"},  # no properties at all
    ],
)
def test_unsupported_features_raise_unsupported_schema_error(schema):
    validate_json_schema_document(schema)  # sanity: these are valid JSON Schema...
    with pytest.raises(UnsupportedSchemaError):
        check_supported_subset(schema)  # ...just outside our subset


def test_non_object_root_type_raises_unsupported(tmp_path):
    path = _write_schema(tmp_path, {"type": "array", "items": {"type": "string"}})
    with pytest.raises(UnsupportedSchemaError):
        load_and_validate_schema(path)


def test_required_referencing_unknown_property_is_unsupported():
    schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["b"]}
    with pytest.raises(UnsupportedSchemaError):
        check_supported_subset(schema)


def test_union_type_other_than_nullable_is_unsupported():
    schema = {"type": "object", "properties": {"a": {"type": ["string", "integer"]}}}
    with pytest.raises(UnsupportedSchemaError):
        check_supported_subset(schema)


def test_empty_enum_is_unsupported():
    schema = {"type": "object", "properties": {"a": {"enum": []}}}
    with pytest.raises(UnsupportedSchemaError):
        check_supported_subset(schema)


# -- the happy path: full supported subset --------------------------------


def test_full_supported_subset_passes(tmp_path):
    schema = {
        "type": "object",
        "description": "A person record",
        "properties": {
            "name": {"type": "string", "description": "Full name", "minLength": 1, "maxLength": 100},
            "age": {"type": "integer", "minimum": 0, "maximum": 150},
            "score": {"type": "number", "exclusiveMinimum": 0},
            "active": {"type": "boolean"},
            "role": {"type": "string", "enum": ["admin", "user", "guest"]},
            "nickname": {"type": ["string", "null"]},
            "tags": {"type": "array", "items": {"type": "string"}, "minItems": 0, "maxItems": 10},
            "address": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "zip": {"type": "string", "pattern": "^[0-9]{5}$"},
                },
                "required": ["city"],
            },
            "history": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"year": {"type": "integer"}, "event": {"type": "string"}},
                    "required": ["year", "event"],
                },
            },
        },
        "required": ["name", "age"],
    }
    path = _write_schema(tmp_path, schema)
    result = load_and_validate_schema(path)
    assert result == schema
