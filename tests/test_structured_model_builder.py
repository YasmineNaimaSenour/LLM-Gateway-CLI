import pytest
from pydantic import ValidationError

from src.structured.model_builder import build_model


def test_scalar_types_and_required_vs_optional():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "score": {"type": "number"},
            "active": {"type": "boolean"},
        },
        "required": ["name", "age"],
    }
    Model = build_model(schema)

    instance = Model.model_validate({"name": "Ada", "age": 30})
    dumped = instance.model_dump()
    assert dumped["name"] == "Ada"
    assert dumped["age"] == 30
    assert dumped["score"] is None
    assert dumped["active"] is None

    with pytest.raises(ValidationError):
        Model.model_validate({"age": 30})  # missing required 'name'

    with pytest.raises(ValidationError):
        Model.model_validate({"name": "Ada", "age": "not a number"})


def test_enum_accepts_only_listed_values():
    schema = {
        "type": "object",
        "properties": {"role": {"type": "string", "enum": ["admin", "user", "guest"]}},
        "required": ["role"],
    }
    Model = build_model(schema)
    assert Model.model_validate({"role": "admin"}).model_dump()["role"] == "admin"
    with pytest.raises(ValidationError):
        Model.model_validate({"role": "superadmin"})


def test_numeric_and_string_constraints_enforced():
    schema = {
        "type": "object",
        "properties": {
            "age": {"type": "integer", "minimum": 0, "maximum": 120},
            "code": {"type": "string", "minLength": 2, "maxLength": 4, "pattern": "^[A-Z]+$"},
        },
        "required": ["age", "code"],
    }
    Model = build_model(schema)
    Model.model_validate({"age": 30, "code": "AB"})  # should not raise

    with pytest.raises(ValidationError):
        Model.model_validate({"age": -1, "code": "AB"})
    with pytest.raises(ValidationError):
        Model.model_validate({"age": 200, "code": "AB"})
    with pytest.raises(ValidationError):
        Model.model_validate({"age": 30, "code": "a"})  # violates pattern (lowercase)
    with pytest.raises(ValidationError):
        Model.model_validate({"age": 30, "code": "TOOLONG"})  # exceeds maxLength


def test_nested_object_is_validated_recursively():
    schema = {
        "type": "object",
        "properties": {
            "address": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "zip": {"type": "string"}},
                "required": ["city"],
            }
        },
        "required": ["address"],
    }
    Model = build_model(schema)
    instance = Model.model_validate({"address": {"city": "London"}})
    assert instance.model_dump()["address"]["city"] == "London"

    with pytest.raises(ValidationError):
        Model.model_validate({"address": {}})  # missing required nested 'city'


def test_array_of_objects_with_item_constraints():
    schema = {
        "type": "object",
        "properties": {
            "events": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"year": {"type": "integer"}, "label": {"type": "string"}},
                    "required": ["year", "label"],
                },
                "minItems": 1,
            }
        },
        "required": ["events"],
    }
    Model = build_model(schema)
    instance = Model.model_validate({"events": [{"year": 1990, "label": "born"}]})
    assert instance.model_dump()["events"][0]["year"] == 1990

    with pytest.raises(ValidationError):
        Model.model_validate({"events": []})  # violates minItems
    with pytest.raises(ValidationError):
        Model.model_validate({"events": [{"year": 1990}]})  # missing required 'label'


def test_nullable_type_allows_none():
    schema = {
        "type": "object",
        "properties": {"nickname": {"type": ["string", "null"]}},
        "required": ["nickname"],
    }
    Model = build_model(schema)
    assert Model.model_validate({"nickname": None}).model_dump()["nickname"] is None
    assert Model.model_validate({"nickname": "Ada"}).model_dump()["nickname"] == "Ada"


def test_additional_properties_false_forbids_extra_fields():
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": False,
    }
    Model = build_model(schema)
    Model.model_validate({"name": "Ada"})
    with pytest.raises(ValidationError):
        Model.model_validate({"name": "Ada", "extra": "nope"})


def test_additional_properties_default_allows_extra_fields():
    schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
    Model = build_model(schema)
    instance = Model.model_validate({"name": "Ada", "extra": "fine"})
    assert instance.model_dump()["name"] == "Ada"


def test_property_names_needing_sanitization_round_trip_via_alias():
    schema = {
        "type": "object",
        "properties": {"zip-code": {"type": "string"}, "class": {"type": "string"}},
        "required": ["zip-code", "class"],
    }
    Model = build_model(schema)
    instance = Model.model_validate({"zip-code": "12345", "class": "A"})
    dumped = instance.model_dump(by_alias=True)
    assert dumped == {"zip-code": "12345", "class": "A"}
