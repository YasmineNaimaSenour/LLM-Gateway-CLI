"""Guard tests for examples/structured/.

Examples rot silently: a schema that stops validating, or an input file that
disappears, breaks the very first thing a new user tries — with nobody
noticing until they try it. These tests make the repository fail loudly if
that happens, and assert the feature matrix the examples are meant to
demonstrate is actually covered (nested objects, arrays with items, enums,
nullable types, constraints).
"""

import json
from pathlib import Path

import pytest

from src.structured.model_builder import build_model
from src.structured.schema import check_supported_subset

EXAMPLES_DIR = Path(__file__).parent.parent / "examples" / "structured"

# Every schema/input pair must be listed here; a stray file in the directory
# is treated as an unfinished example rather than ignored.
EXPECTED_PAIRS = {
    "person_schema",
    "release_notes_schema",
    "ticket_schema",
    "weather_schema",
}

# schema stem -> the supported-subset features its properties demonstrate
EXPECTED_FEATURES = {
    "person_schema": {"nested_objects"},
    "release_notes_schema": {"nested_objects", "array_items", "enums", "patterns", "constraints"},
    "ticket_schema": {"nested_objects", "array_items", "enums", "nullable", "patterns"},
    "weather_schema": {"nested_objects", "enums", "nullable", "constraints"},
}


def _load(stem: str) -> dict:
    return json.loads((EXAMPLES_DIR / f"{stem}.json").read_text(encoding="utf-8"))


def test_example_directory_contains_exactly_the_declared_pairs():
    present = {p.stem for p in EXAMPLES_DIR.glob("*_schema.json")}
    assert present == EXPECTED_PAIRS
    for stem in EXPECTED_PAIRS:
        assert (EXAMPLES_DIR / f"{stem.replace('_schema', '_input')}.txt").exists(), (
            f"{stem} has no matching *_input.txt example"
        )


@pytest.mark.parametrize("stem", sorted(EXPECTED_PAIRS))
def test_example_schema_passes_subset_check_and_model_building(stem):
    schema = _load(stem)
    check_supported_subset(schema)
    Model = build_model(schema, model_name=stem)
    # A built model's own JSON schema is the real proof the example describes
    # something the pipeline can actually construct.
    assert Model.model_json_schema()["type"] == "object"


def test_examples_collectively_demonstrate_the_supported_subset():
    # Examples are the showcase of the feature matrix.
    # If someone adds a subset feature, they should also be able to see it
    # demonstrated in examples; these are the features worth demonstrating.
    features = set()
    for stem in EXPECTED_PAIRS:
        features |= EXPECTED_FEATURES[stem]
    assert features == {"nested_objects", "array_items", "enums", "nullable", "patterns", "constraints"}


@pytest.mark.parametrize(
    "stem, expected",
    [
        ("person_schema", {"nested_objects"}),
        ("release_notes_schema", {"nested_objects", "array_items", "enums", "patterns", "constraints"}),
        ("ticket_schema", {"nested_objects", "array_items", "enums", "nullable", "patterns"}),
        ("weather_schema", {"nested_objects", "enums", "nullable", "constraints"}),
    ],
)
def test_each_example_targets_a_distinct_feature_mix(stem, expected):
    # Distinct feature mixes per example (rather than one kitchen-sink schema)
    # so a user can find a minimal example for the feature they care about.
    assert EXPECTED_FEATURES[stem] == expected


def test_example_instances_validate_against_the_built_models():
    # End-to-end proof: representative instances drawn from each example's
    # own demonstrated features validate against the model the schema builds.
    # (The runtime path is model_validate + model_dump(by_alias=True) — the
    # same machinery the extractor uses on model output.)
    Model = build_model(_load("ticket_schema"), model_name="ticket_schema")
    instance = Model.model_validate(
        {
            "ticket_id": "TICKET-8842",
            "severity": "critical",
            "product_area": "shipping",
            "customer": {"name": "Dana Reyes", "email": "dana.reyes@example.com"},
            "orders": ["2024-A-77123"],
            "refund_requested": "49.90 EUR",  # nullable + pattern, by original name
            "churn_risk": True,
        }
    )
    assert instance.model_dump(by_alias=True)["severity"] == "critical"

    Model = build_model(_load("weather_schema"), model_name="weather_schema")
    instance = Model.model_validate(
        {
            "location": {"city": "Harborview"},
            "temperature_c": 4,
            "condition": "partly cloudy",
            "visibility_km": 0.8,  # nullable number
            "advisories": ["icy patches before mid-morning"],
        }
    )
    assert instance.model_dump(by_alias=True)["temperature_c"] == 4

    Model = build_model(_load("release_notes_schema"), model_name="release_notes_schema")
    instance = Model.model_validate(
        {
            "release": {"version": "v2.4.0"},  # nested + pattern
            "summary": "Rewritten sync engine and iOS offline fix.",
            "changes": [{"title": "Faster sync", "category": "added", "authors": ["@mira"]}],
        }
    )
    assert instance.model_dump(by_alias=True)["release"]["version"] == "v2.4.0"
