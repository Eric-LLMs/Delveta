"""Registry parameter-schema normalizer tests.

The live ``capabilities.parameters`` column carries TWO shapes: the canonical
flat ``{slot: spec}`` and a raw JSON-Schema ``{type, required, properties}``.
The whole funnel (Binder, ARP, declaration bridge, extractor) is written
against the flat form, so ``CapabilityEntry`` normalizes at construction — the
ONE Registry intake boundary. These tests pin the transform itself plus the two
downstream regressions it fixes (ARP crash, Binder BIND_INVALID).
"""
from __future__ import annotations

import pytest
from core.application.chat.intent_funnel.argument_acquisition.declaration import (
    declaration_for,
)
from core.application.chat.intent_funnel.argument_acquisition.path_router import route
from core.application.chat.intent_funnel.binder.validator import validate_against_schema
from core.application.chat.intent_funnel.registry import entry as T
from core.application.chat.intent_funnel.registry.parameter_schema import (
    normalize_parameters,
)

_JSON_SCHEMA = {
    "type": "object",
    "required": ["path"],
    "properties": {
        "path": {"type": "string", "description": "the file"},
        "max_chars": {"type": "integer", "description": "read cap"},
    },
}


def _schema_entry(**kw) -> T.CapabilityEntry:
    base = dict(
        capability_id="cap-read-file",
        tool_binding="read_file",
        description="Read a file.",
        parameters=dict(_JSON_SCHEMA),
    )
    base.update(kw)
    return T.CapabilityEntry(**base)


# ── the transform ────────────────────────────────────────────────────────────────

def test_json_schema_becomes_flat():
    flat = normalize_parameters(_JSON_SCHEMA)
    assert set(flat) == {"path", "max_chars"}
    assert flat["path"] == {"type": "string", "description": "the file", "required": True}
    assert flat["max_chars"]["required"] is False
    assert flat["max_chars"]["type"] == "integer"


def test_flat_input_is_unchanged_and_idempotent():
    flat = {"name": {"type": "string", "description": "n", "required": True,
                     "max_len": 120}}
    assert normalize_parameters(flat) == flat
    assert normalize_parameters(normalize_parameters(_JSON_SCHEMA)) == \
        normalize_parameters(_JSON_SCHEMA)


def test_unknown_keys_preserved_and_maxlength_folded():
    out = normalize_parameters({
        "type": "object", "required": [],
        "properties": {"x": {"type": "string", "enum": ["a", "b"],
                             "maxLength": 8, "description": "d"}},
    })
    assert out["x"]["enum"] == ["a", "b"]
    assert out["x"]["max_len"] == 8
    assert "maxLength" not in out["x"]


def test_empty_and_malformed_yield_empty():
    assert normalize_parameters({}) == {}
    assert normalize_parameters(None) == {}
    assert normalize_parameters("nonsense") == {}


def test_per_tool_slot_sets_are_not_merged():
    """The adapter converts the WRAPPER only — each tool keeps its own slots."""
    for schema, expect in (
        ({"type": "object", "required": ["command"],
          "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}}},
         {"command": "command", "timeout": "timeout"}),
        ({"type": "object", "required": ["action"],
          "properties": {"action": {"type": "string"}, "run_id": {"type": "string"}}},
         {"action": "action", "run_id": "run_id"}),
    ):
        assert set(normalize_parameters(schema)) == set(expect.values())


# ── the Registry intake boundary ─────────────────────────────────────────────────

def test_capability_entry_normalizes_at_construction():
    e = _schema_entry()
    assert all(isinstance(v, dict) for v in e.parameters.values())
    assert set(e.parameters) == {"path", "max_chars"}


def test_arg_slots_untouched():
    e = _schema_entry(arg_slots={"path": "user_input"})
    assert e.arg_slots == {"path": "user_input"}


def test_replace_round_trip_is_idempotent():
    import dataclasses
    e = _schema_entry()
    again = dataclasses.replace(e, description="Read a file (v2).")
    assert again.parameters == e.parameters


# ── downstream regressions the boundary fixes ────────────────────────────────────

def test_arp_no_longer_crashes_on_schema_shaped_entry():
    """Regression guard: ``route`` used to raise AttributeError on a
    JSON-Schema-shaped parameters map (the 59-case CASCADE_ERROR)."""
    e = _schema_entry()
    decision = route(e.parameters, declaration_for(e),
                     evidence={"path": "QUERY"})
    # path is required+MODEL with query evidence -> a MODEL acquisition strategy
    assert decision.declared is True
    assert decision.strategy is not None


def test_binder_accepts_legal_draft_for_schema_shaped_entry():
    """Regression guard: ``validate`` used to return BIND_INVALID for ANY draft
    because the schema's top-level keys leaked into the unknown-slot check."""
    e = _schema_entry()
    bound = validate_against_schema(e.parameters, {"path": "notes.txt"})
    assert bound.state == "COMPLETE"
    assert bound.args == {"path": "notes.txt"}


def test_binder_still_enforces_required_and_unknown_slots():
    e = _schema_entry()
    assert validate_against_schema(e.parameters, {"max_chars": "10"}).state == "MISSING"
    assert validate_against_schema(
        e.parameters, {"path": "a", "bogus": "x"}).state == "INVALID"
