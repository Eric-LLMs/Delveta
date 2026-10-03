"""Argument-acquisition BRIDGE (Phase 4).

Pins the transitional adapter that lets a DECIDED capability with no explicit
``acquisition`` entry still enter the Argument Path Router, and the three
deterministic producers the production provider composes:

* :mod:`..argument_acquisition.declaration` — slot-name ownership table + the
  legacy ``arg_slots`` reconciliation (an explicit source wins over the name
  heuristic); no ``parameters`` -> ``{}`` (undeclared boundary unchanged);
* :mod:`..argument_acquisition.evidence` — query evidence for MODEL slots only,
  nothing for a system owner and nothing for an empty query;
* :mod:`..argument_acquisition.context_values` — SYSTEM_BINDER values from the
  settled turn facts (reusing the Binder's one truth table), nothing fabricated;
* :mod:`..argument_acquisition.provider` — ``for_turn`` composes the three and
  leaves the MODEL side empty.
"""
from __future__ import annotations

import types

from core.application.chat.intent_funnel.argument_acquisition import context_values
from core.application.chat.intent_funnel.argument_acquisition import declaration as decl
from core.application.chat.intent_funnel.argument_acquisition import evidence as ev
from core.application.chat.intent_funnel.argument_acquisition import provider as prov
from core.application.chat.intent_funnel.argument_acquisition.contract import (
    OWNERSHIP_MODEL,
    OWNERSHIP_SYSTEM_BINDER,
    OWNERSHIP_TOOL_DEFAULT,
    OWNERSHIP_UNAVAILABLE,
    SOURCE_QUERY,
    SOURCE_UI_CONTEXT,
)
from core.application.chat.intent_funnel.registry.entry import CapabilityEntry


def _entry(cid="cap-a", *, parameters=None, arg_slots=None) -> CapabilityEntry:
    return CapabilityEntry(
        capability_id=cid, tool_binding="create_folder", description="does a",
        parameters=dict(parameters or {}), arg_slots=dict(arg_slots or {}))


# ── declaration: slot name -> ownership ─────────────────────────────────────────


def test_ownership_comes_from_the_slot_name_table():
    d = decl.declaration_for(_entry(parameters={
        "asset_id": {"type": "string", "required": True},
        "parent_path": {"type": "string"},
        "definition": {"type": "string"},
        "output_dir": {"type": "string"},
        "timeout": {"type": "integer"},
        "max_chars": {"type": "integer"},
        "project_id": {"type": "string"},
        "run_id": {"type": "string"},
        "paths": {"type": "array"},
        "name": {"type": "string"},          # unlisted -> MODEL default
    }))
    by_slot = {s: d[s].ownership for s in d}
    assert by_slot["asset_id"] == OWNERSHIP_SYSTEM_BINDER
    for s in ("parent_path", "definition", "output_dir", "timeout", "max_chars"):
        assert by_slot[s] == OWNERSHIP_TOOL_DEFAULT, s
    for s in ("project_id", "run_id", "paths"):
        assert by_slot[s] == OWNERSHIP_UNAVAILABLE, s
    assert by_slot["name"] == OWNERSHIP_MODEL


def test_legacy_arg_slots_source_overrides_the_name_heuristic():
    d = decl.declaration_for(_entry(
        parameters={"name": {"type": "string"}},
        arg_slots={"name": {"source": "viewer.selection"}}))
    assert d["name"].ownership == OWNERSHIP_SYSTEM_BINDER


def test_legacy_arg_slots_accepts_the_bare_string_shape():
    d = decl.declaration_for(_entry(
        parameters={"name": {"type": "string"}},
        arg_slots={"name": "turn_context"}))
    assert d["name"].ownership == OWNERSHIP_SYSTEM_BINDER


def test_legacy_fixed_is_tool_default_and_plugin_is_model():
    d = decl.declaration_for(_entry(
        parameters={"a": {"type": "string"}, "b": {"type": "string"}},
        arg_slots={"a": {"source": "fixed"}, "b": {"source": "plugin:extract"}}))
    assert d["a"].ownership == OWNERSHIP_TOOL_DEFAULT
    assert d["b"].ownership == OWNERSHIP_MODEL


def test_legacy_key_absent_from_the_schema_declares_nothing():
    # a legacy arg_slots key the Binder would never validate is ignored: slots
    # are taken from the CANONICAL schema (parameters) only.
    d = decl.declaration_for(_entry(
        parameters={"name": {"type": "string"}},
        arg_slots={"ghost": {"source": "user_input"}}))
    assert set(d) == {"name"}


def test_no_parameters_yields_the_empty_declaration():
    assert decl.declaration_for(_entry(parameters={})) == {}


def test_declaration_carries_no_schema_facts():
    d = decl.declaration_for(_entry(parameters={
        "name": {"type": "string", "required": True, "description": "x"}}))
    for forbidden in ("required", "type", "enum", "description", "max_len"):
        assert not hasattr(d["name"], forbidden)


# ── evidence: deterministic, MODEL-only, query-driven ───────────────────────────


def _decl_two() -> dict:
    return decl.declaration_for(_entry(parameters={
        "name": {"type": "string"},
        "asset_id": {"type": "string"},
    }))


def test_query_evidence_is_emitted_for_model_slots_only():
    e = ev.evidence_for(_entry(), _decl_two(), query="create a folder")
    assert e == {"name": SOURCE_QUERY}          # asset_id is SYSTEM_BINDER


def test_empty_query_carries_no_evidence():
    assert ev.evidence_for(_entry(), _decl_two(), query="   ") == {}


def test_no_declaration_yields_no_evidence():
    assert ev.evidence_for(_entry(), {}, query="x") == {}


# ── context values: SYSTEM_BINDER from the settled turn facts ───────────────────


def test_asset_id_is_resolved_from_the_viewer_fact():
    d = _decl_two()
    facts = types.SimpleNamespace(viewer_asset_id="uuid-1", path_asset_id=None,
                                  attachment_asset_id=None)
    values, sources = context_values.system_values_for(_entry(), d, facts=facts)
    assert values == {"asset_id": "uuid-1"}
    assert sources == {"asset_id": SOURCE_UI_CONTEXT}


def test_asset_id_fact_precedence_prefers_attachment():
    d = _decl_two()
    facts = types.SimpleNamespace(viewer_asset_id="v", path_asset_id="p",
                                  attachment_asset_id="a")
    values, _ = context_values.system_values_for(_entry(), d, facts=facts)
    assert values["asset_id"] == "a"            # attachment wins (precedence order)


def test_model_slots_are_never_resolved_from_facts():
    d = _decl_two()
    facts = types.SimpleNamespace(viewer_asset_id=None, path_asset_id=None,
                                  attachment_asset_id=None, name="nope")
    values, sources = context_values.system_values_for(_entry(), d, facts=facts)
    assert values == {} and sources == {}


def test_no_facts_resolves_nothing():
    values, sources = context_values.system_values_for(
        _entry(), _decl_two(), facts=None)
    assert values == {} and sources == {}


# ── provider.for_turn: composes the three, leaves the MODEL side empty ──────────


def test_for_turn_builds_inputs_for_a_declared_capability():
    entry = _entry("cap-a", parameters={
        "asset_id": {"type": "string", "required": True},
        "pages": {"type": "string", "required": True},
    })
    facts = types.SimpleNamespace(viewer_asset_id="uuid-1", path_asset_id=None,
                                  attachment_asset_id=None)
    p = prov.for_turn(entries_by_id={"cap-a": entry}, query="open page 3", facts=facts)
    inputs = p("cap-a")
    assert set(inputs.declaration) == {"asset_id", "pages"}
    assert inputs.declaration["asset_id"].ownership == OWNERSHIP_SYSTEM_BINDER
    assert inputs.declaration["pages"].ownership == OWNERSHIP_MODEL
    assert inputs.evidence == {"pages": SOURCE_QUERY}       # MODEL slot only
    assert inputs.system_values == {"asset_id": "uuid-1"}
    assert inputs.system_sources == {"asset_id": SOURCE_UI_CONTEXT}
    assert inputs.model_values == {} and inputs.model_source == ""   # left for the extractor


def test_for_turn_returns_none_for_an_absent_capability():
    p = prov.for_turn(entries_by_id={}, query="q", facts=None)
    assert p("cap-gone") is None


def test_for_turn_returns_none_for_a_parameterless_capability():
    entry = _entry("cap-a", parameters={})
    p = prov.for_turn(entries_by_id={"cap-a": entry}, query="q", facts=None)
    assert p("cap-a") is None                   # undeclared boundary (§G)
