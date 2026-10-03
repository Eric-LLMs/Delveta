"""Phase 4 Step 1 (2026-10-01) — Argument Path Router unit tests.

Deterministic business logic only: the Readiness double gate (§D), the two
dimensional strategy derivation (§B), the context bundle window (§F), and the
merge/provenance contract (§B/§F). No Qwen, no wiring, no Binder, no Registry
read — the module under test is a pure function of its injected inputs.
"""
from __future__ import annotations

import inspect

import pytest

from core.application.chat.intent_funnel.argument_acquisition import (
    contract as ac,
)
from core.application.chat.intent_funnel.argument_acquisition import (
    context_bundle as cb,
)
from core.application.chat.intent_funnel.argument_acquisition import (
    merge as mg,
)
from core.application.chat.intent_funnel.argument_acquisition import (
    path_router as pr,
)

# ── fixtures: small hand-written schemas/declarations ──────────────────────────
# parameters == Registry schema truth (required lives HERE, per §A.1); the
# declaration carries ownership / allowed_sources ONLY.

_ASSET = ac.SlotDecl(ac.OWNERSHIP_SYSTEM_BINDER,
                     allowed_sources=(ac.SOURCE_UI_CONTEXT, ac.SOURCE_RESOLVER))
_PAGES_Q = ac.SlotDecl(ac.OWNERSHIP_MODEL, allowed_sources=(ac.SOURCE_QUERY,))
_PAGES_H = ac.SlotDecl(ac.OWNERSHIP_MODEL,
                       allowed_sources=(ac.SOURCE_CONVERSATION_5_USER_TURNS,))


def _params(**required: bool) -> dict:
    return {name: {"required": req} for name, req in required.items()}


# ── 1. CONTEXT_DIRECT ─────────────────────────────────────────────────────────


def test_context_direct_when_required_ready_and_no_model_need():
    out = pr.route(
        _params(asset_id=True, pages=False),
        {"asset_id": _ASSET, "pages": _PAGES_Q},
        system_values={"asset_id": "A1"}, evidence={},
    )
    assert out.declared is True
    assert out.strategy == ac.STRATEGY_CONTEXT_DIRECT
    assert out.model_slots == ()            # Qwen call = 0
    assert out.readiness.ready is True
    assert out.readiness.needs_acquisition is False


# ── 2. QUERY_TO_EXTRACTOR ─────────────────────────────────────────────────────


def test_query_to_extractor_when_query_alone_evidences_a_model_slot():
    out = pr.route(
        _params(pages=True),
        {"pages": _PAGES_Q},
        evidence={"pages": ac.SOURCE_QUERY},
    )
    assert out.strategy == ac.STRATEGY_QUERY_TO_EXTRACTOR
    assert out.model_slots == ("pages",)
    assert out.bundle_source == ac.SOURCE_QUERY


# ── 3. QUERY_PLUS_5TURNS_TO_EXTRACTOR ─────────────────────────────────────────


def test_query_plus_5_when_acquisition_depends_on_recent_user_turns():
    out = pr.route(_params(pages=True), {"pages": _PAGES_H},
                   evidence={"pages": ac.SOURCE_CONVERSATION_5_USER_TURNS})
    assert out.strategy == ac.STRATEGY_QUERY_PLUS_5TURNS_TO_EXTRACTOR
    assert out.bundle_source == ac.SOURCE_CONVERSATION_5_USER_TURNS


# ── 3b. required MODEL: source is EVIDENCE-driven, never assumed from allowed ──


def test_required_model_slot_without_any_evidence_is_missing():
    # allowed_sources contains QUERY, but that does NOT create query evidence.
    out = pr.route(_params(pages=True), {"pages": _PAGES_Q})
    assert out.strategy == ac.STRATEGY_MISSING
    assert out.model_slots == ()            # nothing is sent to Qwen


# ── 5. MIXED ──────────────────────────────────────────────────────────────────


def test_mixed_when_system_slots_and_model_acquisition_coexist():
    out = pr.route(
        _params(asset_id=True, pages=False),
        {"asset_id": _ASSET, "pages": _PAGES_Q},
        system_values={"asset_id": "A1"},
        evidence={"pages": ac.SOURCE_QUERY},
    )
    assert out.strategy == ac.STRATEGY_MIXED
    assert out.model_slots == ("pages",)
    assert out.system_slots == ("asset_id",)


# ── 4. MISSING / INVALID (no fabrication) ─────────────────────────────────────


def test_missing_when_a_required_system_slot_has_no_legal_value():
    out = pr.route(_params(asset_id=True), {"asset_id": _ASSET}, system_values={})
    assert out.strategy == ac.STRATEGY_MISSING
    assert out.readiness.unsatisfiable == ("asset_id",)
    assert out.model_slots == ()           # nothing is invented


def test_unavailable_required_slot_is_unsatisfiable():
    decl = {"x": ac.SlotDecl(ac.OWNERSHIP_UNAVAILABLE)}
    out = pr.route(_params(x=True), decl)
    assert out.strategy == ac.STRATEGY_MISSING
    assert out.readiness.unsatisfiable == ("x",)


def test_required_param_without_a_declaration_entry_is_missing():
    # §C: a required parameter with NO declaration entry has no legal source.
    out = pr.route(_params(x=True, pages=False), {"pages": _PAGES_Q})
    assert out.strategy == ac.STRATEGY_MISSING
    assert out.readiness.unsatisfiable == ("x",)
    assert out.model_slots == ()


def test_required_model_slot_with_disallowed_evidence_source_is_missing():
    # evidence exists but its source is NOT in the slot's allowed_sources, so no
    # legal source can be determined -> MISSING (never silently re-source it).
    out = pr.route(_params(pages=True), {"pages": _PAGES_Q},
                   evidence={"pages": ac.SOURCE_CONVERSATION_5_USER_TURNS})
    assert out.strategy == ac.STRATEGY_MISSING


# ── 6. the two dimensions are independent ─────────────────────────────────────


def test_required_readiness_and_acquisition_need_are_independent():
    ready_but_need = pr.route(
        _params(asset_id=True, pages=False),
        {"asset_id": _ASSET, "pages": _PAGES_Q},
        system_values={"asset_id": "A1"},
        evidence={"pages": ac.SOURCE_QUERY},
    )
    assert ready_but_need.readiness.ready is True
    assert ready_but_need.readiness.needs_acquisition is True
    assert ready_but_need.strategy == ac.STRATEGY_MIXED

    ready_no_need = pr.route(
        _params(asset_id=True, pages=False),
        {"asset_id": _ASSET, "pages": _PAGES_Q},
        system_values={"asset_id": "A1"},
        evidence={},
    )
    assert ready_no_need.readiness.ready is True
    assert ready_no_need.readiness.needs_acquisition is False


# ── 7 & 8. optional MODEL: schema presence is not evidence ────────────────────


def test_optional_model_without_evidence_triggers_no_acquisition():
    out = pr.route(_params(asset_id=True, pages=False),
                   {"asset_id": _ASSET, "pages": _PAGES_Q},
                   system_values={"asset_id": "A1"})
    assert out.strategy == ac.STRATEGY_CONTEXT_DIRECT
    assert out.readiness.needs_acquisition is False


def test_optional_model_with_evidence_triggers_acquisition():
    out = pr.route(_params(asset_id=True, pages=False),
                   {"asset_id": _ASSET, "pages": _PAGES_Q},
                   system_values={"asset_id": "A1"},
                   evidence={"pages": ac.SOURCE_QUERY})
    assert out.readiness.needs_acquisition is True
    assert out.strategy != ac.STRATEGY_CONTEXT_DIRECT


def test_evidence_whose_source_crosses_ownership_is_ignored():
    # a system source smuggled onto a MODEL slot is not a model evidence signal
    out = pr.route(_params(pages=False), {"pages": _PAGES_Q},
                   evidence={"pages": ac.SOURCE_UI_CONTEXT})
    assert out.readiness.needs_acquisition is False


# ── 9 & 10. context bundle window ─────────────────────────────────────────────


def test_context_bundle_keeps_only_the_last_five_user_messages():
    history = [{"role": "user", "content": f"u{i}"} for i in range(1, 8)]
    b = cb.build("now", history=history,
                 source=ac.SOURCE_CONVERSATION_5_USER_TURNS)
    assert b.user_turns == ("u3", "u4", "u5", "u6", "u7")
    assert b.query == "now"
    assert len(b.user_turns) == cb.LAST_N_USER_TURNS


def test_context_bundle_excludes_assistant_system_agent_roles():
    history = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "A"},
        {"role": "agent", "content": "G"},
        {"role": "user", "content": "u2"},
    ]
    b = cb.build("now", history=history,
                 source=ac.SOURCE_CONVERSATION_5_USER_TURNS)
    assert b.user_turns == ("u1", "u2")


def test_context_bundle_does_not_read_history_when_query_only():
    b = cb.build("now", history=[{"role": "user", "content": "u1"}],
                 source=ac.SOURCE_QUERY)
    assert b.user_turns == ()
    assert b.source == ac.SOURCE_QUERY


# ── 11 & 12. SYSTEM_BINDER / asset_id never enter the MODEL side ──────────────


def test_system_binder_slots_never_enter_the_model_acquisition_set():
    out = pr.route(_params(asset_id=True, pages=True),
                   {"asset_id": _ASSET, "pages": _PAGES_Q},
                   system_values={"asset_id": "A1"},
                   evidence={"pages": ac.SOURCE_QUERY})
    assert "asset_id" not in out.model_slots
    assert out.model_slots == ("pages",)


def test_asset_id_is_stripped_from_a_model_draft_by_merge():
    res = mg.merge(
        {"asset_id": "A1"},
        {"asset_id": "HALLUCINATED", "pages": "3"},
        declaration={"asset_id": _ASSET, "pages": _PAGES_Q},
        system_sources={"asset_id": ac.SOURCE_UI_CONTEXT},
        model_source=ac.SOURCE_QUERY,
    )
    assert res.args["asset_id"] == "A1"     # system value is truth
    assert res.args["pages"] == "3"
    by_slot = {p.slot: p for p in res.provenance}
    assert by_slot["asset_id"].produced_by == ac.OWNERSHIP_SYSTEM_BINDER
    assert by_slot["pages"].produced_by == ac.OWNERSHIP_MODEL


# ── 13. no fabrication ────────────────────────────────────────────────────────


def test_merge_never_fabricates_slots():
    res = mg.merge({"asset_id": "A1"}, {},
                   declaration={"asset_id": _ASSET, "pages": _PAGES_Q},
                   system_sources={"asset_id": ac.SOURCE_UI_CONTEXT},
                   model_source=ac.SOURCE_QUERY)
    assert res.args == {"asset_id": "A1"}
    assert {p.slot for p in res.provenance} == {"asset_id"}
    assert res.unprovenanced == ()


def test_merge_provenance_uses_the_actual_source_not_allowed_sources():
    # _ASSET allows (UI_CONTEXT, RESOLVER); the ACTUAL source given is RESOLVER.
    # allowed_sources[0] (UI_CONTEXT) must NOT be used as a fallback.
    res = mg.merge({"asset_id": "A1"}, {},
                   declaration={"asset_id": _ASSET, "pages": _PAGES_Q},
                   system_sources={"asset_id": ac.SOURCE_RESOLVER},
                   model_source=ac.SOURCE_QUERY)
    by_slot = {p.slot: p for p in res.provenance}
    assert by_slot["asset_id"].source == ac.SOURCE_RESOLVER


def test_merge_drops_a_system_value_with_no_actual_source():
    # A system value with NO actual source cannot form legal provenance -> it is
    # neither merged nor given a provenance record (never fabricated).
    res = mg.merge({"asset_id": "A1"}, {},
                   declaration={"asset_id": _ASSET, "pages": _PAGES_Q},
                   model_source=ac.SOURCE_QUERY)
    assert res.args == {}
    assert res.provenance == ()
    assert res.unprovenanced == ("asset_id",)


# ── 14. undeclared capability never falls back to the legacy extractor ────────


def test_undeclared_capability_yields_no_strategy_and_no_legacy_fallback():
    out = pr.route(_params(pages=True), {})
    assert out.declared is False
    assert out.strategy is None             # Agent, never a selection/extraction
    # the router does not even reference the legacy extractor
    assert "select_and_extract" not in inspect.getsource(pr)
