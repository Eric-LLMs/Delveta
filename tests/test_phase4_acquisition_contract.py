"""Phase 4 Step 0 — Argument Acquisition CONTRACT skeleton.

Step 0 locks the BUSINESS CONTRACT only. Nothing is implemented and nothing is
wired, so this file has two tiers:

1. GREEN — pins vocabulary that already EXISTS in
   :mod:`..argument_acquisition.contract` and that the approved contract depends
   on (ownerships, strategies, the I4 ownership->source invariant, `SlotDecl`
   carrying no schema, `Readiness`/`ArgumentProvenance` shape).
2. MARKER — asserts the written contract (`docs/phase4-acquisition-contract.md`)
   states the approved rules, so a later edit cannot silently drop one.

Future test cases (the Argument Path Router, the double gate, evidence, the Qwen
schema boundary, the undeclared-capability rule, the backend=off boundary) are
listed as a checklist in the contract doc — deliberately NOT implemented here as
`skip`ped tests, to avoid referencing modules (path_router / Qwen / orchestrator
wiring) that do not exist yet.

Business-Logic-First note: where the current code (e.g. the opt-in helper still
keyed on the LEGACY ``arg_slots``) disagrees with the approved contract (the NEW
``acquisition`` entry), the contract wins and the production change is DEFERRED
to Step 1 — it is NOT pinned as correct here.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.application.chat.intent_funnel.argument_acquisition import (
    OWNERSHIP_MODEL,
    OWNERSHIP_SYSTEM_BINDER,
    OWNERSHIP_TOOL_DEFAULT,
    OWNERSHIP_UNAVAILABLE,
    OWNERSHIPS,
    SOURCES,
    SOURCES_OF,
    SOURCE_CALLBACK_CONTEXT,
    SOURCE_CONVERSATION_5_USER_TURNS,
    SOURCE_DEFAULT,
    SOURCE_QUERY,
    SOURCE_RESOLVER,
    SOURCE_UI_CONTEXT,
    STRATEGIES,
    STRATEGY_CONTEXT_DIRECT,
    STRATEGY_MISSING,
    STRATEGY_MIXED,
    STRATEGY_QUERY_PLUS_5TURNS_TO_EXTRACTOR,
    STRATEGY_QUERY_TO_EXTRACTOR,
    ArgumentProvenance,
    Readiness,
    SlotDecl,
)

SPEC = Path(__file__).resolve().parents[1] / "docs" / "phase4-acquisition-contract.md"

# ── 1. GREEN: approved vocabulary that already exists ───────────────────────────


def test_ownerships_are_exactly_the_four_locked_values():
    assert OWNERSHIPS == frozenset({
        OWNERSHIP_MODEL, OWNERSHIP_SYSTEM_BINDER,
        OWNERSHIP_TOOL_DEFAULT, OWNERSHIP_UNAVAILABLE,
    })


def test_strategies_are_exactly_the_five_locked_values():
    assert STRATEGIES == frozenset({
        STRATEGY_CONTEXT_DIRECT, STRATEGY_QUERY_TO_EXTRACTOR,
        STRATEGY_QUERY_PLUS_5TURNS_TO_EXTRACTOR, STRATEGY_MIXED, STRATEGY_MISSING,
    })


def test_i4_every_ownership_source_subset_of_known_sources():
    # Invariant I4 (contract §C): a slot's allowed_sources can only be sources
    # its ownership may legitimately draw from — one table, no drift.
    assert set(SOURCES_OF) == set(OWNERSHIPS)
    for ownership, sources in SOURCES_OF.items():
        assert sources <= SOURCES, (ownership, sources - SOURCES)


def test_model_may_only_draw_from_query_and_history():
    assert SOURCES_OF[OWNERSHIP_MODEL] == frozenset({
        SOURCE_QUERY, SOURCE_CONVERSATION_5_USER_TURNS})


def test_system_binder_never_shares_a_source_with_model():
    # contract §C: SYSTEM_BINDER draws from UI_CONTEXT/RESOLVER/CALLBACK_CONTEXT
    # only — the two source sets are disjoint, so a slot can never be both.
    sb = SOURCES_OF[OWNERSHIP_SYSTEM_BINDER]
    assert sb == frozenset({
        SOURCE_UI_CONTEXT, SOURCE_RESOLVER, SOURCE_CALLBACK_CONTEXT})
    assert SOURCES_OF[OWNERSHIP_MODEL].isdisjoint(sb)


def test_tool_default_and_unavailable_sources():
    assert SOURCES_OF[OWNERSHIP_TOOL_DEFAULT] == frozenset({SOURCE_DEFAULT})
    assert SOURCES_OF[OWNERSHIP_UNAVAILABLE] == frozenset()


def test_slotdecl_requires_ownership_and_carries_no_schema():
    # ownership is REQUIRED (no default); the declaration carries NO
    # required/type/enum/description — those stay in Registry.parameters (§C).
    SlotDecl(ownership=OWNERSHIP_MODEL)  # does not raise
    with pytest.raises(TypeError):
        SlotDecl()  # type: ignore[call-arg]
    for forbidden in ("required", "type", "enum", "description", "max_len"):
        assert not hasattr(SlotDecl(ownership=OWNERSHIP_MODEL), forbidden)


def test_readiness_defaults_are_the_inert_state():
    r = Readiness()
    assert r.ready is False
    assert r.needs_acquisition is False
    assert r.unsatisfiable == ()


def test_argument_provenance_shape():
    p = ArgumentProvenance(slot="asset_id", source=SOURCE_UI_CONTEXT,
                           produced_by=OWNERSHIP_SYSTEM_BINDER)
    assert (p.slot, p.source, p.produced_by) == (
        "asset_id", SOURCE_UI_CONTEXT, OWNERSHIP_SYSTEM_BINDER)


# ── 2. MARKER: the written contract states the approved rules ───────────────────


def test_spec_document_exists_and_is_specification_only():
    assert SPEC.is_file(), f"missing contract spec: {SPEC}"
    text = SPEC.read_text(encoding="utf-8").lower()
    # Step 0 must not have started implementing.
    assert "nothing implemented" in text or "specification only" in text


def test_spec_locks_the_new_declaration_and_forbids_legacy_arg_slots():
    text = SPEC.read_text(encoding="utf-8").lower()
    assert "acquisition" in text
    assert "arg_slots" in text          # named explicitly as the LEGACY thing
    assert "must not reuse" in text


def test_spec_locks_system_binder_never_reaches_qwen_and_asset_id():
    text = SPEC.read_text(encoding="utf-8").lower()
    assert "system_binder" in text
    assert "asset_id" in text
    assert "never produced by qwen" in text


def test_spec_locks_the_five_strategies_and_the_double_gate():
    text = SPEC.read_text(encoding="utf-8").lower()
    for s in ("context_direct", "query_to_extractor",
              "query_plus_5turns_to_extractor", "mixed", "missing"):
        assert s in text
    assert "double gate" in text
    assert "required readiness" in text
    assert "acquisition need" in text


def test_spec_locks_the_hit_invariant_and_backend_off_boundary():
    text = SPEC.read_text(encoding="utf-8").lower()
    assert "match_hit" in text
    assert "no re-selection" in text or "not re-select" in text
    assert "backend=off" in text
    assert "acquisition-undeclared" in text   # the approved undeclared-reason semantics
    assert "cap_router_hit_deferred" in text   # named as transitional only


def test_spec_locks_the_optional_model_evidence_rule():
    text = SPEC.read_text(encoding="utf-8").lower()
    assert "evidence" in text
    assert "deterministic" in text
    assert "no llm" in text or "never delegated to an llm" in text


def test_spec_rejects_linear_strategy_precedence():
    # The resolved strategy determination is two-dimensional (Required Readiness,
    # then Acquisition Need) — NOT a global linear precedence ladder.
    text = SPEC.read_text(encoding="utf-8").lower()
    assert "required readiness" in text
    assert "acquisition need" in text
    assert "no linear precedence" in text


def test_spec_defines_no_incomplete_declaration_state():
    # Step 0 defines only the opt-in contract + undeclared -> Agent; no extra
    # "incomplete declaration" status or reason is introduced.
    text = SPEC.read_text(encoding="utf-8").lower()
    assert "incomplete declaration" in text


# ── 3. MARKER: follow-up review rulings #1/#3/#4/#5 are pinned ──────────────────


def test_spec_model_slot_source_is_evidence_driven_not_allowed_sources():
    # Ruling #3: WHERE a MODEL slot's value comes from is decided by the ACTUAL
    # evidence signal, never inferred from what the declaration merely allows.
    text = SPEC.read_text(encoding="utf-8").lower()
    assert "evidence-driven" in text
    assert "never inferred from `allowed_sources`" in text


def test_spec_required_param_without_declaration_is_missing_not_legacy_fallback():
    # Ruling #1: a required parameter with NO declaration entry has no legal
    # source -> MISSING/INVALID, never a legacy fallback.
    text = SPEC.read_text(encoding="utf-8").lower()
    assert "with no declaration entry" in text
    assert "no legal acquisition" in text


def test_spec_evidence_is_a_router_input_with_detector_deferred_to_step_2():
    # Ruling #4: evidence is an injected external input with TWO uses; Step 1
    # implements NO detector — the producer + wiring land in Step 2.
    text = SPEC.read_text(encoding="utf-8").lower()
    assert "router input" in text
    assert "step 2" in text


def test_spec_provenance_records_actual_source_not_allowed_sources_fallback():
    # Ruling #5: provenance.source is the ACTUAL source; allowed_sources[0] is
    # never a legal provenance fallback.
    text = SPEC.read_text(encoding="utf-8").lower()
    assert "provenance" in text
    assert "actual" in text
    assert "allowed_sources[0]" in text
