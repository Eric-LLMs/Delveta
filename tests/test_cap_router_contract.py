"""Phase 2 contract pins (2026-10-01) — vocabulary + opt-in rule.

These tests pin the DECLARATION contract only; no behavior is exercised and the
cascade is not wired (``chat_cap_router_backend`` stays "off"). They lock the
decisions made across the Phase 1/2 review so a later refactor cannot silently
re-introduce a default SlotDecl or a wrong ownership/source pairing.
"""
from __future__ import annotations

import pytest

from core.application.chat.intent_funnel import cap_router
from core.application.chat.intent_funnel.argument_acquisition import (
    OWNERSHIP_MODEL,
    OWNERSHIP_SYSTEM_BINDER,
    OWNERSHIP_TOOL_DEFAULT,
    OWNERSHIP_UNAVAILABLE,
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
    SlotDecl,
    is_declared,
)
from core.config import settings


def test_backend_enum_and_default_off():
    assert cap_router.BACKENDS == ("off", "stub", "cap_router")
    assert settings.chat_cap_router_backend == "off"


def test_ownership_sources_subset_invariant():
    # QUERY is a MODEL-only source; UI_CONTEXT can never belong to MODEL.
    assert SOURCE_QUERY in SOURCES_OF[OWNERSHIP_MODEL]
    assert SOURCE_CONVERSATION_5_USER_TURNS in SOURCES_OF[OWNERSHIP_MODEL]
    assert SOURCE_UI_CONTEXT not in SOURCES_OF[OWNERSHIP_MODEL]
    assert SOURCE_UI_CONTEXT in SOURCES_OF[OWNERSHIP_SYSTEM_BINDER]
    assert SOURCE_RESOLVER in SOURCES_OF[OWNERSHIP_SYSTEM_BINDER]
    assert SOURCE_CALLBACK_CONTEXT in SOURCES_OF[OWNERSHIP_SYSTEM_BINDER]
    assert SOURCES_OF[OWNERSHIP_TOOL_DEFAULT] == frozenset({SOURCE_DEFAULT})
    assert SOURCES_OF[OWNERSHIP_UNAVAILABLE] == frozenset()
    # Every declared source is a real member of the source vocabulary.
    assert set().union(*SOURCES_OF.values()) <= SOURCES


def test_opt_in_rule_no_synthesized_default():
    # Ruling 2026-10-01: absence of arg_slots == legacy, and NO default SlotDecl.
    assert is_declared(None) is False
    assert is_declared({}) is False
    assert is_declared({"asset_id": {"ownership": "SYSTEM_BINDER"}}) is True
    assert not hasattr(SlotDecl, "default")


def test_slotdecl_ownership_is_required():
    with pytest.raises(TypeError):
        SlotDecl()  # ownership has no default on purpose
    assert SlotDecl(OWNERSHIP_SYSTEM_BINDER).allowed_sources == ()


def test_strategy_vocabulary():
    assert STRATEGY_CONTEXT_DIRECT in STRATEGIES
    # Strategy names are MODEL-AGNOSTIC (the extractor implementation is recorded
    # separately in telemetry): they name WHERE a value comes from, never WHO
    # extracts it. QUERY_TO_QWEN / QUERY_PLUS_5_USER_TURNS are retired names.
    assert {"CONTEXT_DIRECT", "QUERY_TO_EXTRACTOR", "QUERY_PLUS_5TURNS_TO_EXTRACTOR",
            "MIXED", "MISSING"} == STRATEGIES
