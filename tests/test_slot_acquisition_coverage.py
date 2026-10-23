"""Unified slot acquisition — the rule-5 COVERAGE guard (all chat-exposed caps).

The acquisition waterfall has to hold for EVERY capability the Chat plane can
route, not just the ones that happen to have a handler today. This module pins
that contract as executable code:

1. no natural-language extraction rule runs in a handler — the search trio's
   ``acquire()`` copies the turn's sentence VERBATIM as the ``query`` fallback
   (the model may clean it; an explicit ``r/<sub>`` token is the one DETERMINISTIC
   exception and is not a model slot), and the semantic handlers (add-term /
   create-folder / translate) return the EMPTY ``{}`` draft;
2. the model is authorized ONLY for genuinely semantic slots (search ``query``
   cleaning + the named platform, a rag ``domain`` scope, an add-term
   ``term``/``domain``, a create-folder ``name``, a translate ``text`` and the
   optional ``target_language``) — NOT the result count (``top_k`` / ``limit``) or
   ``subreddit``;
3. an optional slot with no value falls to the tool's own legal default downstream;
4. every exposed capability is accounted for by an EXPLICIT acquisition policy:
   handler-owned (RULE_CONTEXT / MODEL) or DEFERRED_AGENT. A capability with no
   policy and no handler is a policy gap, never an implicit no-args capability.

The exposed set is the Registry's ``is_intent_funnel_capability_allowed``
predicate under the shipped config: everything enabled+active+non-research minus
the chat-hidden ``cap-edit-file`` / ``cap-artifact`` / ``cap-bash`` (the research
lane owns ``cap-research``). A new exposed capability that is neither wired nor
declared fail-open fails this test.
"""
from __future__ import annotations

import pytest

from core.application.chat.intent_funnel.cap_handler import (
    HANDLERS,
    POLICIES,
    STRATEGY_DEFERRED_AGENT,
    STRATEGY_MIXED,
    STRATEGY_MODEL,
    STRATEGY_RULE_CONTEXT,
    SlotPlanningHandler,
    handler_for,
)

# The chat-exposed set: "handler" = owned by a CapabilityHandler; "deferred" =
# admitted but its real tool contract cannot be satisfied on this path, so the
# turn exits to the Agent with an explicit DEFERRED policy.
EXPOSED: dict[str, str] = {
    "cap-add-term": "handler",
    "cap-create-folder": "handler",
    "cap-rag-search": "handler",
    "cap-web-search": "handler",
    "cap-social-search": "handler",
    "cap-vision": "handler",
    "cap-read-document": "handler",
    "cap-read-file": "handler",
    "cap-pdf-extract-text": "handler",
    "cap-pdf-table-to-text": "handler",
    "cap-translate": "handler",
    "cap-mindmap": "deferred",
    "cap-slides": "deferred",
    "cap-summary": "deferred",
}

EXPOSED_HANDLER_IDS = frozenset(cid for cid, k in EXPOSED.items() if k == "handler")

# The ONLY capabilities that opt into model-assisted slots (rule 2): the search
# trio, plus add-term (term/domain), create-folder (name) and translate
# (text/target_language) — every natural-language slot is MODEL-owned; no DET
# extraction rule survives.
SEARCH_TRIO = frozenset({"cap-web-search", "cap-rag-search", "cap-social-search"})
MODEL_OPT_IN = SEARCH_TRIO | frozenset(
    {"cap-add-term", "cap-create-folder", "cap-translate"})


def _hidden_ids() -> frozenset[str]:
    from core.config import settings

    raw = str(getattr(settings, "chat_funnel_hidden_capabilities", "") or "")
    return frozenset(t.strip() for t in raw.split(",") if t.strip())


# ── rule 5: every exposed capability is accounted for ────────────────────────────


def test_roster_covers_exactly_the_exposed_handler_set():
    assert set(HANDLERS) == set(EXPOSED_HANDLER_IDS)


@pytest.mark.parametrize("cid,kind", sorted(EXPOSED.items()))
def test_each_exposed_capability_has_its_declared_disposition(cid, kind):
    if kind == "handler":
        assert handler_for(cid) is not None
    else:
        assert handler_for(cid) is None  # deliberately deferred -> the Agent


def test_every_exposed_capability_has_an_explicit_policy():
    assert set(POLICIES) == set(EXPOSED)


@pytest.mark.parametrize("cid,kind", sorted(EXPOSED.items()))
def test_policy_matches_the_declared_disposition(cid, kind):
    policy = POLICIES[cid]
    if kind == "handler":
        assert policy.strategy in (
            STRATEGY_MODEL, STRATEGY_MIXED, STRATEGY_RULE_CONTEXT)
        assert (policy.strategy in (STRATEGY_MODEL, STRATEGY_MIXED)) == (
            cid in MODEL_OPT_IN)
    else:
        assert policy.strategy == STRATEGY_DEFERRED_AGENT
        assert policy.detail        # the contract gap is documented


def test_social_search_is_the_explicit_mixed_capability():
    # MIXED is not a synonym for MODEL: it declares that the handler owns a
    # deterministic slot (an explicit r/<name> scope) alongside MODEL slots.
    assert POLICIES["cap-social-search"].strategy == STRATEGY_MIXED
    assert all(
        pol.strategy != STRATEGY_MIXED
        for cid, pol in POLICIES.items() if cid != "cap-social-search"
    )


def test_every_handler_capability_declares_a_policy():
    assert set(HANDLERS) <= set(POLICIES)


def test_no_handler_owned_capability_is_hidden_from_chat():
    # A hidden cap is invisible to every Intent Funnel consumer; a handler owned
    # by one would be dead code. The shipped hidden set must not intersect the roster.
    assert set(HANDLERS).isdisjoint(_hidden_ids())


def test_only_the_model_assisted_caps_opt_into_slot_plans():
    for cid, handler in HANDLERS.items():
        assert isinstance(handler, SlotPlanningHandler) == (cid in MODEL_OPT_IN), cid


# ── rule 2 + rule 3: the model owns the semantic slots, NOT the count ────────────


@pytest.mark.parametrize("cid,msg,count_slot", [
    ("cap-web-search", "search for transformers, top 3", "top_k"),
    ("cap-rag-search", "查一下注意力机制，前 5 条", "top_k"),
    ("cap-social-search", "搜索社区讨论，取 3 条", "limit"),
])
async def test_stated_count_is_not_a_model_slot(cid, msg, count_slot):
    handler = handler_for(cid)
    draft = await handler.acquire(query=msg, facts=None)
    assert count_slot not in draft                    # rule 1: NO DET extraction
    plan = handler.slot_plan(query=msg, facts=None, draft=draft)
    assert count_slot not in plan.model_slots         # rule 2: count is NOT model-owned
    assert count_slot not in plan.default_slots       # it is the tool's own default
    assert "query" in plan.model_slots                # rule 2: semantic query cleaning


@pytest.mark.parametrize("cid", sorted(SEARCH_TRIO))
async def test_unstated_count_falls_to_the_tool_default(cid):
    msg = "帮我搜一下注意力机制"                        # no count stated
    handler = handler_for(cid)
    draft = await handler.acquire(query=msg, facts=None)
    count_slot = "limit" if cid == "cap-social-search" else "top_k"
    assert count_slot not in draft                    # rule 1: no rule value
    plan = handler.slot_plan(query=msg, facts=None, draft=draft)
    assert count_slot not in plan.model_slots         # never a model slot
    assert count_slot not in plan.default_slots       # the TOOL applies its default (5)
