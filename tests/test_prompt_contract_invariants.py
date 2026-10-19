"""P1 — the PE contract invariants (interface-only prompt + slot filtering).

This suite pins the UNIFIED CONTRACT: training, production inference and scoring
share ONE input face (tool name + description + slot schema) + ONE output face
(native tool call), and the prompt carries NO semantic rule. No model, no DB, no
endpoint: every assertion reads the PRODUCTION code and the stdlib-only shared
snapshot (``pe_contract``), so it runs anywhere.

What is pinned here:

* C1 — no capability's ``pe_contract.model_slots`` contains a non-model slot
  (``top_k`` / ``limit`` / ``subreddit``), while the full registry params DO;
* C2 — the final output function schema filters those non-model slots out even
  when the Registry row still carries them (the structural guarantee);
* C3 — the training scorer (``evaluate.score_sample``) treats them as ILLEGAL,
  so they can never enter scoring;
* the prompt is INTERFACE-ONLY: every capability uses the same shared prompt and
  that prompt carries no strip / rewrite / frame-removal / count-removal /
  container-strip rule (the model learns behaviour from the SFT data);
* the §2.1 verbatim query rule lives ONLY as the interface fact in the slot
  description (``pe_contract._SEARCH_QUERY_DESC``), which reaches the training
  tool schema, the user slot block AND the final output schema — the count-rule
  wording is retired;
* the ``r/<sub>`` scope reaches the tool deterministically as a non-PE
  ``subreddit`` arg.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from core.application.chat.intent_funnel.argument_acquisition.extractor import (
    SYSTEM_EXTRACT,
    _function_schema,
)
from core.application.chat.intent_funnel.argument_acquisition.prompts import prompt_for
from core.application.chat.intent_funnel.cap_handler.social_search_handler import (
    SocialSearchHandler,
)

# ``pe_contract`` / ``evaluate`` are the stdlib-only shared contract + scorer
# under fine_tuning/ (no production import), so this test reads the SAME snapshot
# the builder / verifier / trainer read.
_SCRIPTS = (Path(__file__).resolve().parents[1] / "fine_tuning"
            / "qwen_parameter_extraction_finetune")
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import evaluate as E  # noqa: E402
import pe_contract as C  # noqa: E402

CAPS = ["cap-web-search", "cap-rag-search", "cap-social-search",
        "cap-translate", "cap-add-term", "cap-create-folder"]
SEARCH_CAPS = ["cap-web-search", "cap-rag-search", "cap-social-search"]

# the slots that are REAL backend parameters but never model-owned
NON_MODEL_SLOTS = {"top_k", "limit", "subreddit"}


def _entry(cid: str):
    """A Registry-shaped entry whose ``parameters`` are the FULL registry table —
    i.e. it still carries ``top_k`` / ``limit`` / ``subreddit`` where present.
    This is the honest worst case: only the ``model_slots`` filter may exclude
    them."""
    return types.SimpleNamespace(
        capability_id=cid, tool_binding=C.tool_name(cid),
        description=C.function_description(cid),
        parameters=C.SCHEMA[cid]["params"],
    )


# ── C1: the model-slot set never contains a non-model slot ────────────────────────


@pytest.mark.parametrize("cid", CAPS)
def test_model_slots_never_contain_a_non_model_slot(cid):
    assert set(C.model_slots(cid)) & NON_MODEL_SLOTS == set()


def test_the_non_model_slots_are_actually_present_in_the_registry_params():
    """The exclusion above is a real FILTER, not an absence: the full registry
    params DO carry these keys, so C1/C2 are load-bearing."""
    web = C.SCHEMA["cap-web-search"]["params"]
    rag = C.SCHEMA["cap-rag-search"]["params"]
    social = C.SCHEMA["cap-social-search"]["params"]
    assert "top_k" in web and "top_k" in rag
    assert {"subreddit", "limit"} <= set(social)


# ── C2: the final output schema filters non-model slots, whatever the Registry row ─


@pytest.mark.parametrize("cid", CAPS)
def test_output_schema_excludes_non_model_slots_even_when_registry_carries_them(cid):
    model_slots = C.model_slots(cid)
    schema = _function_schema(_entry(cid), model_slots)
    assert list(schema["properties"]) == model_slots        # model_slots order only
    assert set(schema["properties"]) & NON_MODEL_SLOTS == set()
    assert set(schema["required"]) <= set(model_slots)


# ── C3: the scorer treats non-model slots as ILLEGAL (never scored) ───────────────


def test_scorer_flags_top_k_as_illegal_for_a_search_cap():
    rec = E.score_sample("cap-web-search", {"query": "注意力机制"}, "cap-web-search",
                         {"query": "注意力机制", "top_k": 3})
    # the stray key is surfaced as illegal; the slot-wise compare (exact_match)
    # only ever looks at model slots, so it is reported through illegal_slot_rate.
    assert rec["illegal_keys"] == ["top_k"]
    assert rec["exact_match"] is True


def test_scorer_flags_subreddit_and_limit_as_illegal_for_social():
    rec = E.score_sample("cap-social-search", {"query": "x", "platform": "reddit"},
                         "cap-social-search",
                         {"query": "x", "platform": "reddit",
                          "subreddit": "MachineLearning", "limit": 5})
    assert rec["illegal_keys"] == ["limit", "subreddit"]


def test_scorer_scope_is_exactly_the_model_slots():
    # A clean prediction over the model slots alone scores as a full match.
    rec = E.score_sample("cap-rag-search", {"query": "q", "domain": "工程"},
                         "cap-rag-search", {"query": "q", "domain": "工程"})
    assert rec["illegal_keys"] == [] and rec["exact_match"] is True


# ── the prompt is INTERFACE-ONLY (no per-capability rule) ─────────────────────────


@pytest.mark.parametrize("cid", CAPS)
def test_every_capability_uses_the_shared_interface_prompt(cid):
    # TRAINING and PRODUCTION carry the SAME prompt for every capability: the
    # extractor default. There is no per-capability prose prompt any more.
    assert prompt_for(cid) == SYSTEM_EXTRACT


def test_interface_prompt_carries_no_semantic_rule():
    """The system prompt states the INTERFACE + output discipline only. Every
    semantic-cleaning rule the old prompts taught is gone."""
    low = SYSTEM_EXTRACT.lower()
    for banned in ("verbatim", "translate", "summarize", "paraphrase", "reformat",
                   "container", "result-count", "execution-instruction frame",
                   "top_k", "limit", "subreddit", "strip", "vocabulary",
                   "词库", "词汇库"):
        assert banned not in low, f"interface prompt still teaches {banned!r}"
    # the output discipline is still stated (that is an interface requirement).
    assert "arguments" in low


# ── the verbatim query rule lives ONLY in the slot description ────────────────────


def test_search_query_desc_is_the_verbatim_interface_fact():
    """``pe_contract._SEARCH_QUERY_DESC`` is the frozen desc string that feeds the
    training tool Schema, the user slot block AND the offline scoring snapshot. It
    states WHICH text to extract (the search topic) and that it is kept VERBATIM —
    and teaches NO count / frame-removal rule."""
    d = C._SEARCH_QUERY_DESC
    assert "VERBATIM" in d
    assert "do not rewrite, translate, summarize, expand or paraphrase" in d
    # the retired count rule and frame-removal wording are gone.
    assert "result-count" not in d
    assert "part of the TOPIC" not in d
    assert "execution-instruction frame" not in d


@pytest.mark.parametrize("cid", SEARCH_CAPS)
def test_verbatim_desc_reaches_the_training_and_output_schema_faces(cid):
    """The identical desc string is rendered onto the query slot by
    ``pe_contract.model_schema`` (training / user slot block) AND by
    ``extractor._function_schema`` (the final output Schema the prod evaluator
    builds) — so the model reads ONE wording everywhere."""
    assert C.SCHEMA[cid]["params"]["query"]["description"] == C._SEARCH_QUERY_DESC
    contract_face = C.model_schema(cid)["properties"]["query"]["description"]
    assert contract_face == C._SEARCH_QUERY_DESC
    output_face = _function_schema(_entry(cid), C.model_slots(cid)) \
        ["properties"]["query"]["description"]
    assert output_face == C._SEARCH_QUERY_DESC


def test_no_capability_prompt_teaches_container_stripping():
    """The retired ``add-term`` container-strip prose must not reappear anywhere
    the model reads."""
    for cid in CAPS:
        p = prompt_for(cid)
        assert "TOPIC name ONLY" not in p
        assert "词汇库" not in p and "container" not in p.lower()


# ── §10.4: an explicit r/<sub> scope reaches the tool deterministically ───────────


def test_social_handler_slot_plan_is_the_frozen_pair():
    """The working-tree contract: ``platform`` is the only social semantic slot,
    ``subreddit`` / ``limit`` are NOT model-owned."""
    h = SocialSearchHandler()
    msg = "在 reddit 上搜一下 transformer"
    plan = h.slot_plan(query=msg, facts=None, draft={"query": msg})
    assert tuple(plan.model_slots) == ("query", "platform")
    assert plan.default_slots == ()


async def test_social_subreddit_scope_reaches_the_tool():
    """An explicit ``r/<sub>`` scope reaches the tool DETERMINISTICALLY: the scope
    rides the tool-only ``subreddit`` arg, ``platform`` is pinned to reddit and
    neither is a PE model slot (so the model is still scored on ``query`` alone)."""
    h = SocialSearchHandler()
    msg = "r/MachineLearning 上关于 transformer 的讨论"
    draft = await h.acquire(query=msg, facts=None)
    plan = h.slot_plan(query=msg, facts=None, draft=draft)
    assert draft.get("subreddit") == "MachineLearning"   # deterministic tool arg
    assert draft.get("platform") == "reddit"             # scope pinned, never widened
    assert "subreddit" not in plan.model_slots           # NEVER a PE slot
    assert plan.model_slots == ("query",)                # platform not re-owned either
