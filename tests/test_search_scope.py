"""The ONE deterministic search-scope rule (``cap_handler.search_scope``).

Every consumer of the per-turn ``r/<sub>`` decision — the runtime handler, the
dataset builder (``prepare_dataset``), the SFT scorer (``evaluate``) and the
offline Formal-500 scorer (``pe_score_v2``) — must read the SAME rule. This suite
pins that rule so no caller re-implements the regex or the slot narrowing.

Three scenarios (the confirmed contract):

* explicit ``r/<sub>`` scope   -> ``subreddit`` + ``platform=reddit`` are TOOL-ONLY
  exec args, ``platform`` is dropped from the model slots (effective ``{query}``);
* explicit named platform      -> no deterministic scope: the model owns
  ``{query, platform}``;
* generic social               -> no deterministic scope: the model owns
  ``{query, platform}`` and must OMIT ``platform`` (never ``auto``/reddit).

No model, no DB, no network: every assertion is pure.
"""
from __future__ import annotations

import pytest

from core.application.chat.intent_funnel.cap_handler.search_scope import (
    effective_model_slots,
    execution_scope_args,
    explicit_subreddit,
)

SOCIAL = "cap-social-search"
WEB = "cap-web-search"
RAG = "cap-rag-search"

# the static per-capability slot order each consumer supplies
SOCIAL_STATIC = ("query", "platform")
WEB_STATIC = ("query",)
RAG_STATIC = ("query", "domain")


# ── explicit_subreddit: the structured token, never a word/slash boundary ─────────


@pytest.mark.parametrize("message,expected", [
    ("r/MachineLearning 上关于 transformer 的讨论", "MachineLearning"),
    ("看看 r/learnpython 的帖子", "learnpython"),
    ("r/rust_lang 里怎么说", "rust_lang"),            # underscores allowed
])
def test_explicit_subreddit_reads_the_token(message, expected):
    assert explicit_subreddit(message) == expected


@pytest.mark.parametrize("message", [
    "社区里 car/road 的讨论",          # no ``r/`` boundary (leading ``r`` of a word)
    "reddit 上怎么说",                 # a platform NAME is not a structured scope
    "搜索关于 R/ 的内容",              # bare ``r/`` with no name
    "",                                # empty
    None,                              # type-hostile
])
def test_explicit_subreddit_ignores_non_scopes(message):
    assert explicit_subreddit(message) is None


# ── execution_scope_args: TOOL-ONLY, social-only ─────────────────────────────────


def test_scope_args_pin_and_fill_only_for_social():
    assert execution_scope_args(SOCIAL, "r/MachineLearning 的讨论") == {
        "subreddit": "MachineLearning", "platform": "reddit"}


@pytest.mark.parametrize("cap,raw", [
    (SOCIAL, "reddit 上搜一下 transformer"),   # a named platform is NOT a scope
    (SOCIAL, "大家怎么讨论 transformer"),       # generic
    (WEB, "r/MachineLearning 的帖子"),          # scope token but not social
    (RAG, "r/MachineLearning"),
])
def test_scope_args_empty_without_a_structured_social_scope(cap, raw):
    assert execution_scope_args(cap, raw) == {}


# ── effective_model_slots: the pin drops the slot; everything else is unchanged ───


def test_effective_slots_drops_the_pinned_slot_for_social():
    assert effective_model_slots(SOCIAL, "r/MachineLearning 的讨论", SOCIAL_STATIC) \
        == ("query",)


def test_effective_slots_keep_the_pair_without_a_scope():
    for raw in ("reddit 上搜一下 transformer", "大家怎么讨论 transformer", ""):
        assert effective_model_slots(SOCIAL, raw, SOCIAL_STATIC) == ("query", "platform")


@pytest.mark.parametrize("cap,raw,static,expected", [
    (WEB, "r/x 的帖子", WEB_STATIC, WEB_STATIC),
    (RAG, "r/x 的帖子", RAG_STATIC, RAG_STATIC),
])
def test_effective_slots_are_bytes_identical_for_non_social_caps(cap, raw, static,
                                                                expected):
    # the rule is a no-op for every capability without a deterministic scope, so a
    # non-social turn keeps its static slots unchanged (order preserved).
    assert effective_model_slots(cap, raw, static) == expected


def test_subreddit_is_never_an_effective_model_slot():
    # even when a caller passes a static list that (wrongly) contains subreddit, the
    # scope only ever DROPS slots — subreddit is never ADDED to the model face.
    slots = effective_model_slots(SOCIAL, "r/MachineLearning 的讨论",
                                  ("query", "platform"))
    assert "subreddit" not in slots


# ── single source: the runtime handler and the training contract agree ────────────


def test_handler_and_contract_read_the_same_rule():
    import sys
    from pathlib import Path
    scripts = (Path(__file__).resolve().parents[1] / "scripts"
               / "qwen_parameter_extraction_finetune")
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import pe_contract as C

    msg = "r/MachineLearning 上关于 transformer 的讨论"
    # the training contract's static slots narrowed by the ONE rule == what the
    # handler's slot_plan yields for the same turn.
    assert effective_model_slots(SOCIAL, msg, tuple(C.model_slots(SOCIAL))) == ("query",)
    # and the agreement holds for the no-scope case too.
    assert effective_model_slots(SOCIAL, "reddit 上搜一下", tuple(C.model_slots(SOCIAL))) \
        == ("query", "platform")
