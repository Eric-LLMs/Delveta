"""PE scoring v2 (``scripts/qwen_parameter_extraction_finetune/pe_score_v2``) — the
per-sample, contract-aligned 口径, on synthetic records.

The v2 scorer is the CURRENT versioned standard (the legacy ``eval_pilot.score`` is
frozen for historical reproduction). These tests pin its three obligations:

* PE Argument EM is scored over each turn's EFFECTIVE model slots — an explicit
  ``r/<sub>`` turn scores over ``{query}`` only (never credited / penalised for the
  deterministically pinned ``platform`` or the tool-only ``subreddit``);
* the deterministic EXECUTION parameters (``subreddit`` + the ``reddit`` platform
  pin) are validated SEPARATELY and never folded into PE Argument EM;
* a model-emitted key outside the effective slots is surfaced as a non-model
  emission.

No model, no DB, no network, no file reads: every record here is synthetic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SCRIPTS = (Path(__file__).resolve().parents[1] / "scripts"
            / "qwen_parameter_extraction_finetune")
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import pe_score_v2 as V  # noqa: E402


def _rec(cap, query, gold, pred):
    return {"id": "t", "capability_id": cap, "query": query,
            "gold_arguments": gold, "pred_arguments": pred}


# ── scenario A: explicit r/<sub> — effective slots == {query} ─────────────────────


def test_explicit_subreddit_scores_over_query_only():
    rec = _rec("cap-social-search",
               "r/MachineLearning 上关于 transformer 的讨论",
               {"query": "transformer 的讨论", "platform": "reddit",
                "subreddit": "MachineLearning", "limit": 5},
               {"query": "transformer 的讨论"})
    s = V.score_record(rec)
    assert s["pe"]["effective_slots"] == ["query"]
    assert s["pe"]["pe_em"] is True
    assert s["pe"]["non_model_emitted"] == []


def test_explicit_subreddit_exec_is_scored_separately():
    rec = _rec("cap-social-search", "r/MachineLearning 的讨论",
               {"query": "讨论", "platform": "reddit", "subreddit": "MachineLearning"},
               {"query": "讨论"})
    ex = V.score_record(rec)["exec"]
    assert ex["has_scope"] is True and ex["ok"] is True
    assert ex["checks"]["subreddit"] == {
        "expected": "MachineLearning", "gold": "MachineLearning", "ok": True}
    assert ex["checks"]["platform"] == {
        "expected": "reddit", "gold": "reddit", "ok": True}


def test_wrong_exec_scope_does_not_touch_pe_em():
    """A correct PE (query alone) still EM-passes even when the GOLD scope differs —
    the scope error is reported under the exec metric, not hidden inside PE."""
    rec = _rec("cap-social-search", "r/MachineLearning 的讨论",
               {"query": "讨论", "platform": "reddit", "subreddit": "learnpython"},
               {"query": "讨论"})
    s = V.score_record(rec)
    assert s["pe"]["pe_em"] is True
    assert s["exec"]["ok"] is False           # the scope is wrong, reported apart


def test_model_emitted_subreddit_is_a_non_model_emission():
    rec = _rec("cap-social-search", "r/MachineLearning 的讨论",
               {"query": "讨论", "platform": "reddit", "subreddit": "MachineLearning"},
               {"query": "讨论", "subreddit": "MachineLearning"})
    assert V.score_record(rec)["pe"]["non_model_emitted"] == ["subreddit"]


# ── scenario B: explicit platform — the model owns {query, platform} ──────────────


def test_named_platform_keeps_the_pair_and_has_no_exec_scope():
    rec = _rec("cap-social-search", "在 reddit 上搜一下 transformer",
               {"query": "transformer", "platform": "reddit"},
               {"query": "transformer", "platform": "reddit"})
    s = V.score_record(rec)
    assert s["pe"]["effective_slots"] == ["query", "platform"]
    assert s["pe"]["pe_em"] is True
    assert s["exec"]["has_scope"] is False and s["exec"]["ok"] is None


def test_missing_named_platform_is_a_pe_miss():
    rec = _rec("cap-social-search", "在 reddit 上搜一下 transformer",
               {"query": "transformer", "platform": "reddit"},
               {"query": "transformer"})
    assert V.score_record(rec)["pe"]["pe_em"] is False


# ── scenario C: generic social — the model must OMIT platform ─────────────────────


def test_generic_social_omits_platform_and_passes():
    rec = _rec("cap-social-search", "大家怎么讨论 transformer",
               {"query": "transformer"}, {"query": "transformer"})
    s = V.score_record(rec)
    assert s["pe"]["effective_slots"] == ["query", "platform"]   # static, nothing pinned
    assert s["pe"]["pe_em"] is True


def test_generic_social_that_invents_reddit_is_a_pe_miss():
    """The retired bug: a generic request defaulted to reddit. Under v2 an invented
    ``platform`` makes the turn a PE miss (it is a model slot, not a non-model key)."""
    rec = _rec("cap-social-search", "大家怎么讨论 transformer",
               {"query": "transformer"}, {"query": "transformer", "platform": "reddit"})
    s = V.score_record(rec)
    assert s["pe"]["pe_em"] is False
    assert s["pe"]["non_model_emitted"] == []      # platform IS an effective slot


# ── web / rag are untouched: no scope, static slots ───────────────────────────────


@pytest.mark.parametrize("cap,query,slot,gold,pred,em", [
    ("cap-web-search", "r/MachineLearning 的帖子", "query",
     {"query": "帖子"}, {"query": "帖子"}, True),
    ("cap-rag-search", "r/x 的资料", "domain",
     {"query": "资料", "domain": "工程"}, {"query": "资料", "domain": "工程"}, True),
])
def test_non_social_caps_keep_static_slots(cap, query, slot, gold, pred, em):
    s = V.score_record(_rec(cap, query, gold, pred))
    assert slot in s["pe"]["effective_slots"]
    assert s["pe"]["pe_em"] is em
    assert s["exec"]["has_scope"] is False


# ── aggregation surfaces the two metrics apart ────────────────────────────────────


def test_aggregate_reports_exec_scope_apart_from_pe_em():
    recs = [
        _rec("cap-social-search", "r/MachineLearning 的讨论",
             {"query": "讨论", "platform": "reddit", "subreddit": "MachineLearning"},
             {"query": "讨论"}),
        _rec("cap-web-search", "查一下注意力机制", {"query": "注意力机制"},
             {"query": "注意力机制"}),
    ]
    agg = V.aggregate([V.score_record(r) for r in recs])
    ov = agg["overall"]
    assert ov["samples"] == 2
    assert ov["pe_em"] == 1.0
    assert ov["exec_scope_samples"] == 1 and ov["exec_scope_correct"] == 1.0


# ── the offline-reference schema snapshot is NOT the live Registry ────────────────


def test_schema_snapshot_is_labelled_offline_not_live():
    snap = V.schema_snapshot()
    assert snap["kind"] == "offline_reference_contract_snapshot"
    assert snap["live_registry"] is False
    assert snap["source"].endswith("pe_contract.py")
    assert V.snapshot_sha256() == V.snapshot_sha256()          # deterministic
    assert "cap-social-search" in snap["capabilities"]
