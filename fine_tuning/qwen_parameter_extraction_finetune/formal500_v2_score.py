"""formal500_v2_score — the VERSIONED scorer for the Formal-500 v2 benchmark.

This is a NEW, independent scorer. It does NOT change and does NOT replace the
legacy Formal-500 evaluator (``logs/_e2e_virtual_metrics.py`` + the frozen
``logs/_argbench/ownership.py``), which stays byte-for-byte reproducible for the
historical v1 numbers. This module is the CURRENT口径, aligned to the current
parameter-extraction contract.

Two separations the legacy scorer conflated
-------------------------------------------
1. **PE model slots vs everything else.** PE scoring runs ONLY over the slots the
   extractor model is actually asked for (``formal500_ownership.pe_model_slots``
   = ``pe_contract.model_slots`` for the six MODEL capabilities, the handler /
   declaration truth for the rest). A handler value (``question``), a settled
   context value (``pages``), a function default (``top_k`` / ``limit`` /
   ``subreddit``) or an undeclared slot (``cap-slides.count``) is NEVER folded
   into a PE score.
2. **The complete Tool contract.** The full expected tool arguments
   (``expected_tool_arguments`` = PE slots + handler/context/system slots) are
   scored on their OWN metric, and each slot's PRODUCER (ownership bucket) is
   reported, so a non-PE miss can never hide inside a PE number.

The eight metrics (each states its numerator / denominator / match rule)
------------------------------------------------------------------------
M1 tool_selection          num: pred capability == gold capability      den: all cases
M2 pe_model_strict         num: restricted pred == gold PE args (exact) den: cases with >=1 PE slot
M3 pe_model_lenient        num: gold PE slot value matches pred         den: gold PE slot instances
M4 binder_pass             num: binder_state == COMPLETE                den: cases that reached binding
M5 virtual_exec_reached    num: virtual_executor_called is true         den: all cases
M6 stage_failures          histogram of fallback_reason / deepest_stage (counts, no denominator)
M7 tool_contract_match     num: pred == expected_tool_arguments (exact) den: cases with a non-empty expectation
M8 non_pe_slot_sources     per-ownership-bucket supply/match counts (the non-PE side of the contract)

The dataset version and the scorer version are recorded on every report.
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import formal500_ownership as OWN  # noqa: E402

SCORER_VERSION = "formal500_v2_score/1"
DATASET_VERSION = "formal500-v2"


# ── canonical value comparison (same spirit as the live Binder's equality) ────
def _canon(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        # 5 == 5.0 for a stated count; a string "5" is NOT equal to int 5.
        return float(value)
    if isinstance(value, list):
        return [_canon(v) for v in value]
    if isinstance(value, dict):
        return {k: _canon(v) for k, v in sorted(value.items())}
    return value


def _eq(a, b) -> bool:
    return _canon(a) == _canon(b)


def _restrict(args: dict, slots) -> dict:
    return {s: args[s] for s in slots if s in (args or {})}


# ── prediction normalization ─────────────────────────────────────────────────
def from_execution_trace(trace_row: dict) -> dict:
    """Map one ``execution_trace.jsonl`` record to the scorer's normalized shape.

    The execution trace nests everything under ``stages``; the scorer reads only
    the fields it scores, so this adapter is the ONE place the trace schema is
    touched.
    """
    stages = trace_row.get("stages") or {}
    return {
        "id": trace_row.get("case_id") or trace_row.get("id"),
        "pred_capability_id": stages.get("decided_capability"),
        "pred_arguments": dict(stages.get("model_values") or {}),
        "binder_state": stages.get("binder_state"),
        "virtual_executor_called": bool(stages.get("virtual_executor_called")),
        "fallback_reason": stages.get("fallback_reason"),
        "deepest_stage": stages.get("deepest_stage"),
        "plan_kind": stages.get("plan_kind"),
    }


# ── the eight metrics ────────────────────────────────────────────────────────
def _ratio(num: int, den: int):
    return {"numerator": num, "denominator": den,
            "rate": (round(num / den, 6) if den else None)}


def score(dataset: list[dict], predictions: list[dict]) -> dict:
    """Score ``predictions`` against the v2 ``dataset``.

    ``predictions`` may be raw ``execution_trace`` rows (they are normalized
    through :func:`from_execution_trace`) or already-normalized records.
    """
    preds = [p if "pred_capability_id" in p else from_execution_trace(p)
             for p in predictions]
    by_id = {p["id"]: p for p in preds}
    cases = [r for r in dataset]

    m1n = m1d = 0
    m2n = m2d = 0
    m3n = m3d = 0
    m4n = m4d = 0
    m5n = 0
    m7n = m7d = 0
    forbidden_emitted = 0
    stage_hist: dict = {}
    deepest_hist: dict = {}
    bucket_stats: dict = {}
    pe_extra_slots = 0
    missing_preds = 0

    for row in cases:
        cid = row["id"]
        pred = by_id.get(cid)
        gold_cap = row.get("capability_id")
        gold_args = dict(row.get("gold_arguments") or {})
        pe_slots = list(row.get("pe_model_slots") or OWN.pe_model_slots(gold_cap))
        expected = dict(row.get("expected_tool_arguments") or {})
        ownership = dict(row.get("arg_ownership") or {})
        forbidden = set(row.get("forbidden_arguments") or [])

        if pred is None:
            missing_preds += 1
            pred_args: dict = {}
            pred_cap = None
            binder_state = None
            called = False
            fallback = "NO_PREDICTION"
            deepest = None
        else:
            pred_args = dict(pred.get("pred_arguments") or {})
            pred_cap = pred.get("pred_capability_id")
            binder_state = pred.get("binder_state")
            called = bool(pred.get("virtual_executor_called"))
            fallback = pred.get("fallback_reason")
            deepest = pred.get("deepest_stage")

        # M1 — tool selection
        m1d += 1
        if pred_cap == gold_cap:
            m1n += 1

        # M2 — PE strict (exact key set + values over the PE slots)
        if pe_slots:
            m2d += 1
            if _restrict(pred_args, pe_slots) == _restrict(gold_args, pe_slots):
                m2n += 1

        # M3 — PE lenient (per gold PE slot value match)
        for slot in gold_args:
            if slot in pe_slots:
                m3d += 1
                if _eq(pred_args.get(slot), gold_args[slot]):
                    m3n += 1
        # an extra PE slot the model emitted but gold did not ask for is a miss
        pe_extra_slots += sum(1 for s in pred_args if s in pe_slots and s not in gold_args)

        # M4 — Binder pass (only over cases that reached the Binder)
        if binder_state is not None:
            m4d += 1
            if binder_state == "COMPLETE":
                m4n += 1

        # M5 — virtual execution reached
        if called:
            m5n += 1

        # M6 — stage failures
        key = fallback or "-"
        stage_hist[key] = stage_hist.get(key, 0) + 1
        dkey = deepest or "-"
        deepest_hist[dkey] = deepest_hist.get(dkey, 0) + 1

        # M7 — complete Tool contract match
        if expected:
            m7d += 1
            if _restrict(pred_args, expected) == expected:
                m7n += 1
        # forbidden emissions (the model must never emit these)
        for s in forbidden:
            if s in pred_args:
                forbidden_emitted += 1

        # M8 — non-PE slot sources
        for slot, want in expected.items():
            if ownership.get(slot, OWN.PE_MODEL) == OWN.PE_MODEL:
                continue
            st = bucket_stats.setdefault(
                ownership.get(slot, OWN.UNDECLARED),
                {"expected": 0, "supplied": 0, "matched": 0})
            st["expected"] += 1
            if slot in pred_args:
                st["supplied"] += 1
                if _eq(pred_args[slot], want):
                    st["matched"] += 1

    return {
        "scorer_version": SCORER_VERSION,
        "dataset_version": DATASET_VERSION,
        "cases": len(cases),
        "predictions_seen": len(preds),
        "missing_predictions": missing_preds,
        "metrics": {
            "M1_tool_selection": {
                "rule": "pred capability_id == gold capability_id",
                **_ratio(m1n, m1d)},
            "M2_pe_model_strict": {
                "rule": "pred(arguments restricted to pe_model_slots) == gold_arguments (exact key set + values)",
                **_ratio(m2n, m2d)},
            "M3_pe_model_lenient": {
                "rule": "per gold PE slot: value matches; denominator = gold PE slot instances",
                **_ratio(m3n, m3d),
                "extra_pe_slots_emitted": pe_extra_slots},
            "M4_binder_pass": {
                "rule": "binder_state == COMPLETE, over cases that reached binding",
                **_ratio(m4n, m4d)},
            "M5_virtual_exec_reached": {
                "rule": "virtual_executor_called is true, over all cases",
                **_ratio(m5n, m1d)},
            "M6_stage_failures": {
                "rule": "counts (no denominator): fallback_reason / deepest_stage histogram",
                "fallback_reason": dict(sorted(stage_hist.items())),
                "deepest_stage": dict(sorted(deepest_hist.items())),
                "forbidden_emissions": forbidden_emitted},
            "M7_tool_contract_match": {
                "rule": "pred(arguments restricted to expected_tool_arguments) == expected_tool_arguments (exact)",
                **_ratio(m7n, m7d)},
            "M8_non_pe_slot_sources": {
                "rule": "per ownership bucket: expected / supplied / value-matched slot counts (no ratio)",
                "buckets": {k: dict(v) for k, v in sorted(bucket_stats.items())}},
        },
    }


def load_dataset(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def load_predictions(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Formal-500 v2 scorer")
    ap.add_argument("--dataset", required=True, help="formal500_dataset_v2.jsonl")
    ap.add_argument("--predictions", required=True,
                    help="execution_trace.jsonl (or normalized prediction records)")
    ap.add_argument("--out", default=None, help="write the JSON report here")
    args = ap.parse_args(argv)

    report = score(load_dataset(args.dataset), load_predictions(args.predictions))
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
