"""PE scoring v2 — the PER-SAMPLE, contract-aligned PE scoring口径.

This is a NEW, versioned scoring口径. It does NOT replace the legacy
``eval_pilot.score`` (kept frozen for historical reproduction); it is the CURRENT
standard, used to re-score existing replies against the current contract.

What changes vs the legacy口径
------------------------------
* **PE Argument EM is per-sample.** The legacy scorer scored every ``cap-social-
  search`` turn over the ownership map's FOUR-slot model set
  ``{query, platform, subreddit, limit}``. v2 scores each turn over its EFFECTIVE
  model slots: the contract's static slots — ``pe_contract.model_slots(cap)`` for
  the six in-contract capabilities, the frozen ownership map for every other
  capability — NARROWED by the ONE deterministic scope rule
  (``cap_handler.search_scope``). A turn whose sentence carries an explicit
  ``r/<sub>`` scope therefore scores over ``{query}`` only (``platform`` is pinned
  deterministically, ``subreddit`` / ``limit`` are execution parameters), so a
  model is never credited or penalised for a slot it was not asked for.

* **Execution parameters are scored SEPARATELY.** The deterministic scope args
  (``subreddit`` extracted from the raw sentence + the ``platform`` pin) are
  validated against the gold under their OWN metric — never folded into PE
  Argument EM. A correct scope is not a PE extraction, and a wrong one must not
  hide inside a PE score.

This module imports the SAME rule the runtime handler, the dataset builder and the
SFT scorer use (``cap_handler.search_scope``): there is exactly ONE place the
``r/<sub>`` shape and the slot narrowing are defined.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent                 # logs/_argbench
ROOT = HERE.parents[1]
SCRIPTS = ROOT / "scripts" / "qwen_parameter_extraction_finetune"
for _p in (str(ROOT), str(ROOT / "packages"), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pe_contract as C  # noqa: E402
from core.application.chat.intent_funnel.cap_handler.search_scope import (  # noqa: E402
    effective_model_slots,
    execution_scope_args,
)

SCORING_VERSION = "pe_score_v2"


# ── value comparison (self-contained; no production import) ──────────────────────

def _canon(value):
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, (int, float)):
        return ("num", float(value))
    if isinstance(value, str):
        return ("str", value.strip())
    return (type(value).__name__, str(value))


def _eq(gold_v, pred_v) -> bool:
    if _canon(gold_v) == _canon(pred_v):
        return True
    # numeric-string leniency (3 == "3" == 3.0), never for bools.
    if not isinstance(gold_v, bool) and not isinstance(pred_v, bool):
        try:
            return float(gold_v) == float(pred_v)
        except (TypeError, ValueError):
            return False
    return False


# ── the per-sample slot set ──────────────────────────────────────────────────────

def static_model_slots(cap: str, own_entry: dict | None) -> tuple[str, ...]:
    """The static model slots for ``cap`` under the CURRENT contract: the six
    in-contract caps come from ``pe_contract`` (the authoritative face — the
    ownership map's wider set is NOT used for them); every other cap falls back to
    the frozen ownership map's model schema property order."""
    if cap in C.SCHEMA:
        return tuple(C.model_slots(cap))
    props = ((own_entry or {}).get("model_schema") or {}).get("properties") or {}
    return tuple(props)


def effective_slots(cap: str, raw_query: str, own_entry: dict | None) -> tuple[str, ...]:
    """The per-sample model slots (static minus any deterministically pinned slot)."""
    return effective_model_slots(cap, raw_query, static_model_slots(cap, own_entry))


# ── the two separate scores ──────────────────────────────────────────────────────

def score_pe(rec: dict, own_entry: dict | None) -> dict:
    """PE score over the EFFECTIVE model slots only."""
    cap = rec["capability_id"]
    raw = str(rec.get("query") or rec.get("user_query") or "")
    slots = effective_slots(cap, raw, own_entry)
    slotset = set(slots)
    gold = rec.get("gold_arguments") or {}
    pred = rec.get("pred_arguments")
    pred = pred if isinstance(pred, dict) else {}

    per_arg = {}
    for slot in slots:
        present = slot in gold
        p = pred.get(slot, None)
        if present:
            exact = p is not None and _eq(gold[slot], p)
            per_arg[slot] = {"state": "present", "gold": gold[slot], "pred": p,
                             "exact": exact}
        else:
            per_arg[slot] = {"state": "absent", "gold": None, "pred": p,
                             "exact": p is None}

    considered = [s for s in slotset if s in gold or s in pred]
    field_hits = sum(1 for s in considered if per_arg[s]["exact"])
    required = [s for s in C.required_model_slots(cap) if s in slotset] \
        if cap in C.SCHEMA else \
        [s for s in (((own_entry or {}).get("model_schema") or {}).get("required") or [])
         if s in slotset]
    # illegal = a model-emitted key outside the effective slots (non-model emission)
    illegal = sorted(k for k in pred if k not in slotset)
    return {
        "capability_id": cap,
        "effective_slots": list(slots),
        "pe_em": (set(pred) & slotset) == (set(gold) & slotset)
                 and all(per_arg[s]["exact"] for s in slotset),
        "per_arg": per_arg,
        "fields_considered": len(considered),
        "field_hits": field_hits,
        "required": required,
        "required_present": sum(1 for s in required if s in pred),
        "required_correct": sum(1 for s in required if per_arg.get(s, {}).get("exact")),
        "non_model_emitted": illegal,
    }


def score_exec(rec: dict) -> dict:
    """The deterministic EXECUTION parameters, validated against the gold.

    This checks the deterministic RULE (the ``r/<sub>`` extraction + the platform
    pin) against the gold tool call — NOT the model. It is reported separately from
    PE Argument EM so a scope error can never hide inside an extraction score."""
    cap = rec["capability_id"]
    raw = str(rec.get("query") or rec.get("user_query") or "")
    expected = execution_scope_args(cap, raw)
    gold = rec.get("gold_arguments") or {}
    checks = {}
    for key, value in expected.items():
        checks[key] = {"expected": value, "gold": gold.get(key),
                       "ok": key in gold and _eq(value, gold[key])}
    has_scope = bool(expected)
    return {
        "has_scope": has_scope,
        "checks": checks,
        "ok": (all(c["ok"] for c in checks.values()) if has_scope else None),
    }


def score_record(rec: dict, own_entry: dict | None = None) -> dict:
    """Both scores for one reply record (plus a flattened emission flag)."""
    pe = score_pe(rec, own_entry)
    ex = score_exec(rec)
    return {
        "id": rec.get("id"),
        "capability_id": rec.get("capability_id"),
        "parse_error": bool(rec.get("parse_error")),
        "pe": pe,
        "exec": ex,
    }


# ── aggregation ──────────────────────────────────────────────────────────────────

def _pct(num: int, den: int):
    return round(num / den, 4) if den else None


def _bucket(records: list[dict]) -> dict:
    n = len(records)
    if not n:
        return {"samples": 0}
    consul = sum(r["pe"]["fields_considered"] for r in records)
    hits = sum(r["pe"]["field_hits"] for r in records)
    req_total = sum(len(r["pe"]["required"]) for r in records)
    req_present = sum(r["pe"]["required_present"] for r in records)
    req_correct = sum(r["pe"]["required_correct"] for r in records)
    scoped = [r for r in records if r["exec"]["has_scope"]]
    return {
        "samples": n,
        "pe_em": _pct(sum(1 for r in records if r["pe"]["pe_em"]), n),
        "field_accuracy": _pct(hits, consul),
        "required_missing_rate": _pct(req_total - req_present, req_total),
        "required_accuracy": _pct(req_correct, req_total),
        "non_model_emission_rate": _pct(
            sum(1 for r in records if r["pe"]["non_model_emitted"]), n),
        "parse_error_rate": _pct(sum(1 for r in records if r["parse_error"]), n),
        "exec_scope_samples": len(scoped),
        "exec_scope_correct": _pct(sum(1 for r in scoped if r["exec"]["ok"]), len(scoped)),
    }


def aggregate(records: list[dict]) -> dict:
    by_cap = {}
    for cap in sorted({r["capability_id"] for r in records}):
        by_cap[cap] = _bucket([r for r in records if r["capability_id"] == cap])
    return {"scoring_version": SCORING_VERSION,
            "overall": _bucket(records), "by_capability": by_cap}


# ── the frozen offline-reference schema snapshot (+ SHA-256) ─────────────────────

def schema_snapshot() -> dict:
    """The offline REFERENCE contract face per in-contract capability.

    Explicitly NOT the LIVE Registry: the Registry is unreachable, so this is the
    stdlib-only shared contract (``pe_contract``) the trainer / scorers read —
    including the ``query`` description (``_SEARCH_QUERY_DESC``), so the result/
    count-rule wording is captured and hashable."""
    caps = {}
    for cap in sorted(C.SCHEMA):
        caps[cap] = {
            "model_slots": C.model_slots(cap),
            "function_parameters": C.model_schema(cap),
            "tool": C.tool_name(cap),
        }
    return {
        "kind": "offline_reference_contract_snapshot",
        "live_registry": False,
        "source": "scripts/qwen_parameter_extraction_finetune/pe_contract.py",
        "source_commit": C.SOURCE_COMMIT,
        "note": ("NOT the LIVE Registry — the Registry was unreachable; this is the "
                 "offline reference contract (model slots + rendered face) the "
                 "trainer and scorers read."),
        "capabilities": caps,
    }


def snapshot_sha256() -> str:
    payload = json.dumps(schema_snapshot(), ensure_ascii=False,
                         sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
