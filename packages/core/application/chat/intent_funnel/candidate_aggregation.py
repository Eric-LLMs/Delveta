"""Capability Candidate Aggregation — the data-shaping stage between Recall and ToolIntentModel.

Final semantics (2026-09-26): Recall keeps EVERY hit >= threshold at the raw
stage (a capability may legitimately arrive several times through different
sentences); this stage collapses the query-level candidates into ONE
capability-level card per capability before the model sees them. Pure list
transform — no embedding, no SQL, no retrieval, no adjudication (which
capability wins is the ToolIntentModel's call, not here).
"""
from __future__ import annotations

from .contract import Candidate


def aggregate_by_capability(candidates: list[Candidate]) -> list[Candidate]:
    """Collapse query-level candidates to ONE per capability_id: the highest-
    scoring hit rides with ITS own provenance; ties keep the earlier arrival
    (matcher seed first). Raw Recall keeps every >= threshold hit — this stage
    sits strictly between the raw assembly and ToolIntentModel, so the model
    only ever sees capability-level cards."""
    best: dict[str, Candidate] = {}
    for c in candidates:
        cur = best.get(c.capability_id)
        if cur is None or c.score > cur.score:
            best[c.capability_id] = c
    return list(best.values())


def apply_model_floor(cands: list[Candidate], floor: float) -> list[Candidate]:
    """Phase-E shadow seam: the raw lane moved the quality gate out of
    recall so ALL threshold buckets are recomputable offline; the model-facing
    set re-applies the floor to the aggregated (capability-level) cards and
    keeps every capability whose WINNING hit is at or above it —
    set-equivalent to screening the raw hits, since the representative carries
    the max score (candidate-count cap retired by the 2026-09-26 ruling) —
    plus every matcher-origin card (HIT 1.0 / AMBIGUOUS 0.0: table
    evidence, not calibrated cosine scores, exempt from the floor)."""
    recall_c = [c for c in cands if c.origin == "recall" and c.score >= floor]
    return sorted([c for c in cands if c.origin != "recall"] + recall_c,
                  key=lambda c: c.score, reverse=True)
