"""Capability Candidate Aggregation — the data-shaping stage between Recall and ToolIntentModel.

Final semantics: Recall keeps EVERY hit >= threshold at the raw
stage (a capability may legitimately arrive several times through different
sentences); this stage collapses the query-level candidates into ONE
capability-level card per capability before the model sees them. Pure list
transform — no embedding, no SQL, no retrieval, no adjudication (which
capability wins is the ToolIntentModel's call, not here).
"""
from __future__ import annotations

from dataclasses import dataclass

from .contract import Candidate

# The candidate contract the SELECTION node is defined over (§26.2): the model
# was trained and benchmarked at K = 3 capability cards, so the business layer
# hands it at most this many — it is NOT a model-side limit. The selector is
# consulted ONLY at exactly K = 3 (3 cards + the frozen REJECT card = the 4
# option slots the model expects); K = 2 is V2-ineligible at the ORCHESTRATION
# layer (orchestrator.py gates on ``len(norm.candidates) < 3``) and degrades to
# the Agent. See normalize_top_k below + §25.6/§26.2.
K_DEFAULT = 3


@dataclass(frozen=True)
class NormalizedCandidates:
    """The outcome of the business-layer K normalization (§26.2).

    ``candidates`` is the list the selector may see: at most ``k``, highest score
    first (the caller sorts by score before normalizing). ``direct`` is set ONLY
    for K = 1 — the business layer executes that single capability itself and the
    selector (stub or cap_router) is never consulted. K = 0 yields an empty list
    with ``direct`` None; the caller has already short-circuited an empty
    candidate set before reaching normalization. This dataclass carries the
    PURE-function result only: the ORCHESTRATION layer independently refuses to
    consult the selector at K = 2 (V2-ineligible — the selector needs exactly 3
    cards + REJECT and a 2-card payload is never padded up).
    """

    candidates: tuple[Candidate, ...] = ()
    direct: Candidate | None = None

    @property
    def is_direct(self) -> bool:
        return self.direct is not None


def normalize_top_k(candidates: list[Candidate], k: int = K_DEFAULT) -> NormalizedCandidates:
    """Business-layer K normalization for the SELECTION node (§26.2).

    Pure, no I/O, backend-agnostic — the SAME rule governs the stub lane and the
    cap_router lane, and ``backend=off`` never enters here. The candidate count K
    maps:

    * ``K = 0`` — nothing to select (caller short-circuits on the empty set);
    * ``K = 1`` — the business layer executes it directly; the selector is never
      called, so no model is consulted and no confidence is fabricated;
    * ``K = 2`` — this function returns the 2 candidates, but the ORCHESTRATION
      layer does NOT consult the selector here: the deployed selector is defined
      over exactly 3 capability cards (+ the frozen REJECT card = 4 slots), so a
      2-card payload is V2-ineligible and the turn degrades to the Agent. K = 2
      is a degradation metric — never padded with a fake card;
    * ``K = 3..k`` — the whole list goes to the selector;
    * ``K >= k+1`` — only the top-``k`` (highest score) go to the selector.

    The model's decision head therefore only ever sees at most ``k`` options —
    and, on the deployed lane, exactly ``k`` = 3.
    """
    ordered = list(candidates)
    if not ordered:
        return NormalizedCandidates()
    if len(ordered) == 1:
        return NormalizedCandidates(direct=ordered[0])
    return NormalizedCandidates(candidates=tuple(ordered[:k]))


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
    the max score (candidate-count cap retired by the ruling) —
    plus every matcher-origin card (HIT 1.0 / AMBIGUOUS 0.0: table
    evidence, not calibrated cosine scores, exempt from the floor)."""
    recall_c = [c for c in cands if c.origin == "recall" and c.score >= floor]
    return sorted([c for c in cands if c.origin != "recall"] + recall_c,
                  key=lambda c: c.score, reverse=True)
