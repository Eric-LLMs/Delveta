"""Capability Candidate Aggregation — the data-shaping stage between Recall and cap_router.

Final semantics: Recall keeps EVERY hit >= threshold at the raw
stage (a capability may legitimately arrive several times through different
sentences); this stage collapses the query-level candidates into ONE
capability-level card per capability before the selector sees them. Pure list
transform — no embedding, no SQL, no retrieval, no adjudication (which
capability wins is the cap_router's call, not here).
"""
from __future__ import annotations

from dataclasses import dataclass

from .contract import Candidate

# The top-k cap the business layer applies to the candidate list before it
# reaches the SELECTION node: the selector is defined over a VARIABLE-length
# candidate set, and when more than this many capabilities survive the gate the
# business layer hands it only the top-k (highest score). This is NOT a
# model-side limit and NOT a routing threshold — it is a list-length cap only.
# See normalize_top_k below + §26.2.
K_DEFAULT = 3


@dataclass(frozen=True)
class NormalizedCandidates:
    """The outcome of the business-layer K normalization (§26.2).

    ``candidates`` is the list the selector may see: at most ``k``, highest score
    first (the caller sorts by score before normalizing). There is no "direct"
    outcome: EVERY K >= 1 candidate set — K = 1 and K = 2 included — is handed to
    the selector, so the business layer never selects a capability on the model's
    behalf and no confidence is fabricated. K = 0 yields an empty list; the
    caller has already short-circuited an empty candidate set before reaching
    normalization.
    """

    candidates: tuple[Candidate, ...] = ()


def normalize_top_k(candidates: list[Candidate], k: int = K_DEFAULT) -> NormalizedCandidates:
    """Business-layer K normalization for the SELECTION node (§26.2).

    Pure, no I/O, backend-agnostic — the SAME rule governs the stub lane and the
    cap_router lane, and ``backend=off`` never enters here. The candidate count K
    maps:

    * ``K = 0`` — nothing to select (caller short-circuits on the empty set);
    * ``K = 1..k`` — the WHOLE list goes to the selector (K = 1 and K = 2 are
      ordinary candidate sets, not business-layer special cases);
    * ``K >= k+1`` — only the top-``k`` (highest score) go to the selector.

    The selector is therefore consulted for EVERY non-empty candidate set, over a
    variable-length option list of ``len(candidates)`` cards plus the frozen
    REJECT card appended by the card renderer.
    """
    ordered = list(candidates)
    if not ordered:
        return NormalizedCandidates()
    return NormalizedCandidates(candidates=tuple(ordered[:k]))


def aggregate_by_capability(candidates: list[Candidate]) -> list[Candidate]:
    """Collapse query-level candidates to ONE per capability_id: the highest-
    scoring hit rides with ITS own provenance; ties keep the earlier arrival
    (matcher seed first). Raw Recall keeps every >= threshold hit — this stage
    sits strictly between the raw assembly and cap_router, so the selector
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
