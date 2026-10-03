"""Recall scoring: cosine, row->Candidate conversion, and the quality gate.

Cosine here is a QUALITY GATE (filter obvious garbage), the "which one"
decision belongs to the ToolIntentModel (chain ruling); this
module never adjudicates a capability winner.
"""
from __future__ import annotations

import math

from ..contract import Candidate, RecallResult


def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def hits_from_rows(rows, kind: str) -> list[Candidate]:
    out: list[Candidate] = []
    for r in rows:
        out.append(Candidate(
            capability_id=str(r[0]), score=round(float(r[-1]), 6),
            matched_example=str(r[2]), origin="recall",
            query_kind=kind, language=str(r[3]), query_id=str(r[1]),
            standard_query_id=(str(r[4]) if kind == "similar" and r[4] is not None
                               else None),
        ))
    return out


def gate(hits: list[Candidate], *, min_score: float) -> RecallResult:
    """The quality gate + score order. EVERY hit >= min_score is a candidate
    (a capability may appear several times, through different sentences)."""
    kept = sorted((c for c in hits if c.score >= min_score),
                  key=lambda c: c.score, reverse=True)
    return RecallResult(candidates=tuple(kept))
