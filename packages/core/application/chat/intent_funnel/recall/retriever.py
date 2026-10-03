"""Recall retrieval strategies: the pgvector ANN lane and the degraded in-process lane.

Two INDEPENDENT vector searches per turn (Standard path + Similar path) over
ONE user-query embedding; the hit sets are simply concatenated. The retired
per-capability MAX/AVG merge is GONE by construction: every hit at or above
``min_score`` is kept as its own candidate with full provenance (live-table
ruling).
"""
from __future__ import annotations

from sqlalchemy import text as sql_text

from ..contract import Candidate
from .scoring import cosine, gate, hits_from_rows

# ANN reads a wide neighborhood; it is an ANN WIDTH, not a candidate cap —
# everything >= min_score inside the pool is kept.
_ANN_POOL = 64

_STANDARD_SQL = sql_text(
    "SELECT s.capability_id, s.id, s.query, s.language,"
    "       1 - (s.embedding <=> CAST(:q AS vector)) AS score"
    "  FROM capability_standard_queries s"
    "  JOIN capabilities c ON c.capability_id = s.capability_id"
    " WHERE s.enabled AND c.enabled AND c.status = 'active'"
    "   AND c.intent_kind <> 'research'"
    "   AND s.embedding IS NOT NULL"
    " ORDER BY s.embedding <=> CAST(:q AS vector)"
    " LIMIT :pool"
)
_SIMILAR_SQL = sql_text(
    "SELECT s.capability_id, q.id, q.query, q.language, q.standard_query_id,"
    "       1 - (q.embedding <=> CAST(:q AS vector)) AS score"
    "  FROM capability_similar_queries q"
    "  JOIN capability_standard_queries s ON s.id = q.standard_query_id"
    "  JOIN capabilities c ON c.capability_id = s.capability_id"
    " WHERE q.enabled AND s.enabled AND c.enabled AND c.status = 'active'"
    "   AND c.intent_kind <> 'research'"
    "   AND q.embedding IS NOT NULL"
    " ORDER BY q.embedding <=> CAST(:q AS vector)"
    " LIMIT :pool"
)


def _q_literal(qvec) -> str:
    return "[" + ",".join(repr(float(x)) for x in qvec) + "]"


async def ann_recall(index, qvec, *, min_score: float) -> RecallResult:
    """pgvector lane: TWO independent distance-ordered searches (Standard
    path, Similar path). No per-capability merge — hits are concatenated as
    they came back, every one >= min_score kept."""
    q_literal = _q_literal(qvec)
    async with index.session_factory() as session:
        std_rows = (await session.execute(
            _STANDARD_SQL, {"q": q_literal, "pool": _ANN_POOL})).all()
        sim_rows = (await session.execute(
            _SIMILAR_SQL, {"q": q_literal, "pool": _ANN_POOL})).all()
    hits = hits_from_rows(std_rows, "standard") + hits_from_rows(sim_rows, "similar")
    return gate(hits, min_score=min_score)


def in_process_recall(index, qvec, *, min_score: float) -> RecallResult:
    """In-process lane: exact cosine over the SAME rows load_index read
    (test doubles and ANN faults land here; two paths, no merge). A dim
    mismatch between the query vector and any corpus row is a PROFILE/DIM
    CONFIG FAULT, not a business result: silently truncating (zip) would
    turn it into garbage scores or a fake-empty set that exits as
    NO_CANDIDATE. Raise instead — the funnel maps it to RECALL_UNAVAILABLE."""
    hits: list[Candidate] = []
    for c in getattr(index, "corpus", ()):
        if len(c.vector) != len(qvec):
            raise RuntimeError(
                f"recall: dim mismatch — query {len(qvec)} vs corpus row "
                f"{c.query_id} ({c.capability_id}) {len(c.vector)}; embedder "
                "profile and stored vectors must share one dimension")
        hits.append(Candidate(
            capability_id=c.capability_id, score=round(cosine(qvec, c.vector), 6),
            matched_example=c.query, origin="recall", query_kind=c.kind,
            language=c.language, query_id=c.query_id,
            standard_query_id=c.standard_query_id,
        ))
    return gate(hits, min_score=min_score)


__all__ = ["ann_recall", "in_process_recall"]
