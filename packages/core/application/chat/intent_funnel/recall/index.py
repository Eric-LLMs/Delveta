"""Recall index lifecycle: load the LIVE corpus rows, fingerprint-cache the index."""
from __future__ import annotations

import hashlib
import logging
import types

from sqlalchemy import text as sql_text

logger = logging.getLogger(__name__)

_MARKER_SQL = sql_text(
    "SELECT"
    " (SELECT count(*) || ':' || coalesce(max(updated_at)::text, '-') FROM capabilities),"
    " (SELECT count(*) || ':' || coalesce(max(updated_at)::text, '-') FROM capability_standard_queries),"
    " (SELECT count(*) || ':' || coalesce(max(updated_at)::text, '-') FROM capability_similar_queries)"
)

_LOAD_ROWS_SQL = sql_text(
    "SELECT 'standard' AS kind, s.id, s.capability_id, s.query, s.language,"
    "       NULL AS standard_query_id, s.embedding"
    "  FROM capability_standard_queries s"
    "  JOIN capabilities c ON c.capability_id = s.capability_id"
    " WHERE s.enabled AND c.enabled AND c.status = 'active'"
    "   AND c.intent_kind <> 'research'"
    "   AND s.embedding IS NOT NULL"
    " UNION ALL "
    "SELECT 'similar', q.id, s.capability_id, q.query, q.language,"
    "       q.standard_query_id, q.embedding"
    "  FROM capability_similar_queries q"
    "  JOIN capability_standard_queries s ON s.id = q.standard_query_id"
    "  JOIN capabilities c ON c.capability_id = s.capability_id"
    " WHERE q.enabled AND s.enabled AND c.enabled AND c.status = 'active'"
    "   AND c.intent_kind <> 'research'"
    "   AND q.embedding IS NOT NULL"
)

# marker -> index (fingerprint + vector rows); swapped by construction when any
# live corpus row changes (the store bumps updated_at on every write).
_INDEX_CACHE: dict[str, types.SimpleNamespace] = {}


async def load_index(session_factory):
    """Load the LIVE Recall index (fingerprint + embedded rows for the
    degraded in-process lane), or None when NO embedded row is routable —
    the funnel reports that as RECALL_UNAVAILABLE until the backfill runs.
    Test doubles that patch this seam hand plain duck-typed indexes and stay
    on the in-process lane (8.17 node isolation)."""
    async with session_factory() as session:
        marker_row = (await session.execute(_MARKER_SQL)).first()
    marker = "|".join(str(x) for x in (marker_row or ()))
    cached = _INDEX_CACHE.get(marker)
    if cached is not None:
        return cached
    async with session_factory() as session:
        rows = (await session.execute(_LOAD_ROWS_SQL)).all()
    corpus = tuple(
        types.SimpleNamespace(
            kind=str(r[0]), query_id=str(r[1]), capability_id=str(r[2]),
            query=str(r[3]), language=str(r[4]),
            standard_query_id=str(r[5]) if r[5] is not None else None,
            vector=list(r[6]),
        )
        for r in rows
    )
    if not corpus:
        _INDEX_CACHE.clear()
        return None
    digest = hashlib.sha256(
        "|".join(f"{c.kind}:{c.capability_id}:{c.language}:{c.query}"
                 for c in sorted(corpus, key=lambda c: c.query_id))
        .encode("utf-8")
    ).hexdigest()[:12]
    index = types.SimpleNamespace(
        version="corpus1-" + digest,
        corpus=corpus,
        session_factory=session_factory,
    )
    if len(_INDEX_CACHE) >= 8:
        _INDEX_CACHE.clear()
    _INDEX_CACHE[marker] = index
    return index
