"""Node 2 — Recall: query -> scored hits, evidence only, never adjudicates.

Discipline (chain ruling + live-table ruling): cosine
here is a QUALITY GATE (filter obvious garbage), the "which one" decision
belongs to the ToolIntentModel. The index is the LIVE corpus itself —
``capability_standard_queries`` / ``capability_similar_queries`` are the
runtime truth, so there is no version predicate and no snapshot pair any more.

Predicates: row enabled AND capability enabled+active AND embedding IS NOT
NULL — so until the out-of-band backfill (scripts/embed_corpus.py) has run,
the corpus is empty for scoring purposes and this node honestly reports
RECALL_UNAVAILABLE (8.10), never a half-index.

Package map: :mod:`.index` (corpus load + cache), :mod:`.retriever` (ANN and
in-process lanes), :mod:`.scoring` (cosine + gate), this module (the public
entry that embeds once and picks the lane).
"""
from __future__ import annotations

import logging

from ..contract import RecallResult
from .retriever import ann_recall, in_process_recall

logger = logging.getLogger(__name__)


async def recall(index, query: str, *, embedder,
                 min_score: float) -> RecallResult:
    """All live-corpus hits at or above the quality gate, score-ordered.
    The user query is embedded EXACTLY ONCE for both paths. Raises on
    embedder failure — the funnel maps that to RECALL_UNAVAILABLE and falls
    open to the Agent."""
    if not (query or "").strip():
        return RecallResult(candidates=())
    vectors = await embedder.embed([query])
    if not isinstance(vectors, list) or not vectors or not vectors[0]:
        # empty embedder result is a fault, not an answer (8.10: no silent MISS)
        raise RuntimeError("recall: embedder returned no vector")
    qvec = vectors[0]

    if getattr(index, "session_factory", None) is not None:
        try:
            return await ann_recall(index, qvec, min_score=min_score)
        except Exception as exc:  # noqa: BLE001 - honest degrade, loud WARNING
            logger.warning("recall: ANN query failed (%r); falling back to "
                           "in-process cosine over the live corpus rows", exc)

    # In-process lane (test doubles and ANN faults land here).
    return in_process_recall(index, qvec, min_score=min_score)
