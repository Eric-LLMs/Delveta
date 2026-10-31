"""Reuse an identical successful search inside the same agent turn.

The model occasionally re-issues a search with the *same* query, tool, result count
and scope after it already has the results — pure waste, since the corpus did not
change between two steps of one turn. This module is the single place that decision
lives:

* the cache is bound to the current :class:`~agent.engine.context.AgentTurn`, so it is
  request-scoped and single-turn by construction — nothing survives the turn, nothing
  is written to the database, and no cross-turn state exists;
* the key is the tuple ``(tool, normalized query, effective top_k, scope)``: a
  re-phrasing, a different ``top_k`` or a different scope is a different search and
  executes normally;
* only a successful list result is stored. A tool failure raises before reaching
  :func:`record`, and a degraded notice (``rag_search``'s ``_UNAVAILABLE``) is never
  passed to it, so a failure can never be replayed as a result;
* no in-flight coalescing: two identical calls dispatched in the same parallel group
  both execute (there is no lock and no pending-future map — deliberately out of
  scope), and only later steps reuse.
"""
from __future__ import annotations

from typing import Any

from agent.engine.context import current_turn

# A search key: tool + normalized query + effective top_k + effective scope.
SearchKey = tuple[str, str, str, str]


def normalize_query(query: Any) -> str:
    """Whitespace-collapsed, case-folded query text (the only normalization applied)."""
    return " ".join(str(query or "").split()).casefold()


def key(tool: str, query: Any, top_k: Any, scope: Any = None) -> SearchKey:
    """Build the reuse key for one search call."""
    return (tool, normalize_query(query), str(top_k), "" if scope is None else str(scope))


def lookup(cache_key: SearchKey) -> list | None:
    """Return the earlier result for this turn, or ``None`` when nothing is cached.

    ``None`` (not falsiness) is the miss signal, so a legitimately empty result list is
    still reused instead of being searched again.
    """
    turn = current_turn()
    if turn is None:
        return None
    return turn.search_results.get(cache_key)


def record(cache_key: SearchKey, value: Any) -> None:
    """Cache one successful result list for the rest of the turn (no-op elsewhere)."""
    turn = current_turn()
    if turn is None or not isinstance(value, list):
        return
    turn.search_results[cache_key] = value
