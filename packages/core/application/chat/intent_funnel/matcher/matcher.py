"""Node 1 — Matcher: deterministic EXACT hits from the LIVE tables ONLY.

Live-table ruling: the exact set is the enabled Standard +
Similar query sentences (``entry.intent_corpus`` over
capability_standard_queries / capability_similar_queries), normalized for
comparison; a HIT means the turn's sentence IS one of the human-curated
canonical phrasings, so ``matched_literal`` is always a readable sentence,
never a regex literal. Regexes, patterns, aliases and request examples are
NOT match data: they cannot produce a HIT at all.

A HIT proves the sentence was CURATED as this action — it does not prove a new,
similar sentence is an action; everything non-exact escalates to Recall +
ToolIntentModel (chain ruling). Global guards (negation, research/handoff veto)
live in the funnel common layer, not here — 8.1-a ruling.

Escalation discipline (8.1): no hit -> MISS (falls to Recall); one hit -> HIT;
several capabilities hit -> MATCH_AMBIGUOUS carrying ALL candidate ids upward
for ToolIntentModel. The Matcher itself never picks.

Package map: :mod:`.normalize` (canonical text), :mod:`.index` (exact-lookup
index + fingerprint cache), this module (the one deterministic pass).
"""
from __future__ import annotations

from ..contract import (
    MATCH_AMBIGUOUS,
    MATCH_HIT,
    MATCH_MISS,
    MatchResult,
    TurnFacts,
)
from .index import build_index
from .normalize import normalize


def match(query: str, facts: TurnFacts, view) -> MatchResult:
    """One deterministic exact-phrase pass over the active version's corpus.
    Cost: a single dict lookup — this node is cheap enough to run on every
    shadowed turn.

    ``facts`` is the formal turn-side contract (ruling): the Matcher
    receives the current turn's settled structured facts (viewer / attachment),
    never raw history. Table entries may gate on these facts from P3 on.
    ``view`` is the Registry side (§8.1: the ONLY match data this node reads)."""
    if not (query or "").strip():
        return MatchResult(state=MATCH_MISS, registry_version=view.fingerprint)
    exact = build_index(view)
    q = normalize(query)
    hits = exact.get(q, set())
    if not hits:
        return MatchResult(state=MATCH_MISS, registry_version=view.fingerprint)
    if len(hits) == 1:
        return MatchResult(
            state=MATCH_HIT, capability_id=next(iter(hits)),
            registry_version=view.fingerprint, matched_literal=q[:160],
        )
    return MatchResult(
        state=MATCH_AMBIGUOUS, candidates=tuple(sorted(hits)),
        registry_version=view.fingerprint,
    )
