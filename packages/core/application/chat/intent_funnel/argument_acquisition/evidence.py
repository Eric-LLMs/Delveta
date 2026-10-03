"""Evidence derivation — the deterministic, non-LLM acquisition signal (Phase 4).

The Argument Path Router needs ONE injected signal (:mod:`.evidence`, §E): for
each slot, WHICH source a value can actually be drawn from *right now*. It must
be deterministic and schema-aware — never produced by a model (a model deciding
its own acquisition need is a circular signal).

For the MODEL lane the only deterministic signal available to this step is the
CURRENT QUERY: a MODEL-owned slot is, by definition, extracted from the user's
own sentence (or the recent user turns an escalation would add), so a non-empty
query is the honest evidence that the QUERY source is available for it. A SYSTEM
owner gets NO entry here — its value, when it exists, is produced by
:mod:`.context_values` from the turn's settled facts, not by evidence.

The ``query`` argument is part of the frozen signature: an empty query carries
no query evidence at all, so nothing is emitted (the ARP then lands MISSING /
Agent, never a fabricated source).
"""
from __future__ import annotations

from .contract import OWNERSHIP_MODEL, SOURCE_QUERY


def evidence_for(entry, declaration, *, query: str) -> dict[str, str]:
    """The ``{slot: source}`` evidence signal for one capability. Emits
    ``SOURCE_QUERY`` for every MODEL-owned slot when the query is non-empty;
    emits nothing otherwise. Non-MODEL owners never receive an evidence entry —
    their acquisition is decided by facts, not by evidence."""
    if not str(query or "").strip():
        return {}
    return {
        slot: SOURCE_QUERY
        for slot, decl in (declaration or {}).items()
        if decl.ownership == OWNERSHIP_MODEL
    }
