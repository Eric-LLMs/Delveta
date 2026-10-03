"""Shared ToolIntentModel contracts — the backend-independent seam.

ToolIntentModel contract (chain ruling): ONE call per turn does BOTH
decisions — which capability the sentence demands, and the argument draft for
it. Input discipline (8.17 #2/#3, Action-Contract ruling): query +
TurnFacts + candidate Cards (assembled by :mod:`.prompt` from the Registry
row); the reply is only {capability_id, confidence, arguments}. Candidate and
verdict types live in :mod:`..contract` — the funnel-wide vocabulary.

Every failure mode of the answer side is an exit to the Agent: "NONE" ->
REJECT, an off-card id or a below-floor confidence -> UNCERTAIN, a malformed
reply -> backend-unavailable (:class:`ToolIntentUnavailable`, the signal every
backend raises when IT cannot serve — the ladder falls through, the funnel
never fabricates a verdict from a backend fault).

Prompt construction is in :mod:`.prompt`, reply parsing in :mod:`.parser`,
the dispatch ladder in :mod:`.model`.
"""
from __future__ import annotations


class ToolIntentUnavailable(Exception):
    """This backend cannot serve (not deployed / transport down) — fall through."""
