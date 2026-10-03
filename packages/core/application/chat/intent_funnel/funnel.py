"""Intent Funnel — public façade.

The chain correction pins the ACTIVE target chain to one hop and
the live-table ruling retired the legacy QIR lane entirely:

    Matcher HIT  ─┐
                  ├→ ToolIntentModel (ONE call: select + extract) → Binder (validate) → Execute
    MISS/AMB → Recall ─┘

every non-COMPLETE outcome exits to the Agent (8.10). The cascade is the
product's single formal routing lane (ruling): the old dark-launch
gate was deleted; the chain runs on every turn unless a guardrail vetoes it
(:func:`.policy.funnel_live`).

This module holds the public entry points ONLY. Control flow lives in
:mod:`.orchestrator`; rollout gating and reason naming in :mod:`.policy`;
candidate shaping in :mod:`.candidate_aggregation`; trace/event plumbing in
:mod:`.observability`; the dry-run console lane in :mod:`.preview`; the
full-cascade shadow used by preview/offline tooling in :mod:`.shadow`.
The re-exports below are the compatibility surface for historic import and
monkeypatch sites.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # import cycle: understanding -> actions -> intent_funnel (see __init__)
    from core.application.chat.understanding import TurnRequirements

from .observability import new_trace as _new_trace, persist_event as _persist_event
from .orchestrator import cascade as _cascade, run_cascade as _run_cascade
from .policy import funnel_live, kind_enabled
from .preview import preview
from .shadow import cascade_shadow


async def route(ctx, *, deps, requirements: TurnRequirements) -> TurnRequirements:
    """The one call the orchestrator makes (design: docs/temp.md P0 shape).

    Returns the SAME requirements object untouched whenever the funnel abstains
    — the Agent keeps the turn, byte-identical, zero pollution. Otherwise hands
    off to the single-hop cascade (:func:`orchestrator.cascade`).
    """
    # Migration compat boundary (P1 ruling): a turn L0 already certified is
    # never touched by the new lane while L0 stays in charge. This is a scoping
    # fact of the coexistence period, NOT a statement that L0 is the baseline.
    # ( single-path ruling: kept — BUG-3/L0 retirement is a separate
    # follow-up task.)
    if requirements.requested_action is not None:
        return requirements
    if funnel_live(requirements, deps, ctx):
        return await _cascade(ctx, deps, requirements)
    return requirements
