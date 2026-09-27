"""Intent Funnel — public façade.

The 2026-09-24 chain correction pins the ACTIVE target chain to one hop and
the 2026-09-26 live-table ruling retired the legacy QIR lane entirely:

    Matcher HIT  ─┐
                  ├→ ToolIntentModel (ONE call: select + extract) → Binder (validate) → Execute
    MISS/AMB → Recall ─┘

every non-COMPLETE outcome exits to the Agent (8.10). The chain is gated by
its own switch (``chat_funnel_enabled``, default OFF): with it closed,
route() returns the requirements object untouched.

This module holds the public entry points ONLY. Control flow lives in
:mod:`.orchestrator`; rollout gating and reason naming in :mod:`.policy`;
candidate shaping in :mod:`.candidate_aggregation`; trace/event plumbing in
:mod:`.observability`; the dry-run console lane in :mod:`.preview`; both
shadow lanes in :mod:`.shadow`. The re-exports below are the compatibility
surface for historic import and monkeypatch sites.
"""
from __future__ import annotations

from core.application.chat.understanding import TurnRequirements

from . import shadow
from .observability import new_trace as _new_trace, persist_event as _persist_event
from .orchestrator import cascade as _cascade, run_cascade as _run_cascade
from .policy import funnel_live, kind_enabled
from .preview import preview
from .shadow import cascade_shadow


async def route(ctx, *, deps, requirements: TurnRequirements) -> TurnRequirements:
    """The one call the orchestrator makes (design: docs/temp.md P0 shape).

    Returns the SAME requirements object untouched whenever the funnel is dark
    or abstains — the Agent keeps the turn, byte-identical, zero pollution.
    With the funnel gate open, hands off to the single-hop cascade
    (:func:`orchestrator.cascade`).
    """
    # P1 step-5 tri-state (8.15): unless the switch is off, run the Registry
    # Matcher in the dark and log its would_* verdict. Observation only.
    if deps is not None:
        mode = shadow.matcher_mode()
        if mode != "off":
            await shadow.observe(ctx, deps, requirements, mode)
    # Migration compat boundary (P1 ruling): a turn L0 already certified is
    # never touched by the new lane while L0 stays in charge. This is a scoping
    # fact of the coexistence period, NOT a statement that L0 is the baseline.
    if requirements.requested_action is not None:
        return requirements
    if funnel_live(requirements, deps, ctx):
        return await _cascade(ctx, deps, requirements)
    return requirements
