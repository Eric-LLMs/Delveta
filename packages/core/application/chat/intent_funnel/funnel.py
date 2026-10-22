"""Intent Funnel — public façade.

Two lanes, one chain (single-path ruling); the live-table ruling retired the
legacy QIR lane entirely:

    Matcher HIT        → Argument Path Router → Binder (validate) → Execute
    MISS/AMBIGUOUS     → Recall → Aggregation → cap_router → ARP → Binder → Execute

A HIT is never re-selected. Every non-COMPLETE outcome exits to the Agent (8.10).
The cascade is the
product's single formal routing lane (ruling): the old dark-launch
gate was deleted; the chain runs on every turn with wired deps
(:func:`.policy.funnel_live`). No entry veto re-routes a turn before the
Matcher sees it.

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

if TYPE_CHECKING:  # typing only: the requirements contract lives in understanding
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
    # Zero-input safety guard (NOT an intent rule): a blank/whitespace turn has
    # nothing to match or recall, so it never spends a node or a model hop.
    message = getattr(getattr(ctx, "body", None), "message", "") or ""
    if not message.strip():
        return requirements
    if funnel_live(deps):
        return await _cascade(ctx, deps, requirements)
    return requirements
