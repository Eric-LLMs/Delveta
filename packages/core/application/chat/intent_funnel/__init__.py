"""Intent Funnel — the independent, decoupled routing lane above the Agent.

Design (docs/temp.md, "Chat 意图路由重构"), chain ruling 2026-09-24: one model
hop per turn — Matcher HIT -> ToolIntentModel, Matcher MISS/AMBIGUOUS -> Recall ->
ToolIntentModel (ONE call: capability selection + argument extraction from Candidate
Cards) -> Binder (normalize/validate) -> Shared Tool Runtime. Every failure
exits to the Agent, byte-identical. ToolIntentModel is a swappable provider
(local primary, online fallback, stub) behind the tool_intent/ ladder.

Scope status: P0 moved the orchestration out of ``TurnOrchestrator``; P1 added
the Registry/Matcher. Single path (ruling 2026-09-28): the funnel IS the formal
routing lane — the dark-launch gate (``chat_funnel_enabled``) and the shadow
hook were deleted; ``funnel_live`` only keeps the fail-open safety semantics
(missing deps / guardrail veto). The legacy QIR lane was deleted with migration
0014 (live-table ruling 2026-09-26) — ``funnel_live`` + ``route`` are the whole
public surface.

Import-cycle note (2026-10-01): ``understanding`` imports ``actions``, whose
moved-name façade imports back into ``intent_funnel`` (``registry``/``binder``).
Importing a subpackage runs this package's ``__init__``, so an eager ``funnel``
import here used to re-enter a half-built ``understanding``. The cycle is now
broken structurally: ``funnel`` keeps ``TurnRequirements`` under ``TYPE_CHECKING``
and the runtime consumers (``orchestrator`` / ``preview`` / ``shadow``) import it
at call time, so nothing on the funnel import path pulls ``understanding`` while
it is mid-initialization. This module only re-exports the public API (see
CLAUDE.md, ``__init__.py`` Responsibility Rule).
"""
from __future__ import annotations

from . import contract
from .funnel import funnel_live, route

__all__ = ["contract", "funnel_live", "route"]
