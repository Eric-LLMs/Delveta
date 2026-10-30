"""Intent Funnel — the independent, decoupled routing lane above the Agent.

Chain (single-path ruling): Matcher HIT -> Argument Path Router -> Binder ->
Shared Tool Runtime, with NO re-selection; Matcher MISS/AMBIGUOUS -> Recall ->
Candidate Aggregation -> cap_router -> Argument Path Router -> Binder. Capability
selection is ALWAYS the deployed cap_router service (LayaChoice); argument
acquisition is a separate concern and only calls the extractor when the chosen
strategy is MODEL-owned. Every failure exits to the Agent, byte-identical.

Scope status: P0 moved the orchestration out of ``TurnOrchestrator``; P1 added
the Registry/Matcher. Single path (ruling): the funnel IS the formal
routing lane — the dark-launch gate (``chat_funnel_enabled``) and the shadow
hook were deleted; ``funnel_live`` only keeps the fail-open safety semantics
(missing deps). The legacy QIR lane was deleted with migration 0014
(live-table ruling) and the legacy single-call ToolIntentModel selector was
deleted with the single-chain cleanup — ``funnel_live`` + ``route`` are the
whole public surface.

Import-cycle note: ``understanding`` imports ``actions``, whose
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
from .funnel import funnel_live, resume_with_llm, route

__all__ = ["contract", "funnel_live", "resume_with_llm", "route"]
