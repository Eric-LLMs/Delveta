"""Argument Path Router — the deterministic acquisition-strategy decision.

Phase 4 Step 1. Pure, deterministic, no LLM, no I/O, no wiring.

Given the Registry schema truth (``parameters``) and the capability's
``acquisition`` declaration (``{slot: SlotDecl}``), this module computes the
DOUBLE GATE of the contract §D and DERIVES the acquisition strategy of §B from
two independent dimensions — **Required Readiness**, then **Acquisition Need** —
never a global linear precedence ladder.

The two per-slot signals the contract does not itself compute are INJECTED, so
this module stays a pure function and never couples to the Binder's private
fact table nor to any parser:

* ``system_values`` — the deterministically-resolved values for SYSTEM_BINDER /
  TOOL_DEFAULT slots (Gate 1's "real value now");
* ``evidence`` — a pre-computed deterministic / schema-aware / non-LLM signal
  ``{slot: source}`` deciding whether an OPTIONAL MODEL slot has a real
  acquisition need (Gate 2, §E).

Nothing here calls the extractor, reads a Registry field, or touches the Binder.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from .contract import (
    OWNERSHIP_MODEL,
    OWNERSHIP_SYSTEM_BINDER,
    OWNERSHIP_TOOL_DEFAULT,
    OWNERSHIP_UNAVAILABLE,
    OWNERSHIPS,
    SOURCES_OF,
    SOURCE_CONVERSATION_5_USER_TURNS,
    SOURCE_QUERY,
    STRATEGY_CONTEXT_DIRECT,
    STRATEGY_MISSING,
    STRATEGY_MIXED,
    STRATEGY_QUERY_PLUS_5TURNS_TO_EXTRACTOR,
    STRATEGY_QUERY_TO_EXTRACTOR,
    Readiness,
    SlotDecl,
)

# The two "system" owners: filled by the turn's settled facts / tool default,
# never by the model. They are the non-MODEL side of a MIXED acquisition.
_SYSTEM_OWNERSHIPS = frozenset({OWNERSHIP_SYSTEM_BINDER, OWNERSHIP_TOOL_DEFAULT})

# The MODEL side may draw only from these (kept explicit so a bad injected
# evidence source cannot smuggle a system source into the model lane).
_MODEL_SOURCES = frozenset(SOURCES_OF[OWNERSHIP_MODEL])


@dataclass(frozen=True)
class RouteDecision:
    """The router's deterministic verdict — routing metadata only (8.8).

    * ``declared`` — the capability opted in (a non-empty declaration was
      supplied). ``False`` means the caller must NOT fall back to the legacy
      extractor; it exits to the Agent (acquisition-undeclared, §G).
    * ``strategy`` — one of the five STRATEGIES, or ``None`` when undeclared.
    * ``readiness`` — the double-gate result (§D).
    * ``model_slots`` — the MODEL-owned slots to acquire (the extractor schema
      set); empty for CONTEXT_DIRECT / MISSING.
    * ``system_slots`` — the system-owned slots that are actually filled (the
      non-MODEL side of a MIXED merge).
    * ``bundle_source`` — the context_bundle source the contextual strategies
      need (``QUERY`` or ``CONVERSATION_5_USER_TURNS``), or ``None``.
    """

    declared: bool
    strategy: str | None
    readiness: Readiness = field(default_factory=Readiness)
    model_slots: tuple[str, ...] = ()
    system_slots: tuple[str, ...] = ()
    bundle_source: str | None = None


def _ownership(declaration: Mapping[str, SlotDecl], slot: str) -> str:
    decl = declaration.get(slot)
    if decl is None:
        # A required parameter with no declaration entry: nothing can legally
        # fill it. Conservatively UNAVAILABLE -> UNSAT -> MISSING (no separate
        # "incomplete declaration" state is defined by the contract §C).
        return OWNERSHIP_UNAVAILABLE
    if decl.ownership not in OWNERSHIPS:
        return OWNERSHIP_UNAVAILABLE
    return decl.ownership


def _effective_sources(decl: SlotDecl) -> tuple[str, ...]:
    """The sources a slot may draw from: its declared ``allowed_sources`` when
    set, else the ownership's default source set (``SOURCES_OF``). Order is
    stabilised so the derivation is deterministic."""
    if decl.allowed_sources:
        return tuple(sorted(decl.allowed_sources))
    return tuple(sorted(SOURCES_OF.get(decl.ownership, frozenset())))


def _has_evidence(slot: str, decl: SlotDecl, evidence: Mapping[str, str]) -> bool:
    src = (evidence or {}).get(slot)
    return src in _MODEL_SOURCES and src in _effective_sources(decl)


def _model_slots_needing(parameters: Mapping, declaration: Mapping[str, SlotDecl],
                         evidence: Mapping[str, str]) -> list[str]:
    """Gate 2 (§D): a REQUIRED MODEL slot always needs acquisition; an OPTIONAL
    MODEL slot needs it IFF a deterministic evidence signal exists (§E). A slot
    merely present in the schema triggers nothing."""
    out: list[str] = []
    for slot, spec in (parameters or {}).items():
        if _ownership(declaration, slot) != OWNERSHIP_MODEL:
            continue
        required = bool((spec or {}).get("required", True))
        if required:
            out.append(slot)
        elif _has_evidence(slot, declaration[slot], evidence):
            out.append(slot)
    return out


def _filled_system_slots(declaration: Mapping[str, SlotDecl],
                         system_values: Mapping[str, str]) -> tuple[str, ...]:
    out = []
    for slot, decl in (declaration or {}).items():
        if decl.ownership in _SYSTEM_OWNERSHIPS and str(
                (system_values or {}).get(slot) or "").strip():
            out.append(slot)
    return tuple(out)


def _slot_acquisition_source(slot: str, decl: SlotDecl,
                             evidence: Mapping[str, str]) -> str | None:
    """Where a needed MODEL slot's value MUST come from (§B). Decided by the
    ACTUAL evidence signal alone — never inferred from ``allowed_sources``; the
    declaration's source list only gates whether a source is ACCEPTABLE, it does
    not CREATE evidence. Returns ``None`` when no legal source can be determined
    (-> MISSING / INVALID, §D), never a guess and never a legacy fallback."""
    src = (evidence or {}).get(slot)
    if src in _MODEL_SOURCES and src in _effective_sources(decl):
        return src
    return None


def readiness(parameters: Mapping, declaration: Mapping[str, SlotDecl], *,
              evidence: Mapping[str, str] | None = None,
              system_values: Mapping[str, str] | None = None) -> Readiness:
    """The DOUBLE GATE (§D).

    Gate 1 — Required Readiness: every REQUIRED slot must be deterministically
    ready now (SYSTEM_BINDER/TOOL_DEFAULT with a real value) — a MODEL one is
    PENDING_MODEL, an UNAVAILABLE (or undeclared) one is UNSAT.
    Gate 2 — Acquisition Need: see :func:`_model_slots_needing`."""
    evidence = evidence or {}
    system_values = system_values or {}
    unsatisfiable: list[str] = []
    pending: list[str] = []
    for slot, spec in (parameters or {}).items():
        if not bool((spec or {}).get("required", True)):
            continue
        ownership = _ownership(declaration, slot)
        if ownership in _SYSTEM_OWNERSHIPS:
            if not str(system_values.get(slot) or "").strip():
                unsatisfiable.append(slot)
        elif ownership == OWNERSHIP_MODEL:
            pending.append(slot)
        else:  # UNAVAILABLE, or a required slot with no declaration entry
            unsatisfiable.append(slot)
    ready = not unsatisfiable and not pending
    return Readiness(
        ready=ready,
        needs_acquisition=bool(
            _model_slots_needing(parameters, declaration, evidence)),
        unsatisfiable=tuple(unsatisfiable),
    )


def route(parameters: Mapping, declaration: Mapping[str, SlotDecl], *,
          evidence: Mapping[str, str] | None = None,
          system_values: Mapping[str, str] | None = None) -> RouteDecision:
    """Derive the acquisition strategy (§B). Two independent dimensions, no
    linear precedence ladder.

    An empty declaration is the OPT-IN boundary (§C/§G): the capability did not
    declare acquisition, so no strategy is produced and the caller exits to the
    Agent — never to the legacy extractor."""
    declaration = declaration or {}
    if not declaration:
        return RouteDecision(declared=False, strategy=None)

    rd = readiness(parameters, declaration,
                   evidence=evidence, system_values=system_values)
    if rd.unsatisfiable:
        # Gate 1 UNSAT: required args cannot be legally obtained -> Agent.
        return RouteDecision(declared=True, strategy=STRATEGY_MISSING, readiness=rd)

    needed = _model_slots_needing(parameters, declaration, evidence or {})
    if not needed:
        # All required slots deterministically ready, no MODEL acquisition need.
        return RouteDecision(declared=True, strategy=STRATEGY_CONTEXT_DIRECT,
                             readiness=rd)

    # A needed MODEL slot's source is EVIDENCE-DRIVEN (§B): if no actual evidence
    # signal (or one outside its allowed sources) exists, its source cannot be
    # legally determined -> MISSING / INVALID; never a guess, never a fallback.
    sources = {s: _slot_acquisition_source(s, declaration[s], evidence or {})
               for s in needed}
    if any(v is None for v in sources.values()):
        return RouteDecision(declared=True, strategy=STRATEGY_MISSING, readiness=rd)

    bundle_source = (
        SOURCE_CONVERSATION_5_USER_TURNS
        if any(v == SOURCE_CONVERSATION_5_USER_TURNS for v in sources.values())
        else SOURCE_QUERY
    )
    filled = _filled_system_slots(declaration, system_values or {})
    if filled:
        # A MODEL acquisition AND filled system-owned slots -> merge both sides.
        strategy = STRATEGY_MIXED
    elif bundle_source == SOURCE_CONVERSATION_5_USER_TURNS:
        strategy = STRATEGY_QUERY_PLUS_5TURNS_TO_EXTRACTOR
    else:
        strategy = STRATEGY_QUERY_TO_EXTRACTOR
    return RouteDecision(
        declared=True, strategy=strategy, readiness=rd,
        model_slots=tuple(needed), system_slots=filled,
        bundle_source=bundle_source,
    )
