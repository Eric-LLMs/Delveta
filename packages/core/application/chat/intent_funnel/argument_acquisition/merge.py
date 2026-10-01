"""Argument merge — system values + MODEL values -> one draft (Phase 4 Step 1).

The MIXED strategy's second half (contract §B): merge the deterministically
resolved system-owned slots (SYSTEM_BINDER / TOOL_DEFAULT) with the MODEL-owned
slots an acquisition returns, and record per-slot provenance
(:class:`ArgumentProvenance`, the frozen type — no new provenance kind is
introduced here).

The system value is TRUTH in both directions (§F): a MODEL value that collides
with a system-owned slot is dropped, never merged over it. Merging is
side-effect-free and does not touch the Binder — the Binder still validates the
merged draft downstream.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .contract import (
    OWNERSHIP_MODEL,
    OWNERSHIP_SYSTEM_BINDER,
    ArgumentProvenance,
    SlotDecl,
)


@dataclass(frozen=True)
class MergeResult:
    """The merged draft plus one provenance entry per slot actually present.

    ``unprovenanced`` names the slots whose value arrived WITHOUT an actual
    source: they are neither merged nor given a provenance record (never
    fabricated, §B/§F)."""

    args: dict[str, str]
    provenance: tuple[ArgumentProvenance, ...] = ()
    unprovenanced: tuple[str, ...] = ()


def merge(system_values: Mapping[str, str],
          model_values: Mapping[str, str], *,
          declaration: Mapping[str, SlotDecl] | None = None,
          system_sources: Mapping[str, str] | None = None,
          model_source: str) -> MergeResult:
    """Merge the system side and the MODEL side, system value winning on any
    collision, and emit per-slot provenance.

    Provenance records the ACTUAL source, supplied as inputs: ``model_source``
    (the acquisition source, query or last-5 user turns) for the MODEL side and
    ``system_sources`` for the system side. A declaration's ``allowed_sources``
    is NEVER used as a provenance fallback — a slot whose actual source is
    unknown is left unprovenanced rather than fabricated.

    ``model_values`` may only contribute MODEL-owned slots; anything else is
    stripped (a model-authored system value is never trusted)."""
    declaration = declaration or {}
    system_sources = system_sources or {}
    args: dict[str, str] = {}
    prov: dict[str, ArgumentProvenance] = {}
    unprovenanced: list[str] = []

    for slot, value in (model_values or {}).items():
        decl = declaration.get(slot)
        if decl is not None and decl.ownership != OWNERSHIP_MODEL:
            continue  # a model value on a system slot: stripped, system is truth
        if not model_source:
            unprovenanced.append(slot)  # no actual source -> cannot record it
            continue
        args[slot] = value
        prov[slot] = ArgumentProvenance(slot=slot, source=model_source,
                                        produced_by=OWNERSHIP_MODEL)

    for slot, value in (system_values or {}).items():
        source = system_sources.get(slot)
        if not source:
            # No ACTUAL source supplied: the contract cannot form legal
            # provenance, so the slot is not merged (never fabricated).
            unprovenanced.append(slot)
            continue
        decl = declaration.get(slot)
        produced_by = decl.ownership if decl is not None else OWNERSHIP_SYSTEM_BINDER
        args[slot] = value  # system wins
        prov[slot] = ArgumentProvenance(slot=slot, source=source,
                                        produced_by=produced_by)

    return MergeResult(args=args, provenance=tuple(prov.values()),
                       unprovenanced=tuple(unprovenanced))
