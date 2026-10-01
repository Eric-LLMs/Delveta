"""Acquisition inputs — the injected seam between the cascade and the ARP.

Phase 4 Step 2 (2026-10-01). :mod:`.path_router` is a PURE function of four
injected inputs and never reads a Registry field, a parser or a fact table. This
module defines the ONE shape those inputs take, plus the production default
provider. The cascade resolves it per decided capability from the injected deps
seam (``executors.base.ChatDeps.acquisition_inputs``) — it never builds one
itself and never invents a slot->source mapping (§C/§E).

* ``declaration`` — the capability's ``{slot: SlotDecl}`` acquisition entry (the
  OPT-IN, §C). Empty ⇒ acquisition-undeclared ⇒ Agent (§G).
* ``evidence``    — the deterministic, schema-aware, non-LLM ``{slot: source}``
  signal (§E). Produced OUTSIDE the router; this step wires NO real producer.
* ``system_values`` / ``system_sources`` — the deterministically-resolved
  SYSTEM_BINDER / TOOL_DEFAULT slot values (Gate 1, §D) and the ACTUAL source
  each was resolved from (merge provenance, §B).
* ``model_values`` / ``model_source`` — the MODEL side of a MIXED merge (Step 3,
  §F/§B): the acquired MODEL-owned values and the ACTUAL acquisition source they
  came from. In production Qwen is not wired, so these stay empty and every
  MODEL strategy exits ``ACQUISITION_MODEL_PENDING``; tests inject them to
  exercise the merge deterministically. An empty ``model_source`` means the
  MODEL value has no actual source — it is left unprovenanced and never merged.

Nothing here calls Qwen, touches the Binder, reads a DB or resolves context.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from .contract import SlotDecl


@dataclass(frozen=True)
class AcquisitionInputs:
    """The ARP inputs for exactly ONE decided capability: the four the router
    reads (declaration / evidence / system_values / system_sources) plus the
    MIXED merge's MODEL side (``model_values`` / ``model_source``)."""

    declaration: Mapping[str, SlotDecl] = field(default_factory=dict)
    evidence: Mapping[str, str] = field(default_factory=dict)
    system_values: Mapping[str, str] = field(default_factory=dict)
    system_sources: Mapping[str, str] = field(default_factory=dict)
    # MIXED seam (Step 3): the acquired MODEL values and their ACTUAL source.
    # Production wires no Qwen, so both stay empty -> MODEL strategies exit
    # ACQUISITION_MODEL_PENDING; tests inject them.
    model_values: Mapping[str, str] = field(default_factory=dict)
    model_source: str = ""


# A provider resolves the inputs of ONE capability id, or None when the
# capability carries no acquisition declaration on the new lane. This step ships
# NO production provider (see :func:`no_acquisition_inputs`); tests inject one.
AcquisitionInputProvider = Callable[[str], "AcquisitionInputs | None"]


def no_acquisition_inputs(capability_id: str) -> None:
    """The PRODUCTION default provider: this step wires no real producer, so
    every capability is acquisition-undeclared → the ARP returns
    ``declared=False`` → the turn exits to the Agent (§G). Deterministic, no
    I/O, no Qwen, no Registry read — the honest conservative default."""
    return None
