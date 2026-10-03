"""Production AcquisitionInputProvider — per-turn bridge to the live Registry.

Phase 4. The cascade resolves acquisition inputs per decided
capability through the injected seam (``executors.base.ChatDeps.acquisition_inputs``).
Tests inject their own provider; PRODUCTION has none, and this module builds the
real one ONCE per turn (it needs the live ``entries_by_id``, the query and the
turn facts, all of which the cascade holds).

:func:`for_turn` composes the three deterministic producers and nothing else:

* :mod:`.declaration`   — the capability's ``{slot: SlotDecl}`` (from its schema
  + legacy ``arg_slots``);
* :mod:`.evidence`      — the non-LLM ``{slot: source}`` signal (query evidence
  for MODEL slots);
* :mod:`.context_values` — the settled SYSTEM_BINDER values from the turn facts.

The MODEL side (``model_values`` / ``model_source``) is deliberately LEFT EMPTY:
the extractor fills it later, at the hop, from the sanctioned context bundle.
The provider never calls a model, never touches the Binder and never reads a DB.
"""
from __future__ import annotations

from .context_values import system_values_for
from .declaration import declaration_for
from .evidence import evidence_for
from .inputs import AcquisitionInputs


def for_turn(*, entries_by_id: dict, query: str, facts):
    """Build the production provider for ONE turn. The returned callable maps a
    decided ``capability_id`` to its :class:`AcquisitionInputs`, or ``None`` when
    the capability declares no acquisition (undeclared -> the ARP routes to the
    Agent, §G) or is absent from the active table."""
    def _provider(capability_id: str) -> AcquisitionInputs | None:
        entry = entries_by_id.get(capability_id)
        if entry is None:
            return None
        declaration = declaration_for(entry)
        if not declaration:
            return None
        evidence = evidence_for(entry, declaration, query=query)
        values, sources = system_values_for(entry, declaration, facts=facts)
        return AcquisitionInputs(
            declaration=declaration,
            evidence=evidence,
            system_values=values,
            system_sources=sources,
        )
    return _provider
