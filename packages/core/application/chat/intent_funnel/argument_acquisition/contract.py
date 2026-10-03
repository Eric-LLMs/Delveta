"""Argument Acquisition contract — the declaration + runtime vocabulary
(Phase 2, 2026-10-01).

Two ORTHOGONAL dimensions, kept apart on purpose (ruling 2026-10-01):

* ``ownership``  — WHO may produce a slot value. ``MODEL`` (extracted by Qwen),
  ``SYSTEM_BINDER`` (filled by the Binder from settled turn facts),
  ``TOOL_DEFAULT`` (a runtime default), ``UNAVAILABLE`` (never produced).
* ``source``     — WHERE a value actually came from. ``QUERY``,
  ``UI_CONTEXT``, ``CONVERSATION_5_USER_TURNS``, ``CALLBACK_CONTEXT``,
  ``RESOLVER``, ``DEFAULT``.

``Registry.parameters`` (required / type / enum / description) stays the ONLY
schema truth; ``arg_slots`` declares acquisition METADATA only
(``ownership`` / ``allowed_sources`` / ``escalation``) and never overrides or
duplicates the schema. Opt-in is explicit: a capability with NO ``arg_slots``
keeps legacy semantics untouched (:func:`is_declared`). This module is the pure
vocabulary — the deterministic Path Router that DERIVES a strategy from a
declaration lives in :mod:`..path_router`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ── ownership: WHO may produce a slot value ────────────────────────────────────────
OWNERSHIP_MODEL = "MODEL"
OWNERSHIP_SYSTEM_BINDER = "SYSTEM_BINDER"
OWNERSHIP_TOOL_DEFAULT = "TOOL_DEFAULT"
OWNERSHIP_UNAVAILABLE = "UNAVAILABLE"
OWNERSHIPS = frozenset({
    OWNERSHIP_MODEL, OWNERSHIP_SYSTEM_BINDER,
    OWNERSHIP_TOOL_DEFAULT, OWNERSHIP_UNAVAILABLE,
})

# ── source: WHERE the value may come from ──────────────────────────────────────────
SOURCE_QUERY = "QUERY"
SOURCE_UI_CONTEXT = "UI_CONTEXT"
SOURCE_CONVERSATION_5_USER_TURNS = "CONVERSATION_5_USER_TURNS"
SOURCE_CALLBACK_CONTEXT = "CALLBACK_CONTEXT"
SOURCE_RESOLVER = "RESOLVER"
SOURCE_DEFAULT = "DEFAULT"
SOURCES = frozenset({
    SOURCE_QUERY, SOURCE_UI_CONTEXT, SOURCE_CONVERSATION_5_USER_TURNS,
    SOURCE_CALLBACK_CONTEXT, SOURCE_RESOLVER, SOURCE_DEFAULT,
})

# Invariant I4 (ruling 2026-10-01): allowed_sources MUST be a subset of the
# sources its ownership can legitimately draw from — one table, no drift.
SOURCES_OF: dict[str, frozenset[str]] = {
    OWNERSHIP_MODEL: frozenset({SOURCE_QUERY, SOURCE_CONVERSATION_5_USER_TURNS}),
    OWNERSHIP_SYSTEM_BINDER: frozenset({
        SOURCE_UI_CONTEXT, SOURCE_RESOLVER, SOURCE_CALLBACK_CONTEXT}),
    OWNERSHIP_TOOL_DEFAULT: frozenset({SOURCE_DEFAULT}),
    OWNERSHIP_UNAVAILABLE: frozenset(),
}

# ── acquisition strategies (the Path Router's OUTPUT label) ────────────────────────
# Strategy names are MODEL-AGNOSTIC: they name WHERE a MODEL-owned value comes
# from (the query, or the query plus the last 5 user turns), never WHICH model
# extracts it. The extractor implementation is recorded separately in telemetry
# (currently Qwen); swapping it must not rename a strategy.
STRATEGY_CONTEXT_DIRECT = "CONTEXT_DIRECT"
STRATEGY_QUERY_TO_EXTRACTOR = "QUERY_TO_EXTRACTOR"
STRATEGY_QUERY_PLUS_5TURNS_TO_EXTRACTOR = "QUERY_PLUS_5TURNS_TO_EXTRACTOR"
STRATEGY_MIXED = "MIXED"
STRATEGY_MISSING = "MISSING"
STRATEGIES = frozenset({
    STRATEGY_CONTEXT_DIRECT, STRATEGY_QUERY_TO_EXTRACTOR,
    STRATEGY_QUERY_PLUS_5TURNS_TO_EXTRACTOR, STRATEGY_MIXED, STRATEGY_MISSING,
})


def is_declared(arg_slots: dict | None) -> bool:
    """The acquisition OPT-IN switch (ruling 2026-10-01): a capability enters the
    new acquisition contract IFF it declares a NON-EMPTY ``arg_slots``.

    Absent / empty -> pure LEGACY semantics: no :class:`SlotDecl` is synthesized,
    no ownership is reinterpreted, and the capability never routes through the
    new acquisition path. This is deliberate — several slots are ALREADY
    Binder-owned today (``asset_id`` via :data:`..binder._CONTEXT_SLOT_SOURCES`);
    a blanket default would silently rewrite their owner. ``arg_slots`` presence
    is the explicit opt-in, nothing else is.
    """
    return bool(arg_slots)


@dataclass(frozen=True)
class SlotDecl:
    """The acquisition METADATA of one slot (``arg_slots[slot]``). NEVER carries
    ``required``/``type``/``enum``/``description`` — those live in
    ``Registry.parameters`` alone (invariant I1).

    ``ownership`` is a REQUIRED field (no default): a declared slot must state
    its owner explicitly. There is deliberately NO "default SlotDecl" — see
    :func:`is_declared`."""

    ownership: str
    allowed_sources: tuple[str, ...] = ()
    escalation: tuple[str, ...] = ()


@dataclass(frozen=True)
class Readiness:
    """Result of the two deterministic gates (ruling 2026-10-01):

    * ``ready`` — every REQUIRED slot has a valid deterministic value already
      (gate 1, Required Readiness);
    * ``needs_acquisition`` — a MODEL-owned slot must be obtained (gate 2:
      ``MUST`` required-MODEL, or ``MAY`` optional-MODEL with query/history
      evidence);
    * ``unsatisfiable`` — required slots that can be filled by nothing (exit to
      a callback if declared, else MISSING -> Agent)."""

    ready: bool = False
    needs_acquisition: bool = False
    unsatisfiable: tuple[str, ...] = ()


@dataclass(frozen=True)
class ArgumentProvenance:
    """Per-slot provenance for audit / E2E: ``source`` (WHERE) and
    ``produced_by`` (WHO, = the slot's ownership). ``detail`` optionally names
    the fine-grained UI fact (``attachment`` / ``viewer.asset_id`` / ``path`` /
    ...) behind a ``UI_CONTEXT`` value, reusing the existing fact vocabulary."""

    slot: str
    source: str
    produced_by: str
    detail: str = field(default="", repr=False)
