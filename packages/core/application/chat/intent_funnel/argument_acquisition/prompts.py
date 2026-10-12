"""System-prompt registry for the model-owned argument extractor.

Every capability's system prompt is the SAME minimal INTERFACE prompt: it states
ONLY the interface (one already-chosen tool; fill the slots the request lists,
from the user's sentence) plus the output discipline. It carries NO semantic rule
— no "strip the execution frame", no "remove the result count", no "strip the
container word", no "never translate / reformat", no per-capability prose.

The model learns the extraction BEHAVIOUR from the SFT data. The per-capability
semantics live in the TOOL DESCRIPTION and the SLOT DESCRIPTIONS carried by
``tools`` (Registry truth, mirrored by ``pe_contract``) — the SAME face sent at
training time and at inference time. A rule expressed as prose here would make
the training prompt and the production prompt disagree the moment either drifts,
which is exactly what this registry exists to prevent.

No capability overrides the interface prompt, so :func:`prompt_for` returns the
shared :data:`..extractor.SYSTEM_EXTRACT` for every capability — byte-identical
to the extractor's own default. The registry stays explicit (rather than deleted)
so a future capability CAN pin its own interface prompt without touching callers.
"""
from __future__ import annotations

from .extractor import SYSTEM_EXTRACT

# capability_id -> system prompt. Empty by construction: the per-capability rule
# prompts were removed so training and production carry ONE interface prompt.
_PROMPTS: dict[str, str] = {}


def prompt_for(capability_id: str) -> str:
    """The system prompt for ``capability_id`` — the shared interface prompt (the
    extractor default) unless a capability registers its own."""
    return _PROMPTS.get(str(capability_id or ""), SYSTEM_EXTRACT)
