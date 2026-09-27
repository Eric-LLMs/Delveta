"""Node 3 — ToolIntentModel: one call adjudicates the capability AND extracts arguments.

Chain ruling (2026-09-24): the funnel makes AT MOST one ToolIntentModel call per turn.
The former ``recheck`` second hop is deleted — it added zero information
(candidates narrowed to the one already picked, identical downstream outcome
for every verdict), and the two online TTFBs it cost were the cascade timeout.
Binder failures now exit straight to the Agent with BIND_* reasons.

Backend ladder (settings ``chat_tool_intent_backend``):
  ``stub``   — deterministic margin rules, NO extraction power -> arguments
               stay None -> BIND_MISSING exit (honest, documented);
  ``local``  — deployed small ToolIntentModel (first choice, ms-level);
  ``online`` — platform LLM route, minimal card payload (fallback);
  ``auto``   — local -> online -> stub (the deployed order of the ruling).

The contract each backend honors: CONFIDENT only with a real verdict on-card
and confidence above the floor; anything else — low confidence, off-card
invention, a backend that cannot serve — exits DOWN to the Agent (8.10),
never a fabricated route.

Package map: :mod:`.prompt` (card/prompt assembly), :mod:`.parser` (reply ->
verdict gate), :mod:`.base` (unavailable signal), :mod:`.backends` (model
access only), this module (the ONE-call dispatch ladder).
"""
from __future__ import annotations

import logging

from ..contract import TOOL_INTENT_UNCERTAIN, ToolIntentVerdict
from .base import ToolIntentUnavailable
from .parser import verdict_from_reply

logger = logging.getLogger(__name__)

BACKENDS = ("stub", "local", "online", "auto")


def _backend() -> str:
    from core.config import settings

    backend = (settings.chat_tool_intent_backend or "stub").strip().lower()
    if backend not in BACKENDS:
        logger.warning("unknown chat_tool_intent_backend=%r; using stub", backend)
        return "stub"
    return backend


async def _model_call(backend, query, candidates, entries_by_id, llm, facts) -> ToolIntentVerdict:
    from .backends import local, online

    if backend == "local":
        from core.config import settings

        data = await local.model_reply(query, candidates, entries_by_id,
                                 url=settings.chat_tool_intent_local_url, facts=facts)
    else:
        data = await online.model_reply(query, candidates, entries_by_id, llm=llm, facts=facts)
    return verdict_from_reply(data, candidates)


async def select_and_extract(query: str, candidates, *, entries_by_id: dict,
                     llm=None, facts=None) -> ToolIntentVerdict:
    """Run the ONE ToolIntentModel pass under the configured backend ladder.

    Per the 2026-09-26 ruling the funnel only calls this hop when the
    model-facing candidate set is NON-EMPTY (an empty set short-circuits to
    NO_CANDIDATE before any spend). An empty list remains a legitimate
    defensive input: the prompt renders the explicit "(none registered for
    this turn)" card set and the model can only answer NONE -> REJECT."""
    from core.config import settings

    backend = _backend()
    chain = ({"auto": ("local", "online", "stub"),
              "local": ("local",), "online": ("online",), "stub": ("stub",)}[backend])
    for step in chain:
        if step == "stub":
            from .backends import stub

            return stub.evaluate(candidates, margin=settings.chat_funnel_margin)
        try:
            return await _model_call(step, query, candidates, entries_by_id, llm, facts)
        except ToolIntentUnavailable as exc:
            logger.info("tool_intent %s unavailable (%r); falling through the ladder", step, exc)
    return ToolIntentVerdict(TOOL_INTENT_UNCERTAIN, None,
                             "no tool_intent backend served")  # pragma: no cover
