"""Deterministic search-scope acquisition — the ONE rule for the search trio.

A turn's raw sentence can carry a STRUCTURED scope the user typed (today: an
explicit ``r/<name>`` reddit community). That scope has two consequences, and
both must be decided the SAME way everywhere — the runtime handler, the dataset
builder, the SFT scorer and the offline Formal-500 scorer:

* the scope becomes a deterministic, TOOL-ONLY execution argument
  (:func:`execution_scope_args`) — ``subreddit`` filled from the raw sentence and
  ``platform`` pinned to reddit so a scope can never be silently widened to a
  platform that cannot honour it;
* the pinned slot is REMOVED from the effective model slots for that turn
  (:func:`effective_model_slots`), so the model can never overwrite the
  deterministic pin and is never scored on a slot it was not asked for.

This module is the SINGLE source of the ``r/<name>`` shape and the slot decision.
No caller re-implements the regex or the slot narrowing; each caller supplies its
OWN static per-capability slot list (the runtime handler's, or the training
contract's), and this function applies only the deterministic per-turn delta.

``subreddit`` is NEVER a PE model slot: it stays out of the effective model slots,
the training target, the rendered output schema and the PE Argument EM. The
execution parameters are scored separately (:mod:`pe_score_v2`), never folded
into PE Argument EM.
"""
from __future__ import annotations

import re

# An explicit reddit community scope the user TYPED: ``r/<name>``. This is a fixed
# STRUCTURED token, not natural-language guessing — the same shape the reddit UI
# uses. The leading ``(?<![A-Za-z0-9_])`` prevents matching inside a word (e.g.
# ``car/road``); a subreddit name is 2..21 chars of ``[A-Za-z0-9_]``.
_SUBREDDIT_RE = re.compile(r"(?<![A-Za-z0-9_])r/([A-Za-z0-9][A-Za-z0-9_]{1,20})")

# The ONE capability with a deterministic execution scope today. A scope pins the
# slot(s) it owns; the pin value is the deterministic execution value.
_SOCIAL_SCOPE_PINS: dict[str, str] = {"platform": "reddit"}


def explicit_subreddit(message: str) -> str | None:
    """The reddit community named as a structured ``r/<name>`` token, or ``None``."""
    match = _SUBREDDIT_RE.search(str(message or ""))
    return match.group(1) if match else None


def execution_scope_args(capability_id: str, raw_query: str) -> dict[str, object]:
    """The deterministic, TOOL-ONLY arguments a scope injects for this turn.

    Empty when the sentence carries no structured scope. These keys are NEVER PE
    model slots — they are execution parameters the caller merges into the tool
    draft and reports under a SEPARATE correctness metric."""
    if capability_id == "cap-social-search":
        subreddit = explicit_subreddit(raw_query)
        if subreddit is not None:
            return {"subreddit": subreddit, **_SOCIAL_SCOPE_PINS}
    return {}


def effective_model_slots(capability_id: str, raw_query: str,
                          static_slots) -> tuple[str, ...]:
    """The per-sample model slots for ``capability_id`` on ``raw_query``.

    Starts from the caller's ``static_slots`` (the runtime handler's plan order, or
    the training contract's ``pe_contract.model_slots``) and REMOVES every slot a
    deterministic scope pins this turn — so the pin can never be overwritten by the
    model and the model is never scored on a slot it was not asked for. A turn with
    no scope keeps the static slots unchanged (bytes-identical for web/rag and for
    a generic social turn)."""
    pinned = set(execution_scope_args(capability_id, raw_query)) & set(static_slots)
    return tuple(slot for slot in static_slots if slot not in pinned)
