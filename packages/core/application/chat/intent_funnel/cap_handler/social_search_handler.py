"""SocialSearchHandler — argument acquisition for ``cap-social-search``.

Owns ONLY this capability's parameter preparation; it never executes a tool and
never re-implements the social/search adapters. The platform/subreddit/limit
DISPOSITION is deterministic (no fabrication: a single named platform is used,
none/several fall back to ``auto``, a subreddit is never invented); the shared
extractor is consulted ONLY for the ``query`` cleanup and the no-platform-named
fallback, both authorized by ``slot_plan()``. The turn's sentence is the only
value source.

The four schema slots of ``cap-social-search``:

* ``query``    — the user's sentence, taken VERBATIM (strip only) as the
  deterministic FALLBACK. ``slot_plan()`` additionally authorizes the shared
  extractor to return a CLEANED search topic; only a valid, constraint-passing
  model value replaces the verbatim sentence.
* ``platform`` — DETECTED deterministically from the sentence over the tool's
  real enum ``{reddit, x, zhihu, auto}``:
    * exactly ONE platform is named -> that platform;
    * NONE named -> ``auto`` (the tool's "merge every configured platform" mode);
    * SEVERAL named -> ``auto`` (a set cannot be expressed as one enum, and a
      single platform would silently drop the others).
  A named platform is matched on unambiguous tokens only (``reddit`` / ``r/<sub>``
  / ``twitter`` / ``x.com`` / ``推特`` / ``zhihu`` / ``知乎``). The bare letter
  ``x`` is deliberately NOT matched: it collides with ``RTX`` / ``X-ray`` /
  ``X战警`` / ``x^2`` far more often than it means the platform, and the safe
  fallback (``auto``) is benign. ``slot_plan()`` adds a MODEL fallback ONLY when
  NO platform is named by those tokens: the model may then supply one that DET's
  token list misses, while the ``auto`` value stays the fallback. A sentence
  naming SEVERAL platforms keeps the honest ``auto`` and is never narrowed by a
  model.
* ``subreddit`` — emitted ONLY when the resolved platform is ``reddit`` AND the
  sentence names one (``r/<name>``); otherwise omitted. A subreddit is reddit's
  own scope, never invented for another platform.
* ``limit``    — a RULE-obtained value: extracted by the shared DET count rule
  when the sentence states a count (``acquire()``), so the model never re-derives
  it; otherwise the ``search_social`` tool applies its own default (10, clamped
  to 1..25).

Note: ``platform`` is ALWAYS emitted (never omitted). Omitting it would let the
tool fall back to its own default of ``reddit`` — i.e. silently narrow an
unscoped turn to one platform — which is exactly the guess this handler refuses
to make; ``auto`` is the honest "no platform named" signal.

The returned draft (``{slot: value}``) is passed through the SAME Binder /
ActionExecutor / ToolRuntime handoff every other capability uses.
"""
from __future__ import annotations

import re

from .search_args import result_count
from .slot_plan import SlotPlan

# Unambiguous platform tokens only. Order of keys is irrelevant: the detector
# counts how many DISTINCT platforms matched, not which one matched first.
_REDDIT_RE = re.compile(r"reddit|红迪|(?<![A-Za-z0-9])r/[A-Za-z0-9_]+", re.IGNORECASE)
_X_RE = re.compile(r"twitter|推特|x\.com", re.IGNORECASE)
_ZHIHU_RE = re.compile(r"zhihu|知乎", re.IGNORECASE)

# ``r/<name>`` with a non-alphanumeric left boundary so "car/1" never matches.
_SUBREDDIT_RE = re.compile(r"(?<![A-Za-z0-9])r/([A-Za-z0-9_]+)", re.IGNORECASE)

_PLATFORM_RES = {
    "reddit": _REDDIT_RE,
    "x": _X_RE,
    "zhihu": _ZHIHU_RE,
}


def _named_platforms(message: str) -> list[str]:
    """The DISTINCT platforms named by unambiguous DET tokens, in enum order."""
    return [p for p, rx in _PLATFORM_RES.items() if rx.search(message)]


def _detect_platform(message: str) -> str:
    """The tool enum value for ``message``: the one named platform, else ``auto``."""
    named = _named_platforms(message)
    if len(named) == 1:
        return named[0]
    # none named -> no guess; several named -> a set is not one enum. Both -> auto.
    return "auto"


class SocialSearchHandler:
    """``cap-social-search`` parameter handler (stateless)."""

    capability_id = "cap-social-search"

    async def acquire(self, *, query: str, facts) -> dict[str, object] | None:
        """Return the ``search_social`` argument draft, or ``None`` when no query
        exists (the Agent then owns the turn). ``facts`` is accepted for the
        common handler contract but unused — social search takes nothing from
        turn context.
        """
        message = str(query or "").strip()
        if not message:
            return None
        platform = _detect_platform(message)
        draft: dict[str, object] = {"query": message, "platform": platform}
        # subreddit is reddit's own scope and is emitted only when actually named.
        if platform == "reddit":
            match = _SUBREDDIT_RE.search(message)
            if match:
                draft["subreddit"] = match.group(1)
        # A stated count is a RULE-obtained value (DET), kept as-is; an absent one
        # stays the tool's default (slot_plan resolves which).
        count = result_count(message)
        if count is not None:
            draft["limit"] = count
        return draft

    def slot_plan(self, *, query: str, facts, draft: dict) -> SlotPlan:
        """Authorize the model for ``query`` (a cleaned topic); and for
        ``platform`` ONLY when the sentence names NO platform by DET tokens (the
        genuinely-needed fallback — the model may spot one DET's list misses,
        while ``auto`` stays the fallback). A multi-platform sentence keeps its
        honest ``auto``. ``limit`` is the DET count rule's value when the sentence
        states one (already in the draft), otherwise the tool's default — never
        asked of a model."""
        message = str(query or "")
        model_slots: list[str] = ["query"]
        if (str(draft.get("platform") or "") == "auto"
                and not _named_platforms(message)):
            model_slots.append("platform")
        default_slots = () if "limit" in draft else ("limit",)
        return SlotPlan(model_slots=tuple(model_slots), default_slots=default_slots)
