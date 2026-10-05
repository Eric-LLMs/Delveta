"""SocialSearchHandler — argument acquisition for ``cap-social-search``.

Owns ONLY this capability's parameter preparation; it never executes a tool and
never re-implements the social/search adapters. It is fully DETERMINISTIC — no
model call — because the two failure modes the audit found here are both
fabrication: a model guessing a single platform the user never named, and a
model inventing a subreddit. The turn's sentence is the only source.

The four schema slots of ``cap-social-search``:

* ``query``    — the user's sentence, taken VERBATIM (strip only).
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
  fallback (``auto``) is benign.
* ``subreddit`` — emitted ONLY when the resolved platform is ``reddit`` AND the
  sentence names one (``r/<name>``); otherwise omitted. A subreddit is reddit's
  own scope, never invented for another platform.
* ``limit``    — NOT emitted: the ``search_social`` tool applies its own default
  (10, clamped to 1..25) whenever the slot is absent.

Note: ``platform`` is ALWAYS emitted (never omitted). Omitting it would let the
tool fall back to its own default of ``reddit`` — i.e. silently narrow an
unscoped turn to one platform — which is exactly the guess this handler refuses
to make; ``auto`` is the honest "no platform named" signal.

The returned draft (``{slot: value}``) is passed through the SAME Binder /
ActionExecutor / ToolRuntime handoff every other capability uses.
"""
from __future__ import annotations

import re

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


def _detect_platform(message: str) -> str:
    """The tool enum value for ``message``: the one named platform, else ``auto``."""
    named = [p for p, rx in _PLATFORM_RES.items() if rx.search(message)]
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
        # limit deliberately omitted: search_social defaults it (10, 1..25).
        return draft
