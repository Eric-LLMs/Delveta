"""SocialSearchHandler — argument acquisition for ``cap-social-search``.

Owns ONLY this capability's parameter preparation; it never executes a tool and
never re-implements the social/search adapters. The turn's sentence is the only
value source. This handler runs NO natural-language extraction rule (no regex /
token / quoted-phrase detection): every natural-language slot is understood by
the unified extractor (the acquisition contract's MODEL lane).

The four schema slots of ``cap-social-search``:

* ``query``    — the user's sentence, taken VERBATIM by ``acquire()`` as the
  deterministic FALLBACK only. ``slot_plan()`` authorizes the shared extractor
  to return a CLEANED search TOPIC; only a valid, constraint-passing model value
  replaces the verbatim sentence.
* ``platform`` — a MODEL-owned optional slot (tool enum ``{reddit, x, zhihu,
  auto}``): the model returns the platform the user named, and OMITS it when the
  sentence names none. No token list is scanned here (the former deterministic
  platform detector was removed — it had no requirement basis). Whether an
  unnamed platform should default to ``auto`` vs. the tool's own default is an
  OPEN contract question, not decided here.
* ``subreddit`` — a MODEL-owned optional slot: the model returns it only when a
  reddit scope is named; it is never invented.
* ``limit``    — a MODEL-owned optional slot: the stated count is understood by
  the model; when omitted the ``search_social`` tool applies its own default
  (10, clamped to 1..25).

The returned draft (``{slot: value}``) is passed through the SAME Binder /
ActionExecutor / ToolRuntime handoff every other capability uses.
"""
from __future__ import annotations

from .slot_plan import SlotPlan


class SocialSearchHandler:
    """``cap-social-search`` parameter handler (stateless)."""

    capability_id = "cap-social-search"

    async def acquire(self, *, query: str, facts) -> dict[str, object] | None:
        """Return the ``search_social`` argument draft, or ``None`` when no query
        exists (the Agent then owns the turn). ``facts`` is accepted for the
        common handler contract but unused — social search takes nothing from
        turn context. The verbatim sentence is the fallback the model may clean.
        """
        message = str(query or "").strip()
        if not message:
            return None
        return {"query": message}

    def slot_plan(self, *, query: str, facts, draft: dict) -> SlotPlan:
        """Authorize the model for the natural-language slots it owns: the
        cleaned ``query`` topic, the named ``platform`` / ``subreddit`` and the
        stated ``limit`` count. An absent count stays the tool's own default."""
        return SlotPlan(model_slots=("query", "platform", "subreddit", "limit"),
                        default_slots=("limit",))
