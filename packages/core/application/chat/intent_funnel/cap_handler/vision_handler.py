"""VisionHandler — argument acquisition for ``cap-vision``.

Owns ONLY this capability's parameter preparation; it never loads image bytes
and never re-implements any recognition logic — the real asset load and the
multimodal call stay entirely inside the shared ``vision`` tool. The two schema
slots of ``cap-vision``:

* ``asset_id`` — the ONLY legal source is :class:`~..contract.TurnFacts`. The id
  is resolved with the SAME precedence the Binder uses
  (``attachment_asset_id`` -> ``path_asset_id`` -> ``viewer_asset_id``); a
  sentence-copied or model-invented id is never a source. There is NO
  fail-open path: when no fact carries an asset id the handler returns ``None``
  and the turn exits to the Agent's existing missing/clarify handling — never a
  guessed id, never a fabricated path, never an empty string.
* ``question`` — the user's sentence, taken VERBATIM (strip only), so the
  question they ask ABOUT the image reaches the model unaltered. Omitted when
  the message is empty (the tool then runs its own default analysis prompt).

The returned draft (``{slot: value}``) is passed through the SAME Binder /
ActionExecutor / ToolRuntime handoff every other capability uses — the Binder
re-resolves ``asset_id`` from the same facts, so its value is identical here.
"""
from __future__ import annotations

# Mirrors binder._CONTEXT_SLOT_SOURCES["asset_id"] exactly: attachment first,
# then the drive path, then the on-screen viewer. Keeping the two in lockstep
# means the draft this handler emits is the value the Binder settles on.
_ASSET_FACT_FIELDS = ("attachment_asset_id", "path_asset_id", "viewer_asset_id")


def _resolve_asset_id(facts) -> str:
    """The first non-empty asset id among the context facts, else ``""``."""
    if facts is None:
        return ""
    for field in _ASSET_FACT_FIELDS:
        value = str(getattr(facts, field, "") or "").strip()
        if value:
            return value
    return ""


class VisionHandler:
    """``cap-vision`` parameter handler (stateless)."""

    capability_id = "cap-vision"

    async def acquire(self, *, query: str, facts) -> dict[str, object] | None:
        """Return the ``vision`` argument draft, or ``None`` when no image is in
        context (fail-closed — the Agent owns the clarification)."""
        asset_id = _resolve_asset_id(facts)
        if not asset_id:
            return None
        draft: dict[str, object] = {"asset_id": asset_id}
        message = str(query or "").strip()
        if message:
            draft["question"] = message
        return draft
