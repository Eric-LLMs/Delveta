"""TranslateHandler — argument acquisition for ``cap-translate``.

``cap-translate`` (tool binding ``translate``) does NOT run the shared Qwen
extractor. The handler certifies exactly ONE deterministic slot:

* ``text`` — the user's sentence, copied VERBATIM (strip only). It satisfies the
  Registry-required slot and is the instruction the ``translate`` tool reads;
  the tool then decides what to translate (that sentence, a previous chat
  message, the open document page, or the current video) from the turn material
  and returns ONLY the translation.

No ``slot_plan`` exists, so the orchestrator never reaches the extractor on this
capability. The target language is also NOT extracted: the tool's decision call
reads it from the user's sentence (default English).

VIDEO: the translate tool cannot read a video file, so the current frame must
come from the PC client's playhead. ``capture_for`` reuses the SAME deferred
capture lane vision owns (SSE ``capture-request`` -> ``resume_plan``); the
already-decided ``text`` rides ``DeferredCapture.args`` across the pause, and
``resume_plan`` injects only the uploaded frame's ``asset_id``. Every other
viewer kind needs no capture: a cloud document is read server-side by the tool.
"""
from __future__ import annotations

from .deferred_capture import DeferredCapture

_EXPLICIT_ASSET_FIELDS = ("attachment_asset_id", "path_asset_id")


def _explicit_asset(facts) -> str:
    """An attached / drive-path asset already in context, else ``""``.

    ``viewer_asset_id`` is deliberately NOT consulted: a VIDEO viewer carries the
    video's own asset, which the translate tool cannot read as a picture — the
    playhead frame must still come from the client (the same rule ``vision`` uses).
    """
    if facts is None:
        return ""
    for field in _EXPLICIT_ASSET_FIELDS:
        value = str(getattr(facts, field, "") or "").strip()
        if value:
            return value
    return ""


class TranslateHandler:
    """``cap-translate`` parameter handler (stateless)."""

    capability_id = "cap-translate"

    async def acquire(self, *, query: str, facts) -> dict[str, object] | None:
        """Return the verbatim sentence as ``text``; only a blank query abstains."""
        message = str(query or "")
        if not message.strip():
            return None
        return {"text": message}

    def capture_for(self, *, query: str, facts) -> DeferredCapture | None:
        """The video frame this turn needs, or ``None`` when no capture applies.

        Only a VIDEO viewer needs a client capture (the tool cannot read a video
        file). An attachment / drive-path asset is already a readable source, so
        it never triggers a capture. The timestamp is optional: the client
        answers with its live playhead.
        """
        if facts is None or not facts.has_viewer:
            return None
        if _explicit_asset(facts):
            return None
        if str(getattr(facts, "viewer_kind", "") or "").lower() != "video":
            return None
        message = str(query or "").strip()
        if not message:
            return None
        t_ms = getattr(facts, "viewer_t_ms", None)
        return DeferredCapture(
            kind="frame",
            name=str(getattr(facts, "viewer_name", "") or ""),
            t_ms=int(t_ms) if isinstance(t_ms, (int, float)) else None,
            duration_ms=getattr(facts, "viewer_duration_ms", None),
            args={"text": message},
        )
