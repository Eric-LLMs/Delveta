"""Deferred client capture — the picture a certified turn still owes.

A capability handler emits this ONLY after the funnel has already decided the
turn's tool (the handler never fabricates a capture request for an ordinary chat
turn). The orchestrator pauses the stream, the PC client captures/uploads, and
the SAME certified turn resumes with ``asset_id`` filled in — no second funnel
run, no second argument extraction.

``args`` carries the capability's own already-decided arguments across that
pause. ``translate`` needs its verbatim sentence (it is not a model slot, but
the action cannot validate without it); the vision lane leaves ``args`` empty
and keeps using ``question``. ``asset_id`` itself is NEVER carried here — the
resume injects the uploaded capture's id, so the picture can only ever be the
client's real capture.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DeferredCapture:
    """A picture the turn still owes: the client must capture it before execution."""

    kind: str                       # "page" | "frame" | "image"
    name: str = ""
    page: int | None = None
    t_ms: int | None = None
    duration_ms: int | None = None
    question: str = ""
    # Already-decided action arguments that must survive the pause (translate's
    # ``text``). ``asset_id`` is injected on resume, never taken from here.
    args: dict[str, object] = field(default_factory=dict)

    def request(self) -> dict:
        """Wire shape of the ``capture-request`` SSE frame (client-facing spec)."""
        return {
            "kind": self.kind,
            "name": self.name,
            "page": self.page,
            "t_ms": self.t_ms,
            "duration_ms": self.duration_ms,
            "question": self.question,
        }
