"""Wire format for the session-history page cursor.

The cursor names the oldest row a client already holds, as the same composite key the
transcript is ordered by: ``(created_at, turn_rank, id)``. ``created_at`` alone is not a
cursor — a turn's rows share one timestamp in history written before per-turn timestamps —
and ``turn_rank`` alone is not unique either, so the pair plus the row id is the position.

The value is opaque to clients: urlsafe-base64 of ``"<iso8601>|<rank>|<uuid>"`` with empty
rank/id for a page that ended on a timestamp boundary (no in-group split). Decoding is
strict: anything malformed is a 400, never a silently different page.
"""
from __future__ import annotations

import base64
from datetime import datetime
from uuid import UUID

_SEP = "|"


def encode_cursor(before) -> str | None:
    """Encode ``(created_at, turn_rank | None, id | None)``; ``None`` stays ``None``."""
    if before is None:
        return None
    ts, rank, mid = before
    raw = _SEP.join(
        [
            ts.isoformat() if ts is not None else "",
            "" if rank is None else str(int(rank)),
            "" if mid is None else str(mid),
        ]
    )
    return base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii").rstrip("=")


class CursorError(ValueError):
    """The ``before`` parameter did not decode to a usable position."""


def decode_cursor(raw: str) -> tuple[datetime, int | None, UUID | None]:
    """Decode a ``before`` value, or raise :class:`CursorError`."""
    try:
        padded = raw + "=" * (-len(raw) % 4)
        text = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
        ts_raw, rank_raw, id_raw = text.split(_SEP)
        ts = datetime.fromisoformat(ts_raw)
        rank = int(rank_raw) if rank_raw else None
        mid = UUID(id_raw) if id_raw else None
    except Exception as exc:  # noqa: BLE001 - every malformed shape is the same client error
        raise CursorError("malformed cursor") from exc
    if ts.tzinfo is None:
        raise CursorError("cursor timestamp has no timezone")
    if (rank is None) != (mid is None):
        raise CursorError("cursor rank and id must be given together")
    return ts, rank, mid
