"""Wire format for the session-list page cursor.

The list is ordered by ``(created_at DESC, id DESC)``. ``created_at`` alone is not a
cursor: sessions created in one transaction share a timestamp, so the pair plus the row
id names the position. The value is opaque to clients — urlsafe-base64 of
``"<iso8601>|<uuid>"`` — and decoding is strict: anything malformed is a 400, never a
silently different page.
"""
from __future__ import annotations

import base64
from datetime import datetime
from uuid import UUID

_SEP = "|"


def encode_session_cursor(before: tuple[datetime, UUID] | None) -> str | None:
    """Encode ``(created_at, id)``; ``None`` stays ``None``."""
    if before is None:
        return None
    ts, sid = before
    raw = f"{ts.isoformat()}{_SEP}{sid}"
    return base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii").rstrip("=")


class SessionCursorError(ValueError):
    """The ``before`` parameter did not decode to a usable position."""


def decode_session_cursor(raw: str) -> tuple[datetime, UUID]:
    """Decode a ``before`` value, or raise :class:`SessionCursorError`."""
    try:
        padded = raw + "=" * (-len(raw) % 4)
        text = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
        ts_raw, id_raw = text.split(_SEP)
        ts = datetime.fromisoformat(ts_raw)
        sid = UUID(id_raw)
    except Exception as exc:  # noqa: BLE001 - every malformed shape is the same client error
        raise SessionCursorError("malformed cursor") from exc
    if ts.tzinfo is None:
        raise SessionCursorError("cursor timestamp has no timezone")
    return ts, sid
