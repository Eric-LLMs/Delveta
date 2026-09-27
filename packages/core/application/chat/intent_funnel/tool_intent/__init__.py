"""Node 3 — ToolIntentModel package surface.

Implementation map: :mod:`.model` (the ONE-call dispatch ladder),
:mod:`.prompt` (card/prompt assembly), :mod:`.parser` (reply -> verdict gate),
:mod:`.base` (backend-independent contracts), :mod:`.backends` (model access).
The re-exports below are the compatibility surface for historic import and
monkeypatch sites.
"""
from __future__ import annotations

from .backends import local, online, stub
from .base import ToolIntentUnavailable
from .model import BACKENDS, select_and_extract
from .parser import verdict_from_reply as _verdict_from_reply

__all__ = ["select_and_extract", "BACKENDS", "ToolIntentUnavailable"]
