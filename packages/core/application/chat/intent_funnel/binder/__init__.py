"""Binder node — package surface.

ACTIVE path: :func:`validate` (implementation in :mod:`.binder` +
:mod:`.validator`). LEGACY lanes: :mod:`.legacy`, re-exported here only as
the audited compatibility façade (``chat.actions`` lazy surface, L0/executor
call sites, p5 monkeypatch seam). The active funnel never imports legacy
directly.
"""
from __future__ import annotations

from .binder import validate
from .legacy import bind, bind_arguments, validate_action

__all__ = ["validate", "bind", "bind_arguments", "validate_action"]
