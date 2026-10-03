"""cap_router backends — package surface (re-exports only).

Responsibilities are separated and this module holds NONE of them:

* the selector CONTRACT (``CapabilitySelector``) lives in :mod:`..base`;
* backend resolution (:func:`selector_for`) lives in :mod:`.factory`;
* each concrete backend lives in its own module (:mod:`.stub`, :mod:`.service`).

The re-exports below preserve the historic import surface
(``from ..cap_router.backends import CapabilitySelector, selector_for``).
"""
from __future__ import annotations

from ..base import CapabilitySelector
from .factory import selector_for

__all__ = ["CapabilitySelector", "selector_for"]
