"""cap_router backends — package surface (re-exports only).

Responsibilities are separated and this module holds NONE of them:

* the selector CONTRACT (``CapabilitySelector``) lives in :mod:`..base`;
* production resolution (:func:`default_selector`) lives in :mod:`.factory`;
* each concrete selector lives in its own module (:mod:`.stub`, :mod:`.service`).

The re-exports below preserve the historic import surface
(``from ..cap_router.backends import CapabilitySelector, default_selector``).
"""
from __future__ import annotations

from ..base import CapabilitySelector
from .factory import default_selector

__all__ = ["CapabilitySelector", "default_selector"]
