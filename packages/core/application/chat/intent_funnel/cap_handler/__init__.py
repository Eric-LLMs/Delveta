"""cap_handler — per-capability parameter handlers (Intent Funnel).

The public boundary of the cap_handler package: the ``CapabilityHandler``
contract, the ``HANDLERS`` wiring map, the ``handler_for`` lookup, and each
concrete handler. Wiring lives in ``roster.py``; handlers live in their own
``*_handler.py`` modules. Nothing else is exported.
"""
from __future__ import annotations

from .rag_search_handler import RagSearchHandler
from .read_document_handler import ReadDocumentHandler
from .roster import HANDLERS, CapabilityHandler, handler_for
from .social_search_handler import SocialSearchHandler
from .vision_handler import VisionHandler
from .web_search_handler import WebSearchHandler

__all__ = ["CapabilityHandler", "HANDLERS", "RagSearchHandler", "ReadDocumentHandler",
           "SocialSearchHandler", "VisionHandler", "WebSearchHandler", "handler_for"]
