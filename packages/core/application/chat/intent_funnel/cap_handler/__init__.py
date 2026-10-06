"""cap_handler — per-capability parameter handlers (Intent Funnel).

The public boundary of the cap_handler package: the ``CapabilityHandler``
contract, the ``HANDLERS`` wiring map, the ``handler_for`` lookup, and each
concrete handler. Wiring lives in ``roster.py``; handlers live in their own
``*_handler.py`` modules. Nothing else is exported.
"""
from __future__ import annotations

from .create_folder_handler import CreateFolderHandler
from .pdf_extract_text_handler import PdfExtractTextHandler
from .pdf_table_to_text_handler import PdfTableToTextHandler
from .rag_search_handler import RagSearchHandler
from .read_document_handler import ReadDocumentHandler
from .read_file_handler import ReadFileHandler
from .roster import HANDLERS, CapabilityHandler, handler_for
from .social_search_handler import SocialSearchHandler
from .translate_handler import TranslateHandler
from .vision_handler import VisionHandler
from .web_search_handler import WebSearchHandler

__all__ = [
    "HANDLERS",
    "CapabilityHandler",
    "CreateFolderHandler",
    "PdfExtractTextHandler",
    "PdfTableToTextHandler",
    "RagSearchHandler",
    "ReadDocumentHandler",
    "ReadFileHandler",
    "SocialSearchHandler",
    "TranslateHandler",
    "VisionHandler",
    "WebSearchHandler",
    "handler_for",
]
