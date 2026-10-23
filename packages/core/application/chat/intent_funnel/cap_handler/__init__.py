"""cap_handler — per-capability parameter handlers (Intent Funnel).

The public boundary of the cap_handler package: the ``CapabilityHandler``
contract, its structured ``AcquisitionFailure`` exit, the ``HANDLERS`` wiring
map, the ``handler_for`` lookup, and each concrete handler. Wiring lives in
``roster.py``; handlers live in their own ``*_handler.py`` modules. Nothing
else is exported.
"""
from __future__ import annotations

from .acquisition_failure import (
    CATEGORY_EXTRACTION_FAILED,
    CATEGORY_SCOPE_UNSETTLED,
    STAGE_ARGUMENT_EXTRACTION,
    STAGE_SCOPE_RESOLUTION,
    AcquisitionFailure,
    turn_context,
)
from .acquisition_policy import (
    POLICIES,
    STRATEGIES,
    STRATEGY_DEFERRED_AGENT,
    STRATEGY_MIXED,
    STRATEGY_MODEL,
    STRATEGY_NO_ARGS,
    STRATEGY_RULE_CONTEXT,
    AcquisitionPolicy,
    policy_for,
)
from .add_term_handler import AddTermHandler
from .create_folder_handler import CreateFolderHandler
from .pdf_extract_text_handler import PdfExtractTextHandler
from .pdf_table_to_text_handler import PdfTableToTextHandler
from .rag_search_handler import RagSearchHandler
from .read_document_handler import ReadDocumentHandler
from .read_file_handler import ReadFileHandler
from .roster import HANDLERS, CapabilityHandler, handler_for
from .slot_plan import SlotPlan, SlotPlanningHandler
from .social_search_handler import SocialSearchHandler
from .summary_handler import SummaryHandler
from .translate_handler import TranslateHandler
from .vision_handler import VisionHandler
from .web_search_handler import WebSearchHandler

__all__ = [
    "CATEGORY_EXTRACTION_FAILED",
    "CATEGORY_SCOPE_UNSETTLED",
    "HANDLERS",
    "POLICIES",
    "STAGE_ARGUMENT_EXTRACTION",
    "STAGE_SCOPE_RESOLUTION",
    "STRATEGIES",
    "STRATEGY_DEFERRED_AGENT",
    "STRATEGY_MIXED",
    "STRATEGY_MODEL",
    "STRATEGY_NO_ARGS",
    "STRATEGY_RULE_CONTEXT",
    "AcquisitionFailure",
    "AcquisitionPolicy",
    "AddTermHandler",
    "CapabilityHandler",
    "CreateFolderHandler",
    "PdfExtractTextHandler",
    "PdfTableToTextHandler",
    "RagSearchHandler",
    "ReadDocumentHandler",
    "ReadFileHandler",
    "SlotPlan",
    "SlotPlanningHandler",
    "SocialSearchHandler",
    "SummaryHandler",
    "TranslateHandler",
    "VisionHandler",
    "WebSearchHandler",
    "handler_for",
    "policy_for",
    "turn_context",
]
