"""cap_handler roster — the ONE capability_id -> handler wiring point.

A capability with an entry here owns its own argument acquisition; a capability
with none keeps the generic acquisition chain untouched. This module maps and
nothing else: no tool-specific logic lives here, and it never executes a tool.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from .pdf_extract_text_handler import PdfExtractTextHandler
from .pdf_table_to_text_handler import PdfTableToTextHandler
from .rag_search_handler import RagSearchHandler
from .read_document_handler import ReadDocumentHandler
from .social_search_handler import SocialSearchHandler
from .vision_handler import VisionHandler
from .web_search_handler import WebSearchHandler


@runtime_checkable
class CapabilityHandler(Protocol):
    """The per-capability parameter handler contract.

    ``acquire`` returns the capability's final argument draft (``{slot: value}``,
    keys within the Registry schema), or ``None`` when it cannot produce a legal
    draft — the caller then exits to the Agent. A handler never executes a tool,
    never runs the runtime schema gate, and never re-implements a tool body.
    """

    capability_id: str

    async def acquire(self, *, query: str, facts) -> dict[str, object] | None: ...


# The wiring: only capabilities whose parameter acquisition is owned by a handler
# appear here. Everything else falls through to the generic acquisition chain.
HANDLERS: dict[str, CapabilityHandler] = {
    "cap-rag-search": RagSearchHandler(),
    "cap-web-search": WebSearchHandler(),
    "cap-social-search": SocialSearchHandler(),
    "cap-vision": VisionHandler(),
    "cap-read-document": ReadDocumentHandler(),
    "cap-pdf-extract-text": PdfExtractTextHandler(),
    "cap-pdf-table-to-text": PdfTableToTextHandler(),
}


def handler_for(capability_id: str) -> CapabilityHandler | None:
    """The handler owning ``capability_id``, or ``None`` (generic chain)."""
    return HANDLERS.get(capability_id)
