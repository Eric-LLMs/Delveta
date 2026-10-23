"""ReadDocumentHandler — argument acquisition for ``cap-read-document``.

Owns ONLY this capability's parameter preparation; it never loads document bytes
and never parses a PDF/slide — the real file read and extraction stay entirely
inside the shared ``read_document`` tool. The two schema slots of
``cap-read-document``:

* ``asset_id`` — the ONLY legal source is :class:`~..contract.TurnFacts`
  (precedence ``attachment_asset_id`` -> ``path_asset_id`` -> ``viewer_asset_id``).
  There is NO fail-open path: when no fact carries an asset id the handler returns
  ``None`` and the turn exits to the Agent's existing missing/clarify handling —
  never a guessed or fabricated id.
* ``pages`` — acquired by the COMBINED lane
  (:mod:`..argument_acquisition.page_scope_settlement`): the deterministic rule
  runs first over the sentence and the trusted turn facts, then the online LLM
  supplements/corrects it, and the approved merge priority picks the value
  (rule-only -> rule, LLM-only -> LLM, conflict -> LLM, neither -> terminal).
  ``all`` omits the slot (whole document). A whole-document request is NEVER
  turned into the viewer's current page merely because a viewer page exists.

An unsettled scope returns a TERMINAL :class:`AcquisitionFailure`: the Agent must
not be asked to guess which pages the user meant.

Note: there is NO ``query`` / ``question`` slot on this capability — neither the
Registry schema nor the ``read_document`` tool declares one, so none is emitted
(an unknown slot would fail the Binder as ``BIND_INVALID``). The user's question
is not an argument here: the tool reads the document, and the turn's own LLM
composes the answer from the extracted text.

The returned draft (``{slot: value}``) is passed through the SAME Binder /
ActionExecutor / ToolRuntime handoff every other capability uses.
"""
from __future__ import annotations

import logging

from ..argument_acquisition.page_scope_settlement import (
    settle_page_scope,
    unsettled_scope,
)
from .acquisition_failure import (
    CATEGORY_SCOPE_UNSETTLED,
    STAGE_SCOPE_RESOLUTION,
    AcquisitionFailure,
    turn_context,
)
from .scope import resolve_asset_id

logger = logging.getLogger(__name__)


class ReadDocumentHandler:
    """``cap-read-document`` parameter handler (stateless)."""

    capability_id = "cap-read-document"
    # The orchestrator may pass the page-scope resolver into ``acquire`` (the
    # online-LLM half of the combined lane).
    accepts_page_scope = True
    # The orchestrator passes a dict for the rule / LLM / merged evidence.
    accepts_scope_evidence = True

    async def acquire(self, *, query: str, facts, page_scope_resolver=None,
                      history=(), scope_evidence: dict | None = None):
        """Return the ``read_document`` argument draft, or ``None`` when no
        document is in context, or a TERMINAL failure when the page scope cannot
        be settled by the rule and the online LLM together."""
        asset_id = resolve_asset_id(facts)
        if not asset_id:
            return None
        settlement = await settle_page_scope(
            query=query, facts=facts, history=history,
            resolver=page_scope_resolver, capability=self.capability_id,
            asset_id=asset_id)
        if scope_evidence is not None:
            scope_evidence.update(settlement.record())
        if not settlement.ok:
            logger.info(
                "read_document page scope unsettled rule=%s llm=%s detail=%s",
                settlement.rule.get("decision"), settlement.llm.get("status"),
                settlement.detail)
            return AcquisitionFailure(
                category=CATEGORY_SCOPE_UNSETTLED,
                stage=STAGE_SCOPE_RESOLUTION,
                detail=settlement.detail,
                scope=unsettled_scope(settlement),
                context=turn_context(facts, history),
                terminal=True,
            )
        draft: dict[str, object] = {"asset_id": asset_id}
        if settlement.pages is not None:
            draft["pages"] = settlement.pages
        return draft
