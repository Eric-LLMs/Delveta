"""Terminal acquisition failures end the turn honestly — never through the Agent.

Pins the three pieces of the terminal path:

* an unresolved page scope maps ``TurnRequirements.terminal_error`` to
  ``PlanKind.ERROR`` (the mapper's FIRST rule);
* the terminal executor answers with deterministic text, records the turn in
  session memory and runs no tool/LLM/Agent;
* the page handlers return a TERMINAL ``AcquisitionFailure`` when neither the
  rule nor the online LLM settles the pages.
"""
from __future__ import annotations

import types

import pytest

from core.application.chat.execution_plan import PlanKind, build_execution_plan
from core.application.chat.executors.base import TurnRequest
from core.application.chat.executors.terminal import TerminalExecutor
from core.application.chat.intent_funnel.cap_handler import (
    CATEGORY_SCOPE_UNSETTLED,
    pdf_extract_text_handler,
    pdf_table_to_text_handler,
    read_document_handler,
)
from core.application.chat.intent_funnel.cap_handler.acquisition_failure import (
    AcquisitionFailure,
)
from core.application.chat.understanding import TurnRequirements


def _terminal_requirements(**over):
    base = {
        "reason": "ACQUISITION_TERMINAL",
        "capability_id": "cap-read-document",
        "tool_binding": "read_document",
        "category": CATEGORY_SCOPE_UNSETTLED,
        "detail": "neither producer settled the page scope",
        "acquisition": {"category": CATEGORY_SCOPE_UNSETTLED,
                        "stage": "scope_resolution",
                        "detail": "neither producer settled the page scope",
                        "terminal": True},
    }
    base.update(over)
    return TurnRequirements(terminal_error=base)


class _Memory:
    def __init__(self):
        self.rows: list[tuple[str, str]] = []

    async def append_message(self, role: str, content: str):
        self.rows.append((role, content))


def _ctx():
    return types.SimpleNamespace(
        history=[], user_text="extract some pages", session_memory=_Memory())


def test_a_terminal_requirement_maps_to_the_error_plan_first():
    plan = build_execution_plan(_terminal_requirements())
    assert plan.kind is PlanKind.ERROR
    assert plan.action is None
    assert plan.error["capability_id"] == "cap-read-document"


def test_a_terminal_requirement_is_not_an_agent_plan():
    plan = build_execution_plan(_terminal_requirements())
    assert plan.kind is not PlanKind.AGENT


@pytest.mark.asyncio
async def test_the_terminal_executor_answers_honestly_and_records_the_turn():
    ctx = _ctx()
    plan = build_execution_plan(_terminal_requirements())
    result = await TerminalExecutor().run(TurnRequest(ctx=ctx, deps=None, plan=plan))

    assert "which pages" in result.final_answer
    assert result.usage == {}
    assert ctx.session_memory.rows == [
        ("user", "extract some pages"),
        ("assistant", result.final_answer),
    ]


def _facts():
    return types.SimpleNamespace(
        attachment_asset_id="", path_asset_id="", viewer_asset_id="asset-123",
        viewer_current_page=4, viewer_page_from=None, viewer_page_to=None,
        viewer_kind="pdf", viewer_name="survey.pdf",
        has_viewer_selection=False, has_viewer=True, has_attachment=False,
        has_turn_context=False,
    )


async def _ambiguous_resolver(**kwargs):
    from core.application.chat.intent_funnel.argument_acquisition import (
        page_scope_resolver as psr,
    )
    return psr.PageScope(status=psr.STATUS_AMBIGUOUS)


@pytest.mark.asyncio
@pytest.mark.parametrize("module,capability", [
    (read_document_handler, "cap-read-document"),
    (pdf_extract_text_handler, "cap-pdf-extract-text"),
])
async def test_every_shared_file_handler_returns_a_terminal_failure(module, capability):
    handler = module.__dict__[
        {"cap-read-document": "ReadDocumentHandler",
         "cap-pdf-extract-text": "PdfExtractTextHandler"}[capability]]()
    evidence: dict = {}

    out = await handler.acquire(
        query="extract some pages", facts=_facts(),
        page_scope_resolver=_ambiguous_resolver, history=(),
        scope_evidence=evidence)

    assert isinstance(out, AcquisitionFailure)
    assert out.terminal is True
    assert out.category == CATEGORY_SCOPE_UNSETTLED
    assert evidence["ok"] is False


@pytest.mark.asyncio
async def test_the_table_handler_never_terminates_it_settles_or_hands_off():
    # With no page named, the table policy reads the viewer's current page, so an
    # ambiguous model verdict is irrelevant: the turn settles, it never ends.
    handler = pdf_table_to_text_handler.PdfTableToTextHandler()
    out = await handler.acquire(
        query="extract some pages", facts=_facts(),
        page_scope_resolver=_ambiguous_resolver, history=())
    assert out == {"asset_id": "asset-123", "pages": "4"}

    # Without a valid current page it hands the turn to the Agent (non-terminal),
    # never to the terminal executor and never to a whole-document read.
    no_page = _facts()
    no_page.viewer_current_page = None
    out = await handler.acquire(
        query="extract some pages", facts=no_page,
        page_scope_resolver=_ambiguous_resolver, history=())
    assert isinstance(out, AcquisitionFailure)
    assert out.terminal is False
    assert out.category == CATEGORY_SCOPE_UNSETTLED


@pytest.mark.asyncio
async def test_a_whole_document_request_omits_pages_even_with_a_viewer_page():
    handler = read_document_handler.ReadDocumentHandler()

    async def _all(**kwargs):
        from core.application.chat.intent_funnel.argument_acquisition import (
            page_scope_resolver as psr,
        )
        return psr.PageScope(target=psr.TARGET_DOCUMENT,
                             page_scope=psr.SCOPE_ALL)

    out = await handler.acquire(query="summarise it", facts=_facts(),
                                page_scope_resolver=_all, history=())
    assert out == {"asset_id": "asset-123"}
