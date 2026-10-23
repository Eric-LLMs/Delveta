"""cap_handler — :class:`SummaryHandler` (``cap-summary`` argument acquisition).

The handler decides WHICH one content source the turn names; it never reads the
document and never invents an argument. The business rules are pinned here as a
FIXED priority ladder:

1. a clear, contract-compliant verdict (document whole / single / range, or the
   conversation) is executed exactly as resolved — the fallback never overrides
   it;
2. the LLM returned normally but the intent is ambiguous / unknown: with BOTH a
   valid ``asset_id`` and a valid current page, the viewer's CURRENT PAGE of
   that document is summarized (never widened, never another document);
3. without a usable current document page, the LAST non-empty assistant reply
   is summarized (that one reply, never the transcript);
4. otherwise the Agent owns the clarification.

A SYSTEM fault — an unavailable call (failure / timeout / non-JSON) or a
contract-violating reply — skips 2 and 3 entirely and goes to the Agent.

Plus the deterministic lane that needs no model (a literal workspace path) and
the one MODEL-owned slot (``output_dir``), which ``slot_plan`` authorizes only
when the sentence explicitly asks to save — and marks REQUIRED for that turn.
"""
from __future__ import annotations

from core.application.chat.intent_funnel.argument_acquisition.page_scope_resolver import (
    SCOPE_ALL,
    STATUS_AMBIGUOUS,
    STATUS_INVALID,
    STATUS_UNAVAILABLE,
    TARGET_CONVERSATION,
    TARGET_DOCUMENT,
    TARGET_UNKNOWN,
    PageScope,
)
from core.application.chat.intent_funnel.cap_handler import (
    CATEGORY_EXTRACTION_FAILED,
    HANDLERS,
    SummaryHandler,
    handler_for,
)
from core.application.chat.intent_funnel.cap_handler.acquisition_failure import (
    CATEGORY_NO_CONTENT,
    CATEGORY_SCOPE_CONTRACT,
    CATEGORY_SCOPE_UNSETTLED,
    STAGE_CONTENT_SELECTION,
    STAGE_SCOPE_RESOLUTION,
    AcquisitionFailure,
)
from core.application.chat.intent_funnel.contract import TurnFacts

ASSET = "0ea50a94-f4f7-45eb-bbaa-43966d6af575"

HISTORY = [
    {"role": "user", "content": "我们刚才聊了混合检索"},
    {"role": "assistant", "content": "对，RRF 融合两路分数"},
    {"role": "system", "content": "internal routing note"},
]

NO_REPLY_HISTORY = [
    {"role": "user", "content": "我们刚才聊了混合检索"},
    {"role": "system", "content": "internal routing note"},
]


def _scope(*, target=TARGET_DOCUMENT, pages=None, status="ok",
           page_scope=SCOPE_ALL, reason="") -> PageScope:
    return PageScope(target=target, page_scope=page_scope, pages=pages,
                     status=status, reason=reason)


def _resolver(verdict: PageScope, *, seen: dict | None = None):
    async def _resolve(**kwargs):
        if seen is not None:
            seen.update(kwargs)
        return verdict
    return _resolve


def _facts(**kw) -> TurnFacts:
    return TurnFacts(**kw)


# ── clear intent: executed exactly as resolved ─────────────────────────────────


async def test_document_whole_is_the_asset_id_and_no_pages():
    handler = SummaryHandler()
    draft = await handler.acquire(
        query="总结这份文档", facts=_facts(viewer_asset_id=ASSET),
        page_scope_resolver=_resolver(_scope()))
    assert draft == {"asset_id": ASSET}          # no ``pages`` = whole document
    assert "paths" not in draft                  # the asset id is never disguised


async def test_document_single_page_uses_the_resolved_spec():
    handler = SummaryHandler()
    draft = await handler.acquire(
        query="总结第 5 页", facts=_facts(viewer_asset_id=ASSET),
        page_scope_resolver=_resolver(
            _scope(pages="5", page_scope="single")))
    assert draft == {"asset_id": ASSET, "pages": "5"}


async def test_document_range_uses_the_resolved_spec():
    handler = SummaryHandler()
    draft = await handler.acquire(
        query="总结第 3 页到第 5 页", facts=_facts(viewer_asset_id=ASSET),
        page_scope_resolver=_resolver(
            _scope(pages="3-5", page_scope="range")))
    assert draft == {"asset_id": ASSET, "pages": "3-5"}


async def test_a_clear_document_verdict_is_never_overridden_by_the_fallback():
    handler = SummaryHandler()
    draft = await handler.acquire(
        query="总结第 5 页", facts=_facts(viewer_asset_id=ASSET),
        page_scope_resolver=_resolver(
            _scope(pages="5", page_scope="single")),
        history=HISTORY)                          # a reply exists, but must lose
    assert draft == {"asset_id": ASSET, "pages": "5"}


async def test_resolver_seam_receives_the_trusted_asset_and_no_extra_identity():
    seen: dict = {}
    handler = SummaryHandler()
    await handler.acquire(
        query="总结一下", facts=_facts(attachment_asset_id=ASSET, viewer_name="a.pdf"),
        page_scope_resolver=_resolver(_scope(), seen=seen), history=HISTORY)
    assert seen["asset_id"] == ASSET
    assert seen["capability"] == "cap-summary"
    assert seen["history"] == HISTORY


async def test_conversation_target_while_a_document_is_open_uses_the_chat():
    handler = SummaryHandler()
    draft = await handler.acquire(
        query="总结我们刚才讨论的内容",
        facts=_facts(viewer_asset_id=ASSET),
        page_scope_resolver=_resolver(
            _scope(target=TARGET_CONVERSATION, page_scope="unknown")),
        history=HISTORY)
    assert "asset_id" not in draft               # the open document is NOT the source
    assert "我们刚才聊了混合检索" in draft["context"]
    assert "RRF 融合两路分数" in draft["context"]
    assert "internal routing note" not in draft["context"]   # system rows are dropped


# ── priority 2: ambiguous intent, a settled document page is in view ──────────


async def test_an_ambiguous_scope_summarizes_the_current_page_of_the_document():
    handler = SummaryHandler()
    draft = await handler.acquire(
        query="总结一下",
        facts=_facts(viewer_asset_id=ASSET, viewer_current_page=7),
        page_scope_resolver=_resolver(
            _scope(status=STATUS_AMBIGUOUS, target=TARGET_UNKNOWN,
                   page_scope="unknown")),
        history=HISTORY)
    # Exactly the current page of THAT document: not the whole document, not a
    # guessed page, and not the previous assistant reply.
    assert draft == {"asset_id": ASSET, "pages": "7"}


async def test_the_current_page_fallback_also_applies_without_a_scope_model():
    handler = SummaryHandler()
    draft = await handler.acquire(
        query="总结一下", facts=_facts(viewer_asset_id=ASSET,
                                      viewer_current_page=3))
    assert draft == {"asset_id": ASSET, "pages": "3"}


async def test_an_invalid_current_page_never_produces_a_page():
    handler = SummaryHandler()
    draft = await handler.acquire(
        query="总结一下",
        facts=_facts(attachment_asset_id=ASSET, viewer_current_page=0),
        page_scope_resolver=_resolver(
            _scope(status=STATUS_AMBIGUOUS, target=TARGET_UNKNOWN,
                   page_scope="unknown")),
        history=HISTORY)
    assert draft == {"context": "对，RRF 融合两路分数"}   # priority 3 instead


# ── priority 3: no usable current document page -> the last assistant reply ────


async def test_an_ambiguous_scope_without_a_page_uses_only_the_last_assistant_reply():
    handler = SummaryHandler()
    draft = await handler.acquire(
        query="总结一下", facts=_facts(viewer_asset_id=ASSET),
        page_scope_resolver=_resolver(
            _scope(status=STATUS_AMBIGUOUS, target=TARGET_UNKNOWN,
                   page_scope="unknown")),
        history=HISTORY)
    assert draft == {"context": "对，RRF 融合两路分数"}
    assert "我们刚才聊了混合检索" not in draft["context"]     # not the transcript
    assert "internal routing note" not in draft["context"]


# ── unavailable call: an EXTRACTION failure, never the reply fallback ──────────


async def test_an_unavailable_call_goes_to_the_agent_even_with_a_reply():
    # The ONLINE parameter extraction failed. The business rule distinguishes
    # this from "the model answered, but the intent is ambiguous": no page and no
    # previous reply may be substituted.
    handler = SummaryHandler()
    failure = await handler.acquire(
        query="总结一下",
        facts=_facts(viewer_asset_id=ASSET, viewer_current_page=7,
                     viewer_kind="pdf", viewer_name="a.pdf"),
        page_scope_resolver=_resolver(
            _scope(status=STATUS_UNAVAILABLE,
                   reason="LLM call timed out after 20s")),
        history=HISTORY)
    assert isinstance(failure, AcquisitionFailure)     # NOT the assistant reply
    assert failure.category == CATEGORY_EXTRACTION_FAILED
    assert failure.stage == STAGE_SCOPE_RESOLUTION
    assert "LLM call timed out after 20s" in failure.detail
    assert "no fallback" in failure.detail
    assert failure.scope["status"] == STATUS_UNAVAILABLE
    assert failure.context["viewer_asset_id"] == ASSET
    assert failure.context["viewer_current_page"] == 7
    # The Agent learns a reply existed and was deliberately NOT used.
    assert failure.context["has_last_assistant_reply"] is True


async def test_a_document_turn_without_the_resolver_uses_the_last_reply():
    handler = SummaryHandler()
    draft = await handler.acquire(
        query="总结这份文档", facts=_facts(viewer_asset_id=ASSET), history=HISTORY)
    assert draft == {"context": "对，RRF 融合两路分数"}


async def test_no_page_and_no_reply_goes_to_the_agent():
    # Priority 4: neither a usable current document page nor a previous reply.
    handler = SummaryHandler()
    failure = await handler.acquire(
        query="总结一下",
        facts=_facts(viewer_asset_id=ASSET, viewer_kind="pdf",
                     viewer_name="a.pdf"),
        page_scope_resolver=_resolver(
            _scope(status=STATUS_AMBIGUOUS, target=TARGET_UNKNOWN,
                   page_scope="unknown", reason="target not determined")),
        history=NO_REPLY_HISTORY)
    assert isinstance(failure, AcquisitionFailure)
    assert failure.category == CATEGORY_SCOPE_UNSETTLED
    assert failure.stage == STAGE_CONTENT_SELECTION
    # The resolver's own verdict travels with the failure.
    assert failure.scope == {
        "status": STATUS_AMBIGUOUS, "target": TARGET_UNKNOWN,
        "page_scope": "unknown", "pages": None,
        "reason": "target not determined",
    }
    # And so do the trusted viewer identity / page.
    assert failure.context["viewer_asset_id"] == ASSET
    assert failure.context["viewer_current_page"] is None
    assert failure.context["viewer_kind"] == "pdf"
    assert failure.context["has_last_assistant_reply"] is False


async def test_a_missing_resolver_without_a_reply_goes_to_the_agent():
    handler = SummaryHandler()
    failure = await handler.acquire(
        query="总结一下", facts=_facts(viewer_asset_id=ASSET), history=NO_REPLY_HISTORY)
    assert isinstance(failure, AcquisitionFailure)
    assert failure.category == CATEGORY_SCOPE_UNSETTLED
    assert failure.scope["status"] == "not_available"
    assert failure.context["has_last_assistant_reply"] is False


# ── contract violation: straight to the Agent, no implicit fallback ────────────


async def test_a_contract_violating_reply_goes_to_the_agent_even_with_a_reply():
    handler = SummaryHandler()
    failure = await handler.acquire(
        query="总结第 5 页", facts=_facts(viewer_asset_id=ASSET),
        page_scope_resolver=_resolver(
            _scope(status=STATUS_INVALID,
                   reason="page_scope=all must not carry pages")),
        history=HISTORY)
    assert isinstance(failure, AcquisitionFailure)     # NOT the assistant reply
    assert failure.category == CATEGORY_SCOPE_CONTRACT
    assert failure.stage == STAGE_SCOPE_RESOLUTION
    assert "page_scope=all must not carry pages" in failure.detail
    assert "no implicit fallback" in failure.detail


async def test_the_contract_failure_carries_the_request_verdict_and_viewer():
    handler = SummaryHandler()
    failure = await handler.acquire(
        query="总结第 5 页",
        facts=_facts(viewer_asset_id=ASSET, viewer_current_page=7,
                     viewer_page_from=5, viewer_page_to=9),
        page_scope_resolver=_resolver(
            _scope(status=STATUS_INVALID, target=TARGET_DOCUMENT,
                   page_scope="range", pages="9-5",
                   reason="pages must ascend")),
        history=HISTORY)
    assert failure.scope["status"] == STATUS_INVALID
    assert failure.scope["pages"] == "9-5"
    assert failure.scope["reason"] == "pages must ascend"
    assert failure.context["viewer_asset_id"] == ASSET
    assert failure.context["viewer_current_page"] == 7
    assert failure.context["viewer_page_from"] == 5
    assert failure.context["viewer_page_to"] == 9
    # The Agent also learns the fallback was available but deliberately unused.
    assert failure.context["has_last_assistant_reply"] is True
    assert failure.context["history_messages"] == len(HISTORY)


# ── settled scope, empty content ──────────────────────────────────────────────


async def test_conversation_target_with_an_empty_history_is_not_fabricated():
    handler = SummaryHandler()
    failure = await handler.acquire(
        query="总结我们刚才讨论的内容", facts=_facts(viewer_asset_id=ASSET),
        page_scope_resolver=_resolver(
            _scope(target=TARGET_CONVERSATION, page_scope="unknown")),
        history=[])
    assert isinstance(failure, AcquisitionFailure)
    assert failure.category == CATEGORY_NO_CONTENT
    assert failure.stage == STAGE_CONTENT_SELECTION
    assert failure.scope["target"] == TARGET_CONVERSATION


async def test_an_empty_request_is_a_no_content_failure():
    handler = SummaryHandler()
    failure = await handler.acquire(query="   ", facts=None)
    assert isinstance(failure, AcquisitionFailure)
    assert failure.category == CATEGORY_NO_CONTENT


# ── workspace-file lane: the literal token is the only source ───────────────────


async def test_literal_workspace_path_becomes_the_single_path_slot():
    handler = SummaryHandler()
    draft = await handler.acquire(query="总结 ./reports/q3.md", facts=None)
    assert draft == {"path": "./reports/q3.md"}
    assert "paths" not in draft and "asset_id" not in draft


# ── output_dir: a MODEL-owned slot, authorized only on an explicit save cue ────


def test_no_save_cue_authorizes_no_model_slot():
    # Without an explicit request to save, nothing may be written: the slot is
    # never extracted, so no path can be invented.
    plan = SummaryHandler().slot_plan(
        query="总结 ./reports/q3.md", facts=None, draft={"path": "./reports/q3.md"})
    assert plan.model_slots == ()
    assert plan.required_model_slots == ()


def test_a_save_cue_authorizes_a_required_output_dir():
    plan = SummaryHandler().slot_plan(
        query="总结 ./reports/q3.md 并保存到 outputs", facts=None,
        draft={"path": "./reports/q3.md"})
    assert plan.model_slots == ("output_dir",)
    # REQUIRED: if the extractor cannot deliver it, the hop exits to the Agent
    # instead of silently returning the summary in chat.
    assert plan.required_model_slots == ("output_dir",)


def test_an_english_save_cue_also_authorizes_the_slot():
    plan = SummaryHandler().slot_plan(
        query="summarize ./reports/q3.md and save it to outputs", facts=None,
        draft={"path": "./reports/q3.md"})
    assert plan.model_slots == ("output_dir",)
    assert plan.required_model_slots == ("output_dir",)


async def test_the_handler_draft_never_contains_output_dir():
    # The directory comes from the shared extractor, never from handler-local
    # text parsing — even when the sentence names one.
    handler = SummaryHandler()
    draft = await handler.acquire(
        query="总结 ./reports/q3.md 并保存到 outputs", facts=None)
    assert draft == {"path": "./reports/q3.md"}


# ── no document, no path: the conversation lane and its fallback ───────────────


async def test_conversation_without_a_document_uses_the_history():
    handler = SummaryHandler()
    draft = await handler.acquire(
        query="总结我们刚才讨论的内容", facts=None,
        page_scope_resolver=_resolver(
            _scope(target=TARGET_CONVERSATION, page_scope="unknown")),
        history=HISTORY)
    assert set(draft) == {"context"}
    assert "我们刚才聊了混合检索" in draft["context"]


async def test_a_document_verdict_without_a_document_never_fabricates_one():
    seen: dict = {}
    handler = SummaryHandler()
    draft = await handler.acquire(
        query="总结一下", facts=None,
        page_scope_resolver=_resolver(_scope(), seen=seen), history=HISTORY)
    assert "asset_id" not in draft               # identity was never invented
    assert draft == {"context": "对，RRF 融合两路分数"}
    assert seen["asset_id"] == ""


async def test_no_source_and_no_resolver_falls_back_to_the_last_reply():
    handler = SummaryHandler()
    draft = await handler.acquire(query="总结一下", facts=None, history=HISTORY)
    assert draft == {"context": "对，RRF 融合两路分数"}


async def test_the_fallback_reply_draft_carries_no_output_dir():
    # The requested destination is owned by the model slot, so it never appears
    # in the handler draft — a missing extraction is an Agent handoff upstream,
    # never a silent chat-only result.
    handler = SummaryHandler()
    draft = await handler.acquire(
        query="总结一下并保存到 outputs", facts=None, history=HISTORY)
    assert draft == {"context": "对，RRF 融合两路分数"}


# ── roster wiring ───────────────────────────────────────────────────────────────


def test_roster_wires_cap_summary():
    handler = handler_for("cap-summary")
    assert isinstance(handler, SummaryHandler)
    assert handler.capability_id == "cap-summary"
    assert handler.accepts_page_scope is True
    assert HANDLERS["cap-summary"] is handler
