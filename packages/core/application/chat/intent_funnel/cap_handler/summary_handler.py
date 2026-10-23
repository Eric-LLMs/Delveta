"""SummaryHandler — argument acquisition for ``cap-summary``.

``cap-summary`` (tool binding ``summary_gen``) summarizes exactly ONE content
source. This handler decides WHICH source the sentence names, never what the
summary says, and never executes the tool.

The content source follows a FIXED priority ladder:

1. a clear, contract-compliant online-LLM verdict (``document`` whole / single
   / range, or ``conversation_context``) is executed EXACTLY as given — the
   fallback never overrides a recognisably stated intent;
2. the LLM returned normally but could not identify a clear intent
   (``ambiguous`` / ``unknown``): when the viewer carries BOTH a valid
   ``asset_id`` and a valid current page, summarise THAT document's CURRENT
   page. Never widen it to the whole document, never guess a page, never
   substitute another document;
3. no usable current document page (no asset id, or no valid current page):
   summarise the LAST non-empty assistant reply only — never the transcript;
4. neither is available: the Agent owns the clarification.

A SYSTEM fault in the extraction chain — call failure / timeout / service
unavailable, non-JSON or a contract-violating reply, or any actual extraction
exception — skips priorities 2 and 3 entirely and goes straight to the Agent
with the evidence: failure reason, category and stage, the original query, the
conversation context, the viewer's asset id and page, and the verdict already
obtained. Ordinary "the model answered, but the request is unclear" is NOT such
a fault.

The output directory is a MODEL-owned slot, extracted by the SAME online
argument extractor every MODEL/MIXED capability uses. ``slot_plan`` authorizes
that slot ONLY when the sentence explicitly asks to save (a destination cue),
and marks it REQUIRED for that turn: if the extractor is unavailable or returns
no valid directory, the acquisition hop exits to the Agent instead of silently
returning the summary in chat. With no save cue the slot is never requested, so
no path can be invented and ``summary_gen`` returns the body to the chat window
without writing anything. The output slot never changes the source priority
above.

One deterministic lane remains, because it is fully specified by the user's own
sentence: a literal workspace path ("总结 ./reports/q3.md") becomes ``path`` —
the single-path form of the tool's ``paths``, with no id invented. The
save/export cue authorizes the ``output_dir`` model slot but never supplies the
value itself.

``asset_id``/``pages`` still come from turn facts and the scope model. The
returned draft (or failure) passes through the SAME Binder / ActionExecutor /
ToolRuntime handoff every other capability uses.
"""
from __future__ import annotations

import logging

from ..argument_acquisition.page_scope_resolver import (
    SCOPE_UNKNOWN,
    STATUS_AMBIGUOUS,
    STATUS_INVALID,
    STATUS_UNAVAILABLE,
    TARGET_CONVERSATION,
    TARGET_DOCUMENT,
    TARGET_UNKNOWN,
)
from .acquisition_failure import (
    CATEGORY_EXTRACTION_FAILED,
    CATEGORY_NO_CONTENT,
    CATEGORY_SCOPE_CONTRACT,
    CATEGORY_SCOPE_UNSETTLED,
    STAGE_CONTENT_SELECTION,
    STAGE_SCOPE_RESOLUTION,
    AcquisitionFailure,
    turn_context,
)
from .path_token import has_destination_cue, literal_path
from .scope import resolve_asset_id
from .slot_plan import SlotPlan

logger = logging.getLogger(__name__)

# How much of the chat a conversation summary may see. The turn's history is the
# real chat (user + assistant); it is bounded so one long session cannot become
# the whole model input. Documents keep the toolkit's own big-document flow and
# are never bounded here.
MAX_CONTEXT_MESSAGES = 20
MAX_CONTEXT_CHARS = 12_000

# The fallback summarizes ONE previous assistant reply, never the transcript.
MAX_REPLY_CHARS = 12_000

_CONTEXT_ROLES = frozenset({"user", "assistant"})


def _fact(msg, name: str) -> str:
    return str((msg.get(name) if isinstance(msg, dict)
                else getattr(msg, name, "")) or "")


def current_page(facts) -> int | None:
    """The viewer's settled CURRENT page (1-based), or ``None``.

    Read from the turn-fact channel ONLY — never from a page number parsed out
    of the sentence — and a non-positive / non-numeric value is not a usable
    page (the fallback must never guess one).
    """
    raw = getattr(facts, "viewer_current_page", None) if facts is not None else None
    try:
        page = int(raw)
    except (TypeError, ValueError):
        return None
    return page if page > 0 else None


def conversation_transcript(history) -> str:
    """The bounded chat transcript (user + assistant) for this turn, or ``""``.

    Only the roles a conversation summary can legitimately contain are kept;
    system/agent-internal messages are dropped. The tail wins when the bound is
    exceeded, because "what we just discussed" is the recent part.
    """
    rows = [
        f"**{role}:** {content}"
        for role, content in (
            (_fact(m, "role"), _fact(m, "content").strip())
            for m in (history or ())
        )
        if role in _CONTEXT_ROLES and content
    ]
    text = "\n\n".join(rows[-MAX_CONTEXT_MESSAGES:])
    if len(text) > MAX_CONTEXT_CHARS:
        text = text[-MAX_CONTEXT_CHARS:]
    return text.strip()


def last_assistant_reply(history) -> str:
    """The most recent NON-EMPTY assistant message, or ``""``.

    This is the only legal fallback source when the request's own scope is
    ambiguous. User rows and internal messages are never used, and earlier
    assistant turns are not concatenated: the rule is "summarize that reply".
    """
    for msg in reversed(list(history or ())):
        if _fact(msg, "role") != "assistant":
            continue
        content = _fact(msg, "content").strip()
        if content:
            return content[:MAX_REPLY_CHARS]
    return ""


class SummaryHandler:
    """``cap-summary`` parameter handler (stateless)."""

    capability_id = "cap-summary"
    # The orchestrator passes the page-scope resolver into ``acquire``; only a
    # document-bearing / conversation request needs it (the literal path lane is
    # deterministic).
    accepts_page_scope = True

    async def acquire(self, *, query: str, facts,
                      page_scope_resolver=None, history=()
                      ) -> dict[str, object] | AcquisitionFailure:
        """Return the ``summary_gen`` content-source draft, or the structured
        failure the Agent receives when that source cannot be settled.

        ``output_dir`` is NOT produced here: it is the MODEL-owned slot the
        orchestrator extracts through ``slot_plan`` when (and only when) the
        user explicitly asked to save.
        """
        message = str(query or "").strip()
        history = list(history or ())

        if not message:
            return self._failure(
                CATEGORY_NO_CONTENT, STAGE_CONTENT_SELECTION,
                "the request is empty, so no content source can be selected",
                facts=facts, history=history)

        asset_id = resolve_asset_id(facts)
        # A literal workspace path is fully specified by the sentence; it is
        # used only when the turn carries no document identity.
        path = None if asset_id else literal_path(message)

        # ── Priority 1: the model decided the content source ──
        scope = None
        if page_scope_resolver is not None and (asset_id or path is None):
            scope = await page_scope_resolver(
                query=message, facts=facts, history=history,
                capability=self.capability_id, asset_id=asset_id)
            if scope.status == STATUS_INVALID:
                # A broken reply contract is a SYSTEM fault, not ordinary
                # ambiguity. It goes to the Agent with the exact validation
                # failure; the fallback ladder must not paper over it.
                logger.warning(
                    "summary scope reply violated the contract: %s", scope.reason)
                return self._failure(
                    CATEGORY_SCOPE_CONTRACT, STAGE_SCOPE_RESOLUTION,
                    ("the page-scope model reply violated its output contract "
                     f"({scope.reason or 'no reason given'}); no implicit "
                     "fallback was applied"),
                    facts=facts, history=history, scope=scope)
            if scope.status == STATUS_UNAVAILABLE:
                # The ONLINE parameter-extraction call itself failed (timeout /
                # connection / service / non-JSON). This is a SYSTEM fault, not
                # "the user was unclear": no page and no previous reply may be
                # substituted — the Agent owns the clarification.
                logger.warning(
                    "summary scope extraction unavailable: %s", scope.reason)
                return self._failure(
                    CATEGORY_EXTRACTION_FAILED, STAGE_SCOPE_RESOLUTION,
                    ("the online page-scope extraction could not serve "
                     f"({scope.reason or 'no reason given'}); no fallback was "
                     "applied"),
                    facts=facts, history=history, scope=scope)

        if path is not None:
            return {"path": path}

        if scope is not None and scope.ok:
            if scope.target == TARGET_DOCUMENT and asset_id:
                draft: dict[str, object] = {"asset_id": asset_id}
                if scope.pages is not None:
                    draft["pages"] = scope.pages
                return draft
            if scope.target == TARGET_CONVERSATION:
                text = conversation_transcript(history)
                if text:
                    return {"context": text}
                return self._failure(
                    CATEGORY_NO_CONTENT, STAGE_CONTENT_SELECTION,
                    ("the model selected the conversation as the source, but "
                     "the turn carries no non-empty user/assistant message"),
                    facts=facts, history=history, scope=scope)
            # A document verdict without a document identity can never be
            # satisfied — fall through without inventing one.

        # ── Priority 2: the viewer's CURRENT page ──
        # The LLM returned normally but did not settle the intent (or no scope
        # model is wired). A document identity PLUS a settled current page is
        # the deterministic fallback: that page of THAT document, nothing else.
        page = current_page(facts)
        if asset_id and page is not None:
            logger.info(
                "summary scope not settled (status=%s); using the viewer's "
                "current page %d",
                scope.status if scope is not None else "no_resolver", page)
            return {"asset_id": asset_id, "pages": str(page)}

        # ── Priority 3: the last non-empty assistant reply ──
        reply = last_assistant_reply(history)
        if reply:
            logger.info(
                "summary scope not settled (status=%s); using the last assistant reply",
                scope.status if scope is not None else "no_resolver")
            return {"context": reply}

        # ── Priority 4: the Agent owns the clarification ──
        return self._failure(
            CATEGORY_SCOPE_UNSETTLED, STAGE_CONTENT_SELECTION,
            ("the summary scope could not be determined, no document page is "
             "in view, and the conversation has no non-empty assistant reply "
             "to summarize"),
            facts=facts, history=history, scope=scope)

    def slot_plan(self, *, query: str, facts, draft: dict) -> SlotPlan:
        """Authorize ``output_dir`` ONLY when the user explicitly asked to save.

        The model extracts the directory; the slot is REQUIRED for that turn so
        a failed or empty extraction exits to the Agent instead of silently
        dropping the requested output mode. With no save cue the slot is never
        requested — the summary returns to the chat window and writes nothing.
        """
        if not has_destination_cue(str(query or "")):
            return SlotPlan()
        return SlotPlan(model_slots=("output_dir",),
                        required_model_slots=("output_dir",))

    # ── failure ──
    @classmethod
    def _failure(cls, category: str, stage: str, detail: str, *,
                 facts, history, scope=None) -> AcquisitionFailure:
        return AcquisitionFailure(
            category=category,
            stage=stage,
            detail=detail,
            scope=cls._scope_record(scope),
            context=turn_context(facts, history),
        )

    @staticmethod
    def _scope_record(scope) -> dict[str, object]:
        if scope is None:
            return {
                "status": "not_available",
                "target": TARGET_UNKNOWN,
                "page_scope": SCOPE_UNKNOWN,
                "pages": None,
                "reason": "the page-scope resolver is not wired for this turn",
            }
        return {
            "status": str(scope.status or STATUS_AMBIGUOUS),
            "target": str(scope.target or TARGET_UNKNOWN),
            "page_scope": str(scope.page_scope or SCOPE_UNKNOWN),
            "pages": scope.pages,
            "reason": str(scope.reason or ""),
        }
