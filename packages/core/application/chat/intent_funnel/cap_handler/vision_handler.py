"""VisionHandler — argument acquisition for ``cap-vision``.

Owns ONLY this capability's parameter preparation; it never loads image bytes
and never re-implements any recognition logic — the real asset load and the
multimodal call stay entirely inside the shared ``vision`` tool. The schema
slots of ``cap-vision``:

* ``asset_id`` — the ONLY legal source is :class:`~..contract.TurnFacts`. The id
  is resolved with the SAME precedence the Binder uses
  (``attachment_asset_id`` -> ``path_asset_id`` -> ``viewer_asset_id``); a
  sentence-copied or model-invented id is never a source. There is NO
  fail-open path: when no fact carries an asset id the handler returns ``None``
  and the funnel first asks :meth:`capture_for` whether the PC client can supply
  the picture (local page/frame); only when that also abstains does the turn exit
  to the Agent's missing/clarify handling — never a guessed id, never a
  fabricated path, never an empty string.
* ``question`` — the user's sentence, taken VERBATIM (strip only), so the
  question they ask ABOUT the image reaches the model unaltered. Omitted when
  the message is empty (the tool then runs its own default analysis prompt).
* ``page`` — the page axis, settled by the ONLINE MODEL
  (:mod:`..argument_acquisition.page_scope_settlement`) exactly like the three
  file-content handlers' ``pages`` slot: the model reads the sentence and
  returns one page, the whole document, or a range. The tool renders exactly ONE
  page, so a single-page verdict becomes the integer ``page``; ``all`` omits the
  slot and the tool applies its own documented default (the page open in the
  viewer); a multi-page verdict goes to the Agent rather than silently picking
  one of the pages. An unsettled verdict never invents a page.

  The page axis exists only for a PAGINATED target: ``page`` renders a PDF page.
  When the resolved asset IS the viewer's and the viewer kind is known to be
  non-paginated, the model is not consulted — there is nothing for it to settle.
  Every other case (an attachment / drive-path asset, whose kind this turn does
  not declare, or an unknown viewer kind) keeps the model verdict, so a page the
  user DID name is never silently dropped. This is a structural target-type
  check, never a deterministic reading of the sentence.

The returned draft (``{slot: value}``) is passed through the SAME Binder /
ActionExecutor / ToolRuntime handoff every other capability uses — the Binder
re-resolves ``asset_id`` from the same facts, so its value is identical here.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.infrastructure.pdf_pages import parse_pages_spec

from ..argument_acquisition.page_scope_settlement import (
    SOURCE_NONE,
    settle_page_scope,
    unsettled_scope,
)
from .acquisition_failure import (
    CATEGORY_SCOPE_UNSETTLED,
    STAGE_SCOPE_RESOLUTION,
    AcquisitionFailure,
    turn_context,
)

# Mirrors binder._CONTEXT_SLOT_SOURCES["asset_id"] exactly: attachment first,
# then the drive path, then the on-screen viewer. Keeping the two in lockstep
# means the draft this handler emits is the value the Binder settles on.
_ASSET_FACT_FIELDS = ("attachment_asset_id", "path_asset_id", "viewer_asset_id")

# Document viewers whose current page the PC client can actually screenshot. The desktop
# renders a page canvas for PDF only; local office/text viewers have no page image to
# capture, so asking for one would dead-end every retry (they exit to the Agent instead).
_CAPTURE_DOC_KINDS = frozenset({"pdf"})

# Viewer kinds with NO page axis: the vision tool's ``page`` renders a PDF page,
# so a 1-based page cannot be named or read for these targets. ``office`` is
# included because the desktop viewer exposes no page image for it either.
_NON_PAGINATED_VIEWER_KINDS = frozenset({"image", "video", "text", "markdown", "office"})


@dataclass(frozen=True)
class DeferredCapture:
    """A picture the turn still owes: the client must capture it before execution.

    Emitted ONLY after the funnel has already decided ``vision`` is this turn's tool
    (the handler never fabricates a capture request for an ordinary chat turn). The
    orchestrator pauses the stream, the PC client captures/uploads, and the SAME
    certified turn resumes with ``asset_id`` filled in — no second funnel run.
    """

    kind: str                       # "page" | "frame" | "image"
    name: str = ""
    page: int | None = None
    t_ms: int | None = None
    duration_ms: int | None = None
    question: str = ""

    def request(self) -> dict:
        """Wire shape of the ``capture-request`` SSE frame (client-facing spec)."""
        return {
            "kind": self.kind,
            "name": self.name,
            "page": self.page,
            "t_ms": self.t_ms,
            "duration_ms": self.duration_ms,
            "question": self.question,
        }


def _resolve_asset_id(facts) -> str:
    """The first non-empty asset id among the context facts, else ``""``."""
    if facts is None:
        return ""
    for field in _ASSET_FACT_FIELDS:
        value = str(getattr(facts, field, "") or "").strip()
        if value:
            return value
    return ""


def _page_axis_applies(facts, asset_id: str) -> bool:
    """Whether a page number can mean anything for THIS turn's resolved asset.

    Structural only — the page SEMANTICS stay with the online model. The page
    axis is known to be absent only when the resolved asset IS the viewer's and
    the viewer's kind is known to be non-paginated. Every other case keeps the
    model verdict: an attachment / drive-path asset declares no kind on this
    turn, and an unknown viewer kind may be a PDF.
    """
    if facts is None:
        return False
    viewer_asset = str(getattr(facts, "viewer_asset_id", "") or "")
    if not viewer_asset or asset_id != viewer_asset:
        return True
    kind = str(getattr(facts, "viewer_kind", "") or "").strip().lower()
    return kind not in _NON_PAGINATED_VIEWER_KINDS


class VisionHandler:
    """``cap-vision`` parameter handler (stateless)."""

    capability_id = "cap-vision"
    # The orchestrator passes the online page-scope resolver into ``acquire``.
    accepts_page_scope = True
    # The orchestrator passes a dict for the model / merged evidence.
    accepts_scope_evidence = True

    async def acquire(self, *, query: str, facts, page_scope_resolver=None,
                      history=(), scope_evidence: dict | None = None
                      ) -> dict[str, object] | AcquisitionFailure | None:
        """Return the ``vision`` argument draft, or ``None`` when no image is in
        context (fail-closed — the Agent owns the clarification)."""
        asset_id = _resolve_asset_id(facts)
        if not asset_id:
            return None
        message = str(query or "").strip()

        settlement = None
        if _page_axis_applies(facts, asset_id):
            settlement = await settle_page_scope(
                query=message, facts=facts, history=history,
                resolver=page_scope_resolver, capability=self.capability_id,
                asset_id=asset_id)
            if scope_evidence is not None:
                scope_evidence.update(settlement.record())
        elif scope_evidence is not None:
            # Telemetry only: record that the page-scope model was deliberately
            # not consulted because this target has no page axis.
            scope_evidence.update({
                "ok": True, "source": SOURCE_NONE,
                "detail": ("the viewer is not a paginated document, so the "
                           "page axis does not apply; the page-scope model "
                           "was not consulted"),
                "llm": {"status": "not_called"},
                "merged": {"pages": None},
            })

        draft: dict[str, object] = {"asset_id": asset_id}
        if message:
            draft["question"] = message
        page = self._settled_page(settlement, facts=facts, history=history)
        if isinstance(page, AcquisitionFailure):
            return page
        if page is not None:
            draft["page"] = page
        return draft

    @staticmethod
    def _settled_page(settlement, *, facts, history
                      ) -> str | AcquisitionFailure | None:
        """The ONE page the model settled, as the funnel's string slot value, or
        ``None`` for the tool's default.

        ``all`` (and any unsettled verdict) omits ``page``: the tool's own
        documented rule then applies — the page open in the viewer, and a
        multi-page PDF with no page in view is refused rather than rendered as
        page 1. A range / list cannot be satisfied by a single-page tool call,
        so it goes to the Agent instead of silently choosing one page. A
        non-paginated target has no settlement at all (the model was not asked)
        and therefore also omits ``page``.
        """
        if settlement is None or not settlement.ok or settlement.pages is None:
            return None
        try:
            pages = parse_pages_spec(settlement.pages)
        except ValueError:                       # unreachable: the resolver validates
            return None
        if len(pages) == 1:
            # String, like every funnel slot: the Binder and ``validate_action``
            # normalise to str, and the vision tool coerces it to an int.
            return str(pages[0])
        return AcquisitionFailure(
            category=CATEGORY_SCOPE_UNSETTLED,
            stage=STAGE_SCOPE_RESOLUTION,
            detail=("the vision call reads exactly one page, but the sentence "
                    f"names pages {settlement.pages}"),
            scope=unsettled_scope(settlement),
            context=turn_context(facts, history),
        )

    def capture_for(self, *, query: str, facts) -> DeferredCapture | None:
        """The deferred capture this turn needs, or ``None`` when no capture applies.

        Called by the funnel ONLY after ``acquire()`` abstained, i.e. the turn is
        already certified as ``vision`` but no readable asset exists in context.
        A capture is legal only when the viewer itself is the picture source:

        * video — always (the vision tool cannot read a video file; the frame must
          come from the client's playhead). The timestamp is OPTIONAL here: the
          client resolves the live playhead when it answers the capture request,
          so no pre-sent viewer state is required;
        * local image — the file lives only on the PC, so its pixels must be sent;
        * local document (pdf/office/text/markdown) — the current page is rendered
          by the client into a screenshot.

        A cloud document without an asset id, a document with no page, or a turn
        with no viewer returns ``None``: the turn exits to the Agent exactly as
        before (never a capture with no source).
        """
        if facts is None or not facts.has_viewer:
            return None
        # An explicit image already in context (an attachment, a drive path, or an
        # already-uploaded frame/roi capture) IS the picture — never ask for another.
        if (str(getattr(facts, "attachment_asset_id", "") or "").strip()
                or str(getattr(facts, "path_asset_id", "") or "").strip()):
            return None
        message = str(query or "").strip()
        kind = str(getattr(facts, "viewer_kind", "") or "").lower()
        name = str(getattr(facts, "viewer_name", "") or "")
        if kind == "video":
            # The playhead is whatever the CLIENT sees when it answers this request —
            # it is never a prerequisite for deciding the capture (a deferred-data
            # client sends no timestamp at all; the client reports the live one).
            t_ms = getattr(facts, "viewer_t_ms", None)
            return DeferredCapture(
                kind="frame", name=name,
                t_ms=int(t_ms) if isinstance(t_ms, (int, float)) else None,
                duration_ms=getattr(facts, "viewer_duration_ms", None),
                question=message,
            )
        if str(getattr(facts, "viewer_provenance", "") or "").lower() != "local":
            return None
        if kind == "image":
            return DeferredCapture(kind="image", name=name, question=message)
        if kind in _CAPTURE_DOC_KINDS:
            page = getattr(facts, "viewer_current_page", None)
            return DeferredCapture(
                kind="page", name=name,
                page=int(page) if page is not None else None,
                question=message,
            )
        return None
