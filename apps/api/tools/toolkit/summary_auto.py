"""The two-stage ``auto`` summary flow used by ``summary_gen``.

The Intent Funnel certifies ``cap-summary`` with the ``auto`` marker (plus the
turn's ``asset_id`` when the Binder sourced one). THIS module owns the business
flow the marker selects:

**First call** — a small JSON decision over query + the previous chat message +
the open document's identity (never its body). It decides the target
(``previous_message`` / ``document`` / ``current_page`` / ``session``), whether
the answer is already producible from that material (``ready`` + ``summary``),
the page range when one is needed, and the SAVE intent (``save_to_file`` /
``save_path``). When ``ready`` is true the turn ends here — ONE model call. The
only exception is a live chat turn (a content sink is installed, no save): the
same concise summary is then produced by ONE streamed call over the very same
material, so the user watches it being written instead of receiving the whole
short paragraph at once. The summary's type and length do not change.

**Second call** — only when the first call resolved a target whose content it did
not have. The material is MINIMAL: a document target sends that document's
extracted page text only; a session target sends the ordered chat transcript
only. The save decision from the first call is carried through unchanged.

**Default ladder** — an unclear target (or a failed first call) summarises the
previous chat content when it exists, else the viewer's current page, else it
fails honestly. No path, page or content is ever invented.

Document text is extracted LOCALLY by the authorized reader; the page-scope and
argument extractors are never invoked on this lane.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from api.tools.drive_destination import (
    current_cwd,
    declared_error,
    resolve_destination,
    write_mime_type,
)
from api.tools.turn_material import (
    DOCUMENT_KINDS,
    ChatMessage,
    read_document_slice,
    recent_messages,
    turn_history,
    turn_query,
    viewer_info,
)
from core.application.drive_service import DriveService
from core.infrastructure.pdf_pages import parse_pages_spec
from core.infrastructure.request_context import (
    get_request_content_sink,
    get_request_user_id,
)

logger = logging.getLogger(__name__)

# How much of the chat a session summary may see (same bounds the previous
# handler used): the real user/assistant conversation, tail wins.
MAX_CONTEXT_MESSAGES = 20
MAX_CONTEXT_CHARS = 12_000

# One message included in the FIRST call is bounded; a pasted wall of text must
# not dominate the decision prompt.
MAX_PREVIEW_CHARS = 4_000

_FIRST_CALL_SYSTEM = """You are the planning step of a summarizer inside a
document/video reader and chat window.
Decide WHAT the user wants summarized and whether you can already produce it
from the material below. Reply with ONE JSON object and nothing else:
{
  "target": "previous_message" | "document" | "current_page" | "session" | "none",
  "ready": true | false,
  "summary": "<the summary, only when ready is true>",
  "asset_id": "<the open document's asset_id, or empty>",
  "pages": "<1-based page/slide spec: '3', '3-5', '2,5'; empty = the whole document>",
  "save_to_file": true | false,
  "save_path": "<the path the user literally wrote, or empty>",
  "reason": "<one short sentence>"
}
Rules:
- ready=true ONLY when the material below already contains everything needed
  (typically: the user wants the previous message summarized and it is shown).
  Otherwise ready=false and the content will be fetched in a second step.
- target=document when the user points at the open document; set pages to the
  range they named ("第3页到第5页" -> "3-5"). No page named -> empty (whole doc).
- target=current_page when they point at the page currently open.
- target=session when they point at the whole conversation / "前面几条".
- target=previous_message when they point at the last message / "上一条".
- target=none when nothing in the material can be summarized.
- save_to_file=true ONLY when the user explicitly asks to save/export a file.
  save_path is the exact path text they wrote; when they did not write one,
  leave it empty. NEVER invent a path.
- Use ONLY the supplied material; never invent document content."""

# The live lane's body call for the ``ready`` case: the SAME concise summary the
# first call would have carried in its ``summary`` field, written as plain text so
# it can stream. No structure, no headings — the ready summary was always a short
# paragraph, and it stays one.
_READY_SUMMARY_SYSTEM = """You are the summarizing step of a summarizer inside
a document/video reader and chat window.
Reply with ONLY the summary the user asked for, written from the material below.
Rules:
- Write in the user's language and match the scope of their request.
- Be concise: a short paragraph of roughly 3-6 sentences.
- Plain prose only: no title, no Markdown headings, no bullet lists, no JSON.
- End with the summary itself: no follow-up questions, no offers, no notes about
  how it was produced.
- Use ONLY the supplied material; never invent content."""


@dataclass(frozen=True)
class SummaryDecision:
    target: str = "none"
    ready: bool = False
    summary: str = ""
    asset_id: str = ""
    pages: str = ""
    save_to_file: bool = False
    save_path: str = ""
    reason: str = ""


def _text(value) -> str:
    return str(value or "").strip()


def _bounded(text: str, limit: int) -> str:
    text = str(text or "").strip()
    return text if len(text) <= limit else text[:limit] + "…"


def _normalise_path(raw: str) -> str:
    return str(raw or "").strip().rstrip("/\\")


def validate_save_path(raw: str, query: str) -> str:
    """Accept ``raw`` only when the user's own sentence contains it.

    The model may copy a path but may never invent one. A rejected path becomes
    the tool's default directory (still an explicit save), and the rejection is
    logged.
    """
    value = _normalise_path(raw)
    if not value:
        return ""
    haystack = _normalise_path(query)
    if value in haystack:
        return value
    logger.warning(
        "summary auto: dropping a save path the user did not write: %r", raw)
    return ""


def save_target(save_path: str) -> tuple[str, str]:
    """``(output_dir, stem)`` for a save path the user literally wrote.

    The toolkit's persist stage owns collision-proof naming (``<stem>_<stamp>.md``),
    so a FILE path keeps its parent as the directory and its name as the stem —
    it must never become a directory literally named ``q3.md``. A directory path
    (no suffix) is used as the directory. Both parts are copied from the user's
    own text; nothing is invented. ``("", "")`` = no explicit destination.
    """
    raw = _normalise_path(save_path)
    if not raw:
        return "", ""
    target = Path(raw)
    if target.suffix:
        parent = str(target.parent)
        return ("" if parent in ("", ".") else parent), (target.stem or "summary")
    return raw, ""


def _drive_target(save_path: str, stem: str) -> tuple[str, str]:
    """``(file name, parent folder)`` for a Cloud Drive save.

    The name the user wrote is used verbatim ("保存成 report.md" → ``report.md``);
    a path with no file name ("放到 notes 里") names the FOLDER and the summary
    keeps ``<stem>.md``. Nothing is invented when the user named nothing.
    """
    raw = _normalise_path(save_path)
    target = Path(raw)
    if raw and target.suffix and target.name:
        parent = str(target.parent)
        return target.name, ("" if parent in ("", ".") else parent)
    return f"{stem or 'summary'}.md", raw


async def _save_to_drive(ctx, text: str, save_path: str, *, stem: str) -> str | None:
    """Save the rendered summary into the turn's Cloud Drive, or ``None``.

    ``None`` means "not a Cloud turn" — the caller keeps the local workspace
    persist stage untouched. On a Cloud turn the save goes through the shared
    destination rule (a folder the user named > the working folder) and the
    drive service's own name/ACL checks; it never falls back to another
    directory, because a summary in the wrong folder is worse than no file.
    """
    if current_cwd() is None and not declared_error():
        return None
    uid = get_request_user_id()
    if uid is None:
        raise ValueError("saving into the Cloud Drive requires a signed-in user")
    drive = DriveService(ctx.resolve("session_factory"))
    name, parent = _drive_target(save_path, stem)
    destination = await resolve_destination(
        drive, uid, query=turn_query(), explicit=parent)
    if destination is None:  # defensive: the guard above already established a cwd
        raise ValueError("the Cloud Drive destination could not be resolved")
    asset = await drive.save_artifact(
        uid, name, write_mime_type(name), str(text or "").encode("utf-8"),
        folder_path=destination.folder_path or None,
        workspace_id=destination.workspace_id,
    )
    where = destination.join(asset.name)
    logger.info("summary auto: saved to the Cloud Drive at %s", where)
    return f"- {where}"


def parse_decision(data: dict, *, query: str, doc_asset: str,
                   doc_kind: str) -> SummaryDecision:
    """Validate the first call's reply into a decision (never trust blindly)."""
    if not isinstance(data, dict):
        return SummaryDecision(reason="the decision reply was not a JSON object")

    target = _text(data.get("target")).lower()
    if target not in {"previous_message", "document", "current_page",
                      "session", "none"}:
        target = "none"

    # A document identity may only be the one the turn already carries.
    asset = _text(data.get("asset_id"))
    if asset and asset != doc_asset:
        logger.warning(
            "summary auto: dropping an asset_id the turn does not carry: %r", asset)
        asset = ""
    if not asset and target in {"document", "current_page"}:
        asset = doc_asset if doc_kind in DOCUMENT_KINDS else ""

    pages = _text(data.get("pages"))
    if pages:
        try:
            parse_pages_spec(pages)
        except ValueError:
            logger.warning("summary auto: dropping an unparsable page spec: %r", pages)
            pages = ""

    summary = _text(data.get("summary"))
    ready = bool(data.get("ready")) and bool(summary)

    return SummaryDecision(
        target=target,
        ready=ready,
        summary=summary,
        asset_id=asset,
        pages=pages,
        save_to_file=bool(data.get("save_to_file")),
        save_path=validate_save_path(_text(data.get("save_path")), query),
        reason=_text(data.get("reason")),
    )


def _transcript(messages: tuple[ChatMessage, ...]) -> str:
    rows = [f"**{m.role}:** {m.content}" for m in messages]
    text = "\n\n".join(rows[-MAX_CONTEXT_MESSAGES:])
    if len(text) > MAX_CONTEXT_CHARS:
        text = text[-MAX_CONTEXT_CHARS:]
    return text.strip()


def _first_call_prompt(query: str, previous: ChatMessage | None, viewer,
                       doc_asset: str, *,
                       instruction: str = "Decide the summary target and the save intent.",
                       ) -> str:
    parts = [f"<user_request>\n{query}\n</user_request>"]
    if previous is not None:
        parts.append(
            f"<previous_message role=\"{previous.role}\">\n"
            f"{_bounded(previous.content, MAX_PREVIEW_CHARS)}\n</previous_message>")
    else:
        parts.append("<previous_message></previous_message>")
    if doc_asset:
        page = f' page="{viewer.page}"' if viewer.page is not None else ""
        parts.append(
            f'<open_document asset_id="{doc_asset}" name="{viewer.name}" '
            f'kind="{viewer.kind}"{page}/>')
    else:
        parts.append("<open_document/>")
    parts.append(
        f"<chat_rows>{len(turn_history())}</chat_rows>\n"
        f"{instruction}")
    return "\n\n".join(parts)


def _render_persisted(result) -> str:
    lines = [result.summary, *(f"- {f}" for f in result.files)]
    return "\n".join(lines)


async def _stream_ready_summary(*, query: str, previous: ChatMessage | None, viewer,
                                doc_asset: str, decision: SummaryDecision,
                                llm) -> str:
    """Write the ``ready`` summary as plain text, delta by delta, into the turn's sink.

    Same material the first call decided on, same concise style as the ``summary``
    field it carried — only the transport differs. The decided summary stays in hand
    as the fallback, so a stream that yields nothing never costs a broken turn and
    never triggers a second, fuller generation.
    """
    sink = get_request_content_sink()
    if sink is None:  # not a live turn (defensive): the decided body is the answer
        return decision.summary
    prompt = _first_call_prompt(
        query, previous, viewer, doc_asset,
        instruction="Write the summary the user asked for from the material above. "
                    "Reply with the summary only.")
    parts: list[str] = []
    started = False
    try:
        async for delta in llm.complete_stream(
                prompt, system_prompt=_READY_SUMMARY_SYSTEM):
            if not delta:
                continue
            if not started:
                if not delta.strip():
                    continue  # the joined frames start with real text
                started = True
            parts.append(delta)
            sink(delta)
    except Exception:  # the decided summary is a complete fallback
        logger.exception("summary auto: ready-summary stream failed")
        return "".join(parts) if parts else decision.summary
    text = "".join(parts).strip()
    if text:
        logger.info("summary auto: ready stream chars=%d deltas=%d",
                    len(text), len(parts))
        return text
    logger.warning("summary auto: ready-summary stream was empty; using the decided summary")
    return decision.summary


async def run_auto_summary(*, args: dict, ctx, llm, pipeline) -> str:
    """Run the first call, then only the material the second call actually needs."""
    query = turn_query().strip()
    if not query:
        raise ValueError("summary_gen auto mode needs the current user message")

    viewer = viewer_info()
    doc_asset = _text(args.get("asset_id")) or viewer.asset_id
    previous = recent_messages(1)
    previous = previous[-1] if previous else None

    prompt = _first_call_prompt(query, previous, viewer, doc_asset)
    try:
        data = await llm.complete_json_once(
            prompt, system_prompt=_FIRST_CALL_SYSTEM, max_tokens=900)
    except Exception as exc:  # noqa: BLE001 — honest degradation, never invent
        logger.warning("summary auto: first call failed (%s); using the default ladder",
                       exc)
        data = {}
    decision = parse_decision(data, query=query, doc_asset=doc_asset,
                              doc_kind=viewer.kind)

    logger.info(
        "summary auto: first_call target=%s ready=%s reason=%s save=%s path=%s",
        decision.target, decision.ready, decision.reason or "-",
        decision.save_to_file, decision.save_path or "-")

    # On a live chat turn the ready summary is WRITTEN as it streams: the same
    # concise summary over the same material, so the type and length do not change —
    # only the delivery does. Save turns and the non-streaming /chat lane keep the
    # single-call shortcut verbatim.
    streaming = (get_request_content_sink() is not None
                 and hasattr(llm, "complete_stream"))
    if decision.ready:
        if streaming and not decision.save_to_file:
            return await _stream_ready_summary(
                query=query, previous=previous, viewer=viewer,
                doc_asset=doc_asset, decision=decision, llm=llm)
        text = decision.summary
        if decision.save_to_file:
            saved = await _save_to_drive(ctx, text, decision.save_path, stem="summary")
            if saved is not None:
                return f"{text}\n{saved}"
            out_dir, named_stem = save_target(decision.save_path)
            result = await pipeline.persist_text(
                text, out_dir or None, stem=named_stem or "summary")
            return _render_persisted(result)
        return text

    # ── second call: exactly ONE source, chosen by the decision ──
    if decision.target in {"document", "current_page"}:
        if decision.asset_id:
            pages = decision.pages or (
                str(viewer.page) if decision.target == "current_page"
                and viewer.page is not None else "")
            return await _summarize_document(
                query=query, asset_id=decision.asset_id, pages_spec=pages,
                ctx=ctx, llm=llm, pipeline=pipeline, decision=decision)
        logger.info("summary auto: document target without an asset; using the ladder")

    if decision.target == "session":
        transcript = _transcript(turn_history())
        if transcript:
            return await _summarize_text(
                query=query, name="conversation", text=transcript,
                pipeline=pipeline, decision=decision, stem="conversation", ctx=ctx)
        logger.info("summary auto: session target with no chat rows; using the ladder")

    if decision.target == "previous_message" and previous is not None:
        return await _summarize_text(
            query=query, name="previous message", text=previous.content,
            pipeline=pipeline, decision=decision, stem="previous_message", ctx=ctx)

    # ── default ladder: previous content -> current page -> honest failure ──
    if previous is not None and previous.content.strip():
        logger.info("summary auto: unclear target; defaulting to the previous message")
        return await _summarize_text(
            query=query, name="previous message", text=previous.content,
            pipeline=pipeline, decision=decision, stem="previous_message", ctx=ctx)

    if doc_asset and viewer.page is not None:
        logger.info("summary auto: no previous content; defaulting to the current page")
        return await _summarize_document(
            query=query, asset_id=doc_asset, pages_spec=str(viewer.page),
            ctx=ctx, llm=llm, pipeline=pipeline, decision=decision)

    raise ValueError(
        "there is nothing to summarize: the conversation has no previous content "
        "and no document page is open")


async def _summarize_document(*, query, asset_id, pages_spec, ctx, llm,
                              pipeline, decision: SummaryDecision) -> str:
    pages = parse_pages_spec(pages_spec) if pages_spec else None
    try:
        document = await read_document_slice(
            ctx, llm, asset_id, pages=pages, caller="summary_gen")
    except ValueError as exc:
        raise ValueError(f"cannot read the document to summarize: {exc}") from None
    logger.info(
        "summary auto: second_call source=document name=%s pages=%s chars=%d "
        "truncated=%s", document.name, pages_spec or "all", len(document.text),
        document.truncated)
    return await _summarize_text(
        query=query, name=document.name, text=document.text,
        pipeline=pipeline, decision=decision,
        stem=(document.name.rsplit(".", 1)[0] or "summary"), ctx=ctx)


async def _summarize_text(*, query, name, text, pipeline, decision: SummaryDecision,
                          stem: str, ctx) -> str:
    from .sources import from_text

    body = f"Current summary request: {query}\n\n{text}"
    if decision.save_to_file and (
            current_cwd() is not None or declared_error()):
        # Cloud lane: render ONCE, then write the rendered body at the resolved
        # destination. The local persist stage is not used — it is confined to the
        # desktop workspace and would land the file in the wrong place. Rendering
        # must NOT stream into the chat: the turn's answer is the saved-file note.
        rendered = await pipeline.run_sources(
            [from_text(name, body)], None, persist=False, stem=stem,
            stream_text=False)
        saved = await _save_to_drive(ctx, rendered.text, decision.save_path, stem=stem)
        return f"{rendered.text}\n{saved}"

    out_dir, named_stem = save_target(decision.save_path)
    result = await pipeline.run_sources(
        [from_text(name, body)],
        out_dir or None,
        persist=decision.save_to_file,
        stem=(named_stem or stem) if decision.save_to_file else stem,
    )
    if decision.save_to_file:
        logger.info("summary auto: saved files=%s", result.files)
        return _render_persisted(result)
    return result.text
