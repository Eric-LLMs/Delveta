"""``translate``: decide WHAT to translate, translate it, return only the text.

The Intent Funnel does NOT extract a payload with Qwen. Its handler sends the
user's sentence VERBATIM as ``text``; this tool then assembles the turn material
and makes ONE decision call:

* the sentence itself ("计算机技术英文怎么表达");
* a previous chat message ("上一条聊天英语什么意思") — the last 2 rows;
* the open document's current page — extracted locally, no extra model call;
* the current video — subtitle window + the captured frame (when the client
  supplied one through the deferred-capture lane).

The call returns ONLY the translation. The decision call is instructed to work
in two steps — translate the source material first, then re-read it in its
context (the surrounding page / subtitles / chat) and refine the wording — so
the situation shapes the result without being echoed back to the user.

An explicitly supplied payload (``text`` different from the user's current
message — the Agent lane and direct callers) keeps the original behaviour: it is
translated directly, with no source decision.
"""
from __future__ import annotations

import logging

from agent import Context, ToolExecution, ToolOutput, ToolRuntime, define_tool, text_block
from api.tools.turn_material import (
    DOCUMENT_KINDS,
    capture_image_data_url,
    read_document_slice,
    recent_messages,
    turn_query,
    viewer_info,
    video_subtitles,
)

logger = logging.getLogger(__name__)

# The confirmed default target when the request names none.
DEFAULT_TARGET_LANGUAGE = "English"

# One chat row is bounded so a pasted wall of text cannot dominate the call.
MAX_MESSAGE_CHARS = 4_000

_PLAIN_SYSTEM = (
    "You are a translator. Translate the text into natural, fluent {target}."
)

_PLAIN_FALLBACK_SYSTEM = (
    "You are a translation assistant inside a document/video reader. Decide what "
    "the user wants translated from the material, translate it, then refine the "
    "wording against its surrounding context. Answer with the translation only."
)

_DECISION_SYSTEM = """You are a translation assistant inside a document/video reader.
Decide WHAT the user wants translated and translate it. Reply with ONE JSON
object and nothing else:
{
  "source": "sentence" | "chat" | "document" | "video",
  "target_language": "<language name>",
  "translation": "<the translation only>",
  "note": "<only when the selected source is missing from the material>"
}
Rules:
- source=sentence when the user points at the sentence they just typed.
- source=chat when they point at a previous chat message.
- source=document when they point at the open document / its page.
- source=video when they point at the current video (screenshot and/or subtitles).
- target_language: the language the user explicitly asked for; when none is
  named use "English". Never invent a different target.
- Work in two steps: translate the selected material FIRST, then re-read it in
  its context (surrounding page / subtitles / chat) and refine the wording where
  the context changes it. Output ONLY the final translation: no explanation, no
  notes, no "Context:" section.
- Use ONLY the supplied material. When the selected source is absent, leave
  "translation" empty and put one short sentence in "note" instead of inventing
  content."""


def _bounded(text: str, limit: int = MAX_MESSAGE_CHARS) -> str:
    text = str(text or "").strip()
    return text if len(text) <= limit else text[:limit] + "…"


def _target_language(args: dict) -> str:
    # Bounded to a single line: the value rides into the system prompt, so a
    # stray newline from a caller cannot inject extra turns.
    raw = str(args.get("target_language") or "").splitlines()
    value = (raw[0].strip() if raw else "") or DEFAULT_TARGET_LANGUAGE
    return value


def _material_sections(*, query: str, prior, document, subtitles: str,
                       viewer, has_frame: bool) -> str:
    parts: list[str] = [f'<user_request>\n{query}\n</user_request>']
    if prior:
        rows = "\n".join(f"{m.role}: {_bounded(m.content)}" for m in prior)
        parts.append(f"<previous_chat>\n{rows}\n</previous_chat>")
    if document is not None:
        page = f" (page {viewer.page})" if viewer.page is not None else ""
        parts.append(
            f'<document name="{document.name}" asset_id="{document.asset_id}"{page}>\n'
            f"{document.text}\n</document>"
        )
    if subtitles:
        parts.append(f"<video_subtitles>\n{subtitles}\n</video_subtitles>")
    if has_frame:
        parts.append(
            "<video_frame>The current video screenshot is attached as an image."
            "</video_frame>")
    parts.append(
        "Decide the source from the user's request and the material above.")
    return "\n\n".join(parts)


def _render(translation: str, target: str) -> str:
    """The user-visible answer: the translation and nothing else."""
    body = str(translation or "").strip()
    return f"Translation ({target}):\n{body}" if target else body


async def _auto_translate(args: dict, *, llm, ctx) -> str:
    """The decision + translation path (ONE model call)."""
    query = turn_query().strip() or str(args.get("text") or "").strip()
    viewer = viewer_info()

    document = None
    frame_asset = (str(args.get("asset_id") or "").strip()
                   if viewer.kind == "video" else "")
    doc_asset = ""
    if viewer.asset_id and viewer.kind in DOCUMENT_KINDS:
        doc_asset = viewer.asset_id
    elif viewer.kind != "video":
        # An attachment / drive-path asset the Binder injected (no viewer doc).
        doc_asset = str(args.get("asset_id") or "").strip()
    if doc_asset:
        pages = ([viewer.page] if viewer.asset_id == doc_asset
                 and viewer.page is not None else None)
        try:
            document = await read_document_slice(
                ctx, llm, doc_asset, pages=pages, caller="translate")
        except Exception as exc:  # noqa: BLE001 — degrade, never invent content
            logger.info("translate: document read unavailable (%s)", exc)

    subtitles = video_subtitles() if viewer.kind == "video" else ""
    frame_url = ""
    if frame_asset:
        try:
            frame_url = await capture_image_data_url(
                ctx, frame_asset, caller="translate")
        except Exception as exc:  # noqa: BLE001 — subtitles still answer
            logger.info("translate: video frame unavailable (%s)", exc)

    prior = recent_messages(2)
    prompt = _material_sections(
        query=query, prior=prior, document=document,
        subtitles=_bounded(subtitles, 6_000), viewer=viewer,
        has_frame=bool(frame_url))

    logger.info(
        "translate: handler=auto source_material=%s chat_rows=%d frame=%s "
        "document=%s",
        "document" if document else ("video" if viewer.kind == "video" else "chat"),
        len(prior), bool(frame_url), document.name if document else "-")

    try:
        data = await llm.complete_json(
            prompt, system_prompt=_DECISION_SYSTEM,
            images=[frame_url] if frame_url else None)
        if not isinstance(data, dict):
            raise TypeError(f"translate decision returned {type(data).__name__}")
        translation = str(data.get("translation") or "").strip()
        note = str(data.get("note") or "").strip()
        if not translation:
            if note:
                # An honest "the material is not here" line beats an invented text.
                logger.info("translate: source absent (%s)", note)
                return note
            raise ValueError("translate decision returned no translation")
        target = str(data.get("target_language") or "").strip() or DEFAULT_TARGET_LANGUAGE
        logger.info(
            "translate: source=%s target=%s translation_chars=%d note=%s",
            str(data.get("source") or ""), target, len(translation), bool(note))
        return _render(translation, target)
    except Exception as exc:  # noqa: BLE001 — a provider without JSON mode
        logger.warning(
            "translate: structured call unavailable (%s); falling back to one "
            "plain completion", exc)
        plain = prompt + "\n\nAnswer with the translation only — no explanation."
        return str(await llm.complete(plain, _PLAIN_FALLBACK_SYSTEM))


def register(runtime: ToolRuntime, ctx: Context, llm) -> None:
    async def translate(args: dict, exec: ToolExecution) -> str:
        text = str(args.get("text") or "").strip()
        if not text:
            raise ValueError("translate needs text to translate")

        # Explicit payload (Agent lane / direct caller already decided it): the
        # original one-call translation, unchanged. Outside a turn there is no
        # sentence to compare against, so the payload is explicit by definition.
        current = turn_query().strip()
        if not current or text != current:
            target = _target_language(args)
            logger.info("translate: handler=explicit_payload target=%s", target)
            return await llm.complete(text, _PLAIN_SYSTEM.format(target=target))

        return await _auto_translate(args, llm=llm, ctx=ctx)

    runtime.register(
        define_tool(
            name="translate",
            description=(
                "Translate what the user is looking at. "
                "The intent funnel passes the user's sentence verbatim as `text`; "
                "the tool decides whether to translate that sentence, a previous "
                "chat message, the open document page, or the current video, using "
                "the surrounding context to refine the wording, and returns ONLY "
                "the translation. Translating into English when no target language "
                "is named."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The user's request sentence. When it differs "
                                       "from the current message it is an explicit "
                                       "payload to translate directly.",
                    },
                    "target_language": {
                        "type": "string",
                        "description": "Target language name (e.g. 'English', "
                                       "'Chinese'); omitted -> inferred from the "
                                       "request or English.",
                    },
                    "asset_id": {
                        "type": "string",
                        "description": "Captured video-frame asset id injected by the "
                                       "intent funnel after a deferred capture. Never "
                                       "invent one.",
                    },
                },
                "required": ["text"],
            },
            output=ToolOutput(
                schema={"type": "string"}, render=lambda args, value: [text_block(value)]
            ),
            execute=translate,
        )
    )
