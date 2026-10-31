"""Answer synthesis for the query capabilities dispatched on the ACTION lane.

The Intent Funnel's ACTION lane exists so a certified single-tool request skips the
ReAct loop: dispatch the bound tool and return its output. That is right for
side-effect tools (``create_folder`` …), whose deterministic output IS the
confirmation, and wrong for query tools (web / rag / social search), whose output is
retrieved MATERIAL — titles, URLs, snippets, page text — not an answer.

The same split applies to a document the user asked about: ``read_document``,
``pdf_extract_text`` and ``read_file`` hand back the document's own text, which is
material, not an answer. The user's sentence decides what the turn makes of it —
explain, summarize, analyse, compare — or, when the sentence asks for the text itself,
hand that text back untouched.

"Ask for the text itself" is decided by the model inside the SAME completion: when the
request is only to see the material's own text it replies with :data:`_VERBATIM_TOKEN`
and nothing else, and the executor streams the tool's own text in chunks instead of the
model's copy. That keeps one model call per turn while guaranteeing the verbatim answer
is complete, in order, unrewritten and never truncated by the model's output limit. The
marker is internal: it is never streamed and never saved.

This module owns the missing step for exactly those tools: one LLM completion that
turns (user question + retrieved material) into a user-facing answer. It reuses
``deps.llm``, so ``OpenAILLM._call_channel``'s explicit-args > request-contextvar >
client-default resolution applies unchanged: the turn's pinned model / channel,
allowance and billing path stay the same, and no second client is constructed here.

Failure is fail-open DOWNWARDS: a failed or empty completion returns "" and the
caller keeps the raw tool output — retrieved data is never lost.

Two entry points, one contract:

* :func:`synthesize_answer` — the whole answer (non-streaming ``/chat``);
* :func:`stream_synthesis` — the SAME prompt over ``complete_stream``, yielding the
  answer as the model produces it (the streaming ``/chat/stream`` path).
"""
from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)

# The query capabilities whose tool output is material rather than an answer. Every
# other tool keeps the deterministic output it already produces. ``read_document``,
# ``pdf_extract_text`` and ``read_file`` return the user's own text: the turn's
# sentence decides whether it is answered from (explain/summarize/analyse/compare) or
# handed back as-is (the verbatim marker).
SYNTHESIS_TOOLS = frozenset(
    {"web_search", "rag_search", "search_social",
     "read_document", "pdf_extract_text", "read_file"})

# The model's internal control marker for "the user only asked for the text". It is
# never streamed and never saved: the executor answers with the tool's own text.
_VERBATIM_TOKEN = "[[VERBATIM]]"

# Raw material is streamed in chunks of this size so a long document renders
# progressively instead of arriving as one frame.
_VERBATIM_CHUNK_CHARS = 512

# Belt against pathological tool output blowing the prompt (the search tools bound
# their own page text): truncation is stated in the prompt so the model never reads a
# cut-off list as complete.
_MAX_MATERIAL_CHARS = 24000

_SYSTEM_PROMPT = (
    "You answer the user's question from the retrieved material supplied in the "
    "prompt. Use ONLY that material: do not add facts, numbers, dates or events it "
    "does not state, and never invent a source, citation or link. If the material "
    "does not answer the question, say so plainly and name what is missing. The "
    "material is evidence, not the answer. Judge it before you write: whether a source "
    "is authoritative and directly relevant, whether it is current, and whether a date "
    "on the page is its publication date, the event's date, or the date its data "
    "describes. Pages repeating one wire story are one source, not independent "
    "confirmation — never infer reliability from list order or from how many pages "
    "agree. Synthesize: extract the facts the question needs, cross-check the "
    "load-bearing ones, and write one coherent answer that leads with the direct "
    "answer. When sources conflict, compare their quality and time validity first: if "
    "the evidence supports one, use it and explain the necessary difference briefly; "
    "answer the parts that can be confirmed and state uncertainty only for the specific "
    "facts that cannot. Never average, vote, splice or invent a value to make the "
    "sources fit. Do not walk through the results site by site, narrate the search or "
    "expose your reasoning — that is right only when the user explicitly asks to "
    "compare sources. If the material is insufficient, the question is not "
    "automatically unanswerable: answer what the evidence supports, state precisely "
    "what remains unknown, and never ask the user to verify it for you. When the "
    "material carries retrieval metadata, use its time as the reference for today and "
    "now. When "
    "the user's request is ONLY to see the material's own text — to read, extract, "
    "transcribe, export, show or quote the original — reply with exactly "
    f"{_VERBATIM_TOKEN} and nothing else; the system then delivers the material "
    "itself, so never reproduce it. Do not use that marker when the request also "
    "asks for anything done WITH the text (summarize, explain, analyse, compare, "
    "extend, or quote parts while answering): answer normally instead. Otherwise "
    "organize the material into a direct answer — summarize, compare or analyse as "
    "the question requires — instead of repeating it verbatim. Preserve the sources "
    "and citations the material carries, and cite nothing else. When you name a source "
    "you actually used, make the name a Markdown link to that page — "
    "[中国天气网](https://…) — with the URL copied exactly from the material; never "
    "build a URL from a domain name, and write the name as plain text when the "
    "material gives no URL for it. Never print a bare URL or domain, and do not add a "
    "sources list of your own. Reply in the same language as the user's question."
)


def _verbatim_chunks(text: str, size: int = _VERBATIM_CHUNK_CHARS) -> list[str]:
    """The material in fixed-size chunks, in order, with nothing dropped or added."""
    return [text[i:i + size] for i in range(0, len(text), size)] or [text]


async def _close_stream(stream: object) -> None:
    """Stop the model stream once the reply's mode is known (best effort)."""
    close = getattr(stream, "aclose", None)
    if close is None:
        return
    try:
        await close()
    except Exception:  # noqa: BLE001 - closing is best-effort, never a turn failure
        logger.debug("chat.answer-synthesis: closing the model stream failed",
                     exc_info=True)


def build_prompt(query: str, tool: str, raw_output: str) -> str:
    """The one synthesis prompt: the user's question plus the material, verbatim."""
    if len(raw_output) > _MAX_MATERIAL_CHARS:
        material = raw_output[:_MAX_MATERIAL_CHARS] + "\n…[material truncated]"
    else:
        material = raw_output
    return (
        f"User question:\n{query}\n\n"
        f"Material (tool: {tool}):\n{material}\n\n"
        "Answer the user's question from this material: explain, summarize, analyse, "
        "compare or otherwise work on it as the question requires. If the request is "
        f"only to see the material's own text, reply with {_VERBATIM_TOKEN} alone."
    )


async def synthesize_answer(deps, ctx, tool: str, raw_output: str) -> str:
    """Turn retrieved material into a user-facing answer.

    Returns "" when synthesis is not possible (no LLM seam, empty material, an LLM
    error, or an empty reply) — the caller then keeps the raw tool output. Never
    raises: a synthesis failure must not turn a successful tool dispatch into a
    failed turn.
    """
    llm = getattr(deps, "llm", None)
    if llm is None:
        logger.warning("chat.answer-synthesis skipped tool=%s: no llm seam", tool)
        return ""
    if not raw_output.strip():
        return ""
    query = str(getattr(ctx, "user_text", "") or "").strip()

    t0 = time.perf_counter()
    try:
        # No model/base_url/api_key override: the turn's pinned channel wins (see the
        # module docstring). Passing them here would fork onto a different route.
        answer = await llm.complete(
            build_prompt(query, tool, raw_output), system_prompt=_SYSTEM_PROMPT)
    except Exception:  # noqa: BLE001 - fail-open DOWN to the raw tool output
        logger.exception(
            "chat.answer-synthesis failed tool=%s duration_ms=%.0f; "
            "falling back to the raw tool output",
            tool, (time.perf_counter() - t0) * 1000,
        )
        return ""

    duration_ms = (time.perf_counter() - t0) * 1000
    answer = (answer or "").strip()
    if answer.startswith(_VERBATIM_TOKEN):
        # The user asked for the text itself: the tool's own output IS the answer.
        logger.info(
            "chat.answer-synthesis verbatim tool=%s duration_ms=%.0f "
            "material_chars=%d",
            tool, duration_ms, len(raw_output),
        )
        return raw_output
    if not answer:
        logger.warning(
            "chat.answer-synthesis empty tool=%s duration_ms=%.0f; "
            "falling back to the raw tool output",
            tool, duration_ms,
        )
        return ""
    logger.info(
        "chat.answer-synthesis tool=%s duration_ms=%.0f material_chars=%d answer_chars=%d",
        tool, duration_ms, len(raw_output), len(answer),
    )
    return answer


async def stream_synthesis(
    deps, ctx, tool: str, raw_output: str
) -> AsyncIterator[str]:
    """Stream the synthesized answer as the model produces it.

    Same prompt, same channel resolution as :func:`synthesize_answer`; only the wire
    changes (``complete_stream`` instead of an accumulated completion). The caller
    accumulates the deltas — it needs the full text for the session row and the done
    payload.

    Contract:

    * no delta is ever yielded after a failure, and a failure BEFORE the first delta
      yields nothing, so the caller falls back to the raw tool output exactly as the
      non-streaming path does;
    * a failure AFTER the first delta keeps the streamed prefix and yields nothing
      more — never the raw material spliced onto a partial answer;
    * a reply that is only :data:`_VERBATIM_TOKEN` yields the tool's own text in
      chunks and nothing from the model: complete, in order, unrewritten;
    * a seam without ``complete_stream`` degrades to one whole-answer delta.
    """
    llm = getattr(deps, "llm", None)
    if llm is None:
        logger.warning("chat.answer-synthesis skipped tool=%s: no llm seam", tool)
        return
    if not raw_output.strip():
        return

    if not hasattr(llm, "complete_stream"):
        # Duck-typed seam without the streaming capability: same text, one frame.
        answer = await synthesize_answer(deps, ctx, tool, raw_output)
        if answer:
            yield answer
        return

    query = str(getattr(ctx, "user_text", "") or "").strip()
    t0 = time.perf_counter()
    parts: list[str] = []
    # Leading whitespace deltas are held back until the answer really starts: an
    # all-whitespace reply must emit NOTHING so the caller falls back to the raw
    # material, and the joined deltas must stay identical to the caller's message.
    started = False
    # The reply's mode is not known until its first characters arrive: a verbatim
    # reply must never reach the user. Only a marker-length prefix is held back, so
    # the answer is still streamed as it is written.
    mode = ""          # "" unknown | "answer" | "verbatim"
    pending = ""
    stream = llm.complete_stream(
        build_prompt(query, tool, raw_output), system_prompt=_SYSTEM_PROMPT)
    try:
        # No model/base_url/api_key override: the turn's pinned channel wins (see the
        # module docstring). Passing them here would fork onto a different route.
        async for delta in stream:
            if not delta:
                continue
            if mode == "answer":
                if not started:
                    if not delta.strip():
                        continue
                    started = True
                parts.append(delta)
                yield delta
                continue
            pending += delta
            head = pending.lstrip()
            if head.startswith(_VERBATIM_TOKEN):
                mode = "verbatim"
                await _close_stream(stream)   # none of the model's text is used
                break
            if len(head) < len(_VERBATIM_TOKEN) and _VERBATIM_TOKEN.startswith(head):
                continue                      # could still become the marker
            mode = "answer"
            if not head:
                pending = ""
                continue
            started = True
            parts.append(head)
            pending = ""
            yield head
    except Exception:  # noqa: BLE001 - fail-open DOWN to the raw tool output
        logger.exception(
            "chat.answer-synthesis stream failed tool=%s duration_ms=%.0f "
            "streamed_chars=%d; %s",
            tool, (time.perf_counter() - t0) * 1000, sum(len(p) for p in parts),
            "keeping the streamed prefix"
            if parts else "falling back to the raw tool output",
        )
        return

    duration_ms = (time.perf_counter() - t0) * 1000
    if mode == "verbatim":
        # The user asked for the text itself: stream the tool's own text, unchanged
        # and complete — never the model's copy, which could be truncated or rewritten.
        for chunk in _verbatim_chunks(raw_output):
            parts.append(chunk)
            yield chunk
        logger.info(
            "chat.answer-synthesis stream verbatim tool=%s duration_ms=%.0f "
            "material_chars=%d chunks=%d",
            tool, duration_ms, len(raw_output), len(parts),
        )
        return
    answer = "".join(parts)
    if not answer.strip():
        logger.warning(
            "chat.answer-synthesis stream empty tool=%s duration_ms=%.0f; "
            "falling back to the raw tool output",
            tool, duration_ms,
        )
        return
    logger.info(
        "chat.answer-synthesis stream tool=%s duration_ms=%.0f material_chars=%d "
        "answer_chars=%d deltas=%d",
        tool, duration_ms, len(raw_output), len(answer), len(parts),
    )
