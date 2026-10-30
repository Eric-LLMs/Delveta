"""Answer synthesis for the query capabilities dispatched on the ACTION lane.

The Intent Funnel's ACTION lane exists so a certified single-tool request skips the
ReAct loop: dispatch the bound tool and return its output. That is right for
side-effect tools (``create_folder`` …), whose deterministic output IS the
confirmation, and wrong for query tools (web / rag / social search), whose output is
retrieved MATERIAL — titles, URLs, snippets, page text — not an answer.

The same split applies to a document the user asked about: ``read_document`` hands
back the document's own text, which is material, not an answer. The user's sentence
decides what the turn makes of it — explain, summarize, analyse, compare, or (when the
sentence asks for the text itself) reproduce it faithfully. Capabilities whose whole
purpose is to hand the material back unchanged (``pdf_extract_text``, ``read_file``)
stay out of this set: their own exact output is the deliverable.

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
# other tool keeps the deterministic output it already produces. ``read_document``'s
# capability is "read the document and answer from its contents", so its extracted
# text is input to the turn's answer, never the answer itself.
SYNTHESIS_TOOLS = frozenset(
    {"web_search", "rag_search", "search_social", "read_document"})

# Belt against pathological tool output blowing the prompt (the search tools bound
# their own page text): truncation is stated in the prompt so the model never reads a
# cut-off list as complete.
_MAX_MATERIAL_CHARS = 24000

_SYSTEM_PROMPT = (
    "You answer the user's question from the retrieved material supplied in the "
    "prompt. Use ONLY that material: do not add facts, numbers, dates or events it "
    "does not state, and never invent a source, citation or link. If the material "
    "does not answer the question, say so plainly and name what is missing. When the "
    "user asks for the material's own text — to extract, transcribe, export, quote "
    "or read the original — reproduce that text faithfully and completely, without "
    "summarizing, rewriting or omitting any of it. Otherwise organize the material "
    "into a direct answer — summarize, compare or analyse as the question requires — "
    "instead of repeating it verbatim. Preserve the sources, URLs and citations the "
    "material carries, and cite nothing else. Reply in the same language as the "
    "user's question."
)


def build_prompt(query: str, tool: str, raw_output: str) -> str:
    """The one synthesis prompt: the user's question plus the material, verbatim."""
    if len(raw_output) > _MAX_MATERIAL_CHARS:
        material = raw_output[:_MAX_MATERIAL_CHARS] + "\n…[material truncated]"
    else:
        material = raw_output
    return (
        f"User question:\n{query}\n\n"
        f"Material (tool: {tool}):\n{material}\n\n"
        "Answer the user's question from this material: do what the question asks "
        "with it — explain, summarize, analyse or compare it, or reproduce its text "
        "when that is what was asked."
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
    try:
        # No model/base_url/api_key override: the turn's pinned channel wins (see the
        # module docstring). Passing them here would fork onto a different route.
        async for delta in llm.complete_stream(
            build_prompt(query, tool, raw_output), system_prompt=_SYSTEM_PROMPT
        ):
            if not delta:
                continue
            if not started:
                if not delta.strip():
                    continue
                started = True
            parts.append(delta)
            yield delta
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
