"""Answer synthesis for the query capabilities dispatched on the ACTION lane.

The Intent Funnel's ACTION lane exists so a certified single-tool request skips the
ReAct loop: dispatch the bound tool and return its output. That is right for
side-effect tools (``create_folder`` …), whose deterministic output IS the
confirmation, and wrong for query tools (web / rag / social search), whose output is
retrieved MATERIAL — titles, URLs, snippets, page text — not an answer.

This module owns the missing step for exactly those tools: one LLM completion that
turns (user question + retrieved material) into a user-facing answer. It reuses
``deps.llm``, so ``OpenAILLM._call_channel``'s explicit-args > request-contextvar >
client-default resolution applies unchanged: the turn's pinned model / channel,
allowance and billing path stay the same, and no second client is constructed here.

Failure is fail-open DOWNWARDS: a failed or empty completion returns "" and the
caller keeps the raw tool output — retrieved data is never lost.
"""
from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

# The query capabilities whose tool output is retrieved material rather than an
# answer. Every other tool keeps the deterministic output it already produces.
SYNTHESIS_TOOLS = frozenset({"web_search", "rag_search", "search_social"})

# Belt against pathological tool output blowing the prompt (the search tools bound
# their own page text): truncation is stated in the prompt so the model never reads a
# cut-off list as complete.
_MAX_MATERIAL_CHARS = 24000

_SYSTEM_PROMPT = (
    "You answer the user's question from the retrieved material supplied in the "
    "prompt. Use ONLY that material: do not add facts, numbers, dates or events it "
    "does not state, and never invent a source, citation or link. If the material "
    "does not answer the question, say so plainly and name what is missing. Organize "
    "the material into a direct answer — summarize, compare or analyse as the "
    "question requires — instead of repeating it verbatim. Preserve the sources, "
    "URLs and citations the material carries, and cite nothing else. Reply in the "
    "same language as the user's question."
)


def build_prompt(query: str, tool: str, raw_output: str) -> str:
    """The one synthesis prompt: the user's question plus the material, verbatim."""
    if len(raw_output) > _MAX_MATERIAL_CHARS:
        material = raw_output[:_MAX_MATERIAL_CHARS] + "\n…[material truncated]"
    else:
        material = raw_output
    return (
        f"User question:\n{query}\n\n"
        f"Retrieved material (tool: {tool}):\n{material}\n\n"
        "Answer the user's question from this material."
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
