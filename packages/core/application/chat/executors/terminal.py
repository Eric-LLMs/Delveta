"""TerminalExecutor: the honest end of a decided-but-unsettled request.

Serves ``PlanKind.ERROR``: the Intent Funnel DECIDED the capability (Matcher HIT
or cap_router SELECTED), but argument acquisition settled that a required
parameter cannot be produced reliably — today the file capabilities' page scope,
where neither the deterministic rule nor the online model could name the pages.

The turn ends with one deterministic message. This branch:

* makes NO model call (the words ARE the answer);
* never hands the turn to the Agent to guess the parameter;
* never executes a tool;
* records the turn in session memory exactly like the ACTION branch, so the
  user's request and the honest refusal both stay in the transcript.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from core.application.chat.execution_plan import PlanKind
from core.application.chat.executors.base import (
    ChatEvent,
    ExecutorResult,
    ProgressSink,
    TurnRequest,
)
from core.application.chat.intent_funnel.cap_handler.acquisition_failure import (
    CATEGORY_SCOPE_UNSETTLED,
)

# Deterministic terminal messages (no LLM on this branch).
_SCOPE_UNSETTLED = (
    "I could not tell which pages of the document you meant, so I did not run the "
    "request. Please name the pages explicitly (for example “pages 2-4”), or say "
    "that you want the whole document."
)
_GENERIC = (
    "That request could not be run because one of its parameters could not be "
    "determined reliably. Please restate the request with the missing detail."
)


def _acquisition_record(error: dict) -> dict:
    acquisition = error.get("acquisition")
    return acquisition if isinstance(acquisition, dict) else {}


def terminal_message(error: dict | None) -> str:
    """Render the honest answer for this terminal acquisition failure."""
    error = error if isinstance(error, dict) else {}
    acquisition = _acquisition_record(error)
    if str(acquisition.get("category") or "") == CATEGORY_SCOPE_UNSETTLED:
        return _SCOPE_UNSETTLED
    return _GENERIC


class TerminalExecutor:
    kind = PlanKind.ERROR

    async def stream(
        self, req: TurnRequest, *, progress_sink: ProgressSink
    ) -> AsyncIterator[ChatEvent]:
        error = req.plan.error
        message = terminal_message(error)
        ctx = req.ctx
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in (ctx.history or [])
            if isinstance(m.get("content"), str) and m.get("role") in ("user", "assistant")
        ]
        await ctx.session_memory.append_message("user", ctx.user_text)
        await ctx.session_memory.append_message("assistant", message)
        yield {"type": "content", "data": message}
        yield {"type": "done", "data": {
            "answer": message,
            "messages": [*history, {"role": "user", "content": ctx.user_text},
                         {"role": "assistant", "content": message}],
            "usage": {},
            "error": None,
            "cost_usd": None,
            "terminal_reason": str((error or {}).get("reason") or ""),
        }}

    async def run(self, req: TurnRequest) -> ExecutorResult:
        error = req.plan.error
        message = terminal_message(error)
        ctx = req.ctx
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in (ctx.history or [])
            if isinstance(m.get("content"), str) and m.get("role") in ("user", "assistant")
        ]
        await ctx.session_memory.append_message("user", ctx.user_text)
        await ctx.session_memory.append_message("assistant", message)
        messages = [
            *history,
            {"role": "user", "content": ctx.user_text},
            {"role": "assistant", "content": message},
        ]
        return ExecutorResult(messages=messages, final_answer=message, usage={})
