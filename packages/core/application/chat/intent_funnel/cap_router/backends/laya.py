"""cap_router backend: the real LayaChoice decision model.

LayaChoice is served OUT OF PROCESS by the ``deploy/laya`` sidecar (an
independent localhost service that loads the fine-tuned checkpoint and answers
one choice question per turn over ``POST /v1/systemone``). Keeping it out of
process means the Delveta runtime never imports ``laya``/``torch`` and the
1.3 GB checkpoint never enters this image.

This backend does EXACTLY one thing: candidate cards -> a ``B_noprov`` choice
question -> a localhost HTTP call -> the model's chosen capability label -> a
:class:`CapabilityRoute`. It does NOT normalize the candidate list (that is the
business layer's K rule, §26.2), does NOT apply any NONE threshold, does NOT
extract arguments, and never touches the Binder, the Registry or a tool.

Failure contract (ruling 2026-10-01): a missing endpoint, a timeout, a transport
error, a malformed/unexpected reply, a missing answer, a non-choice answer, or a
choice outside the candidate set all raise
:class:`~..base.CapabilityRouterUnavailable`. The caller maps that to the Agent —
selection NEVER falls back to Qwen or the legacy ToolIntentModel, so the Laya
metrics can never be polluted by the model under test.
"""
from __future__ import annotations

import httpx

from ..card_renderer import render_choice_question
from ..base import ROUTE_SELECTED, CapabilityRoute, CapabilityRouterUnavailable

# The question id the choice answer is keyed under, matching the frozen dataset.
QID = "capability"


class LayaSelector:
    """The :class:`..CapabilitySelector` for ``chat_cap_router_backend="laya"``.

    Constructed unconditionally by the factory — the endpoint is NOT checked at
    construction (resolution and availability are separate concerns, ruling
    2026-10-01). A missing endpoint surfaces as
    :class:`CapabilityRouterUnavailable` from :meth:`select`.
    """

    async def select(self, query: str, candidates, *, entries_by_id: dict,
                     facts=None) -> CapabilityRoute:
        # Runtime import: config is read at call time, never captured at import.
        from core.config import settings

        url = (settings.chat_cap_router_laya_url or "").strip()
        if not url:
            raise CapabilityRouterUnavailable("no LayaChoice model (laya) deployed")

        question = render_choice_question(candidates, entries_by_id)
        payload = {"state": query, "questions": {QID: question}}
        timeout = settings.chat_cap_router_laya_timeout_seconds

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url.rstrip("/") + "/v1/systemone", json=payload)
                resp.raise_for_status()
                body = resp.json()
            answers = body["answers"]
            answer = answers[QID]
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise CapabilityRouterUnavailable(
                f"LayaChoice model (laya) unreachable/bad reply: {exc!r}") from exc

        if not isinstance(answer, dict) or answer.get("type") != "choice":
            raise CapabilityRouterUnavailable(
                "LayaChoice model (laya) did not answer a choice question")
        choice = answer.get("choice")
        allowed = {c.capability_id for c in candidates}
        if not isinstance(choice, str) or choice not in allowed:
            raise CapabilityRouterUnavailable(
                f"LayaChoice model (laya) chose off-candidate {choice!r}")

        probs = answer.get("probabilities")
        p = probs.get(choice) if isinstance(probs, dict) else None
        confidence = float(p) if isinstance(p, (int, float)) else None
        return CapabilityRoute(
            ROUTE_SELECTED, choice, confidence=confidence,
            provenance=f"laya choice={choice} p={confidence}")
