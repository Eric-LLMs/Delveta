"""cap_router backend: the deployed cap_router decision SERVICE.

The ``cap_router`` service is served OUT OF PROCESS by the ``deploy/laya``
sidecar (an independent localhost service that loads the current capability
selection checkpoint — the model implementation is LayaChoice — and answers one
choice question per turn over ``POST /v1/systemone``). Keeping it out of process
means the Delveta runtime never imports the model runtime (``laya``/``torch``)
and the multi-GB checkpoint never enters this image.

This backend does EXACTLY one thing: candidate cards -> a ``B_noprov`` choice
question -> a localhost HTTP call -> the model's chosen label -> a
:class:`CapabilityRoute`. It does NOT normalize the candidate list (that is the
business layer's K rule, §26.2), does NOT apply any NONE threshold, does NOT
extract arguments, and never touches the Binder, the Registry or a tool.

4-slot contract (V2): the model answers one of four labels — three capability
ids or the frozen REJECT label. A capability label becomes
``ROUTE_SELECTED``; the REJECT label becomes ``ROUTE_REJECT`` (a NORMAL 4th
decision routing to the real Agent path, never a threshold/fallback).

Failure contract (ruling 2026-10-01): a missing endpoint, a timeout, a transport
error, a malformed/unexpected reply, a missing answer, a non-choice answer, or a
choice outside {candidate ids} ∪ {REJECT} all raise
:class:`~..base.CapabilityRouterUnavailable`. The caller maps that to the Agent —
selection NEVER falls back to the extractor or the legacy ToolIntentModel, so the
cap_router metrics can never be polluted by another model.
"""
from __future__ import annotations

import httpx

from ..card_renderer import REJECT_LABEL, render_choice_question
from ..base import (
    ROUTE_REJECT,
    ROUTE_SELECTED,
    CapabilityRoute,
    CapabilityRouterUnavailable,
)

# The question id the choice answer is keyed under, matching the frozen dataset.
QID = "capability"


class CapRouterSelector:
    """The :class:`..CapabilitySelector` for ``chat_cap_router_backend="cap_router"``.

    Constructed unconditionally by the factory — the endpoint is NOT checked at
    construction (resolution and availability are separate concerns, ruling
    2026-10-01). A missing endpoint surfaces as
    :class:`CapabilityRouterUnavailable` from :meth:`select`.
    """

    async def select(self, query: str, candidates, *, entries_by_id: dict,
                     facts=None) -> CapabilityRoute:
        # Runtime import: config is read at call time, never captured at import.
        from core.config import settings

        url = (settings.chat_cap_router_url or "").strip()
        if not url:
            raise CapabilityRouterUnavailable("no cap_router service deployed")

        question = render_choice_question(candidates, entries_by_id)
        payload = {"state": query, "questions": {QID: question}}
        timeout = settings.chat_cap_router_timeout_seconds

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url.rstrip("/") + "/v1/systemone", json=payload)
                resp.raise_for_status()
                body = resp.json()
            answers = body["answers"]
            answer = answers[QID]
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise CapabilityRouterUnavailable(
                f"cap_router service unreachable/bad reply: {exc!r}") from exc

        if not isinstance(answer, dict) or answer.get("type") != "choice":
            raise CapabilityRouterUnavailable(
                "cap_router service did not answer a choice question")
        choice = answer.get("choice")

        probs = answer.get("probabilities")
        p = probs.get(choice) if isinstance(probs, dict) else None
        confidence = float(p) if isinstance(p, (int, float)) else None

        # REJECT is a NORMAL 4th decision (the V2 4-slot contract) — it routes to
        # the real Agent path, not to the argument chain and never to a fake tool.
        if choice == REJECT_LABEL:
            return CapabilityRoute(
                ROUTE_REJECT, None, confidence=confidence,
                provenance=f"cap_router choice={choice} p={confidence}")

        allowed = {c.capability_id for c in candidates}
        if not isinstance(choice, str) or choice not in allowed:
            raise CapabilityRouterUnavailable(
                f"cap_router service chose off-candidate {choice!r}")

        return CapabilityRoute(
            ROUTE_SELECTED, choice, confidence=confidence,
            provenance=f"cap_router choice={choice} p={confidence}")
