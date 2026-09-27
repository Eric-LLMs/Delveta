"""Shadow lanes — observation with zero execution authority.

Two shadows share this module, both Observer-only:

1. :func:`observe` — the 8.15 Matcher dark-launch hook, formalizing step 3's
   hook. Tri-state switch: ``settings.chat_matcher_mode`` (validated in
   ``core.config`` docs + :func:`orchestrator`-side call in ``funnel.route``):

   * ``off``    — the node never runs (dark-launch default);
   * ``shadow`` — the Registry-backed Matcher runs on every turn and its
     verdict is logged as ``would_*`` telemetry next to the L0 outcome;
     routing is untouched and the Agent keeps the turn byte-identically;
   * ``on``     — deterministic certification INSIDE the new cascade (only
     with ``chat_funnel_enabled``; see orchestrator.run_nodes). Without the
     funnel gate it keeps running SHADOW semantics with a one-time warning: a
     mis-set switch must never silently hand routing to a node measured only
     in the dark.

2. :func:`cascade_shadow` — the Phase-E full-cascade shadow: one dry-run turn
   through the SAME orchestrator body as production (zero cascade
   duplication, so the shadow can never drift from shipped semantics).

Two invariants this module owns:

1. observation, never behavior — every failure is fail-quiet; the only
   observable effect of a broken shadow node is absent log lines (8.15: "结果
   不得影响当前 Agent 行为");
2. cost isolation — the whole observation runs under an ``execution_mode=shadow``
   pin (8.14), so any LLM/embedding usage a future shadow stage records settles
   outside real-user billing. The pin is always reset, exception included.
"""
from __future__ import annotations

import logging

from core.application.chat.understanding import (
    Complexity,
    Confidence,
    Signal,
    TurnRequirements,
)
from core.infrastructure.request_context import (
    reset_request_execution_mode,
    set_request_execution_mode,
)

from . import observability
from .contract import MATCH_AMBIGUOUS, MATCH_HIT, MatchResult, TurnFacts
from .orchestrator import run_cascade

logger = logging.getLogger(__name__)

MODES = ("off", "shadow", "on")

# fallback_reason vocabulary (8.10: prefixed codes, never a bare word)
_FALLBACK_REASONS = {
    MATCH_HIT: "-",
    MATCH_AMBIGUOUS: "matcher_ambiguous",
}
_DEFAULT_FALLBACK = "matcher_miss"
_WOULD_STAGE = "matcher"  # the node's own stage id; ToolIntentModel arrives in P2

_warned_on = False


def matcher_mode() -> str:
    """The sanitized tri-state value; anything unknown fails safe to ``off``."""
    from core.config import settings

    mode = (settings.chat_matcher_mode or "off").strip().lower()
    if mode not in MODES:
        logger.warning("unknown chat_matcher_mode=%r; treating as off", mode)
        return "off"
    return mode


async def observe(ctx, deps, requirements, mode: str) -> None:
    """Run the Matcher in the dark and log its verdict. Never raises, never
    routes; pins execution_mode=shadow for its duration only."""
    global _warned_on
    if mode == "on" and not _warned_on:
        from .policy import funnel_live  # policy owns the gate; no funnel back-edge

        if not funnel_live(requirements, deps, ctx):
            _warned_on = True  # once per process — a mis-set switch, not a per-turn event
            logger.warning(
                "chat_matcher_mode=on but chat_funnel_enabled is off: authoritative "
                "Matcher certification needs the funnel gate; shadow semantics only"
            )
    token = set_request_execution_mode("shadow")
    try:
        await _match_and_log(ctx, deps, requirements, mode)
    except Exception as exc:  # noqa: BLE001 - shadow is observation, never behavior
        logger.info("matcher shadow fail-quiet: %r", exc)
    finally:
        reset_request_execution_mode(token)


async def _match_and_log(ctx, deps, requirements, mode: str) -> None:
    from . import matcher
    from .registry import active_view  # inside: keeps the monkeypatch seam alive

    view = await active_view(session_factory=deps.session_factory)
    if view is None:
        return  # no capabilities in the live table yet — no comparison possible
    res = matcher.match(getattr(ctx.body, "message", "") or "",
                        TurnFacts.of(ctx), view)
    l0_tool = (requirements.requested_action or {}).get("tool")
    # live-table ruling (2026-09-26): there is no version number any more — the
    # Registry view IS the live corpus, identified by its content fingerprint.
    logger.info(
        "matcher_shadow mode=%s state=%s registry_fingerprint=%s "
        "would_route=%s would_capability=%s would_stage=%s confidence=%s "
        "fallback_reason=%s candidates=%s pattern=%s l0_tool=%s agreement=%s",
        mode, res.state, view.fingerprint,
        "t" if res.state == MATCH_HIT else "f",
        res.capability_id or "-", _WOULD_STAGE,
        _confidence(res),
        _FALLBACK_REASONS.get(res.state, _DEFAULT_FALLBACK),
        ",".join(res.candidates) or "-",
        # exact-only Matcher (ruling 2026-09-25): this is the human-readable
        # corpus sentence that produced the HIT — never a regex literal
        (res.matched_literal or "-").replace(" ", "_"),
        l0_tool or "-", _agreement(res, view, l0_tool),
    )


def _agreement(res: MatchResult, view, l0_tool: str | None) -> str:
    """The L0-vs-Matcher verdict pair for the equivalence dataset (8.15).

    ``match``/``mismatch`` compare tool bindings — a HIT whose capability binds
    the same tool L0 certified is agreement; a different tool is the disagreement
    sample P2 promotion needs to select_and_extract. The other three codes locate which
    side abstained (both-miss turns are ``none``)."""
    if res.state != MATCH_HIT:
        return "l0_only" if l0_tool else "none"
    tool = next((e.tool_binding for e in view.entries
                 if e.capability_id == res.capability_id), None)
    if not l0_tool:
        return "matcher_only"
    if tool is None:  # HIT on an entry outside the view — defensive, shouldn't happen
        return "unknown"
    return "match" if tool == l0_tool else "mismatch"


def _confidence(res: MatchResult) -> str:
    """The deterministic Matcher is always sure of a table hit; a real
    calibrated confidence arrives with the ToolIntentModel (P2, 8.13's data-first plan)."""
    return "1.0" if res.state == MATCH_HIT else "0.0"


# ── Phase-E full-cascade shadow (SAME orchestrator body, raw-score seam open) ────

async def cascade_shadow(ctx, *, deps, requirements=None,
                         recall_min_score: float = 0.0,
                         model_candidate_floor: float = 0.58,
                         persist_event: bool = False,
                         turn_key: str = "") -> dict:
    """Phase-E Cascade Shadow: one dry-run turn through the SAME node body as
    production (:func:`orchestrator.run_nodes` — zero orchestration
    duplication, so the shadow can never drift from shipped semantics), with
    the raw-score seam opened: Recall keeps every candidate it scored
    (min_score=0; the live-table ruling already keeps EVERY hit >= threshold as
    its own candidate) and the model-facing set re-applies a floor, so ALL
    threshold buckets are recomputable OFFLINE from the captured raw scores —
    recall is never re-run per threshold. The chain stops at Binder: routing
    metadata only (8.8), no dispatch, no Runtime, no event row; usage is pinned
    ``execution_mode=shadow`` (8.14) and observability is log-only.
    ``would_execute`` is the certified turn's metadata, NOT a permission —
    by construction the cascade cannot execute anything from here.
    Never raises: faults surface as 8.10 fallback reasons, exactly as in
    production.

    Shadow-live A/B seam (2026-09-27, both defaults = byte-identical replay):
    ``persist_event=True`` also writes the 8.12 event row (still pinned
    ``execution_mode=shadow``) so live observation accumulates in the same
    table the observability admin reads; ``turn_key`` rides the row's
    trace_json (plus the stage captures — never the query) as the join key
    against the orchestrator's ``funnel_ab_turn`` line."""
    if requirements is None:
        requirements = TurnRequirements(
            complexity=Complexity.LOW, confidence=Confidence.LOW,
            needs_web=Signal.LOW, needs_memory=False,
        )
    trace = observability.new_trace()
    capture: dict = {}
    token = set_request_execution_mode("shadow")
    try:
        out = await run_cascade(ctx, deps, requirements, trace,
                                recall_min_score=recall_min_score,
                                model_candidate_floor=model_candidate_floor,
                                capture=capture)
        if persist_event:
            # Inside the pin: the row says execution_mode=shadow (8.14). The
            # trace_json carries the turn_key join anchor + stage captures,
            # never the raw query (same no-query rule as production rows).
            _act = (out.requested_action or {}) if out is not None else {}
            await observability.persist_event(deps, ctx, trace, {
                "turn_key": turn_key or None,
                "matcher": capture.get("matcher"),
                "candidates": capture.get("candidates", []),
                "model_verdict": capture.get("tool_intent"),
                "would_execute": {
                    k: _act.get(k) for k in
                    ("capability_id", "args", "funnel_stage", "funnel_kind")
                } if _act else None,
            })
    finally:
        reset_request_execution_mode(token)
    observability.log_trace(trace)
    result = {
        "deepest_stage": trace["stage"], "matcher": trace["matcher"],
        "recall_count": trace["recall_count"], "recall_top": trace["recall_top"],
        "tool_intent": trace["tool_intent"],
        "final_route": trace["final_route"], "fallback_reason": trace["fallback"],
        "registry_version": trace["registry"], "index_version": trace["index"],
        "total_ms": trace["total_ms"], "execution_mode": "shadow",
        "capture": capture,
    }
    act = (out.requested_action or {}) if out is not None else {}
    result["would_execute"] = ({k: act.get(k) for k in (
        "capability_id", "args", "funnel_stage", "funnel_kind",
    )} if out is not None else None)
    return result
