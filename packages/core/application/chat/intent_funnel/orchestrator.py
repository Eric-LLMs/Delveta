"""Funnel orchestration — the one active control flow over the nodes.

Registry → Matcher → (Recall on MISS/AMBIGUOUS) → Candidate Aggregation →
ToolIntentModel (ONE call) → Binder → certified TurnRequirements, every
non-COMPLETE outcome exiting to the Agent byte-identically (8.10). This module
owns sequencing, the wall-clock budget and fail-open classification ONLY —
node algorithms, SQL, prompts, parsing and schema validation live in their own
packages; rollout gating and reason naming live in :mod:`.policy`; the trace
record, its log line and the event row live in :mod:`.observability`.
"""
from __future__ import annotations

import asyncio
import logging

from core.application.chat.understanding import (
    Complexity,
    Confidence,
    Signal,
    TurnRequirements,
)
from core.config import settings

from . import candidate_aggregation, observability, policy
from .contract import (
    MATCH_AMBIGUOUS,
    MATCH_HIT,
    MATCH_MISS,
    REASON_BIND_AMBIGUOUS,
    REASON_BIND_INVALID,
    REASON_BIND_MISSING,
    REASON_CASCADE_ERROR,
    REASON_KIND_DISABLED,
    REASON_NO_CANDIDATE,
    REASON_RECALL_UNAVAILABLE,
    REASON_REGISTRY_UNAVAILABLE,
    REASON_TOOL_INTENT_REJECT,
    REASON_TOOL_INTENT_UNCERTAIN,
    REASON_VERSION_MISMATCH,
    TOOL_INTENT_CONFIDENT,
    TOOL_INTENT_REJECT,
    Candidate,
    TurnFacts,
)

logger = logging.getLogger(__name__)


async def run_cascade(ctx, deps, requirements: TurnRequirements,
                      trace: dict, *,
                      recall_min_score: float | None = None,
                      model_candidate_floor: float | None = None,
                      capture: dict | None = None) -> TurnRequirements | None:
    """One wall-clock-budgeted cascade run with 8.10-classified fail-open.
    Returns a certified TurnRequirements or None (fallback reason in
    ``trace``); the caller decides what None means (production: the original
    object; preview: an agent-route verdict). Never raises.

    Phase-E shadow seam (all default None = byte-identical production): the
    evaluation entry overrides the Recall quality-gate parameters and passes a
    ``capture`` dict to record per-node telemetry the production trace does
    not carry (raw scores, model confidence, binder state). It never gains
    execution authority — the cascade only produces routing metadata (8.8)."""
    import time

    t0 = time.monotonic()
    try:
        out = await asyncio.wait_for(
            run_nodes(ctx, deps, requirements, trace,
                      recall_min_score=recall_min_score,
                      model_candidate_floor=model_candidate_floor, capture=capture),
            settings.chat_funnel_timeout_seconds,
        )
    except TimeoutError:
        out, trace["fallback"] = None, policy.stage_reason(trace["stage"], timed_out=True)
    except Exception as exc:
        out = None
        trace["fallback"] = policy.stage_reason(trace["stage"], timed_out=False)
        logger.info("funnel fail-open at %s: %r", trace["stage"], exc)
    if out is not None:
        trace["final_route"] = "action"
        trace["capability"] = (out.requested_action or {}).get("capability_id")
    elif trace["fallback"] == "-":
        trace["fallback"] = REASON_CASCADE_ERROR  # belt: abstain without a reason is a fault
    trace["total_ms"] = int((time.monotonic() - t0) * 1000)
    return out


async def cascade(ctx, deps, requirements: TurnRequirements) -> TurnRequirements:
    """Run the single-hop cascade under one wall-clock budget, fail-open to the
    ORIGINAL requirements on every abstain/fault (8.10: the Agent's input stays
    byte-identical). Returns the same object the pre-P2 contract guarantees;
    only a certified turn produces a NEW requirements (never a mutation)."""
    trace = observability.new_trace()
    # Phase 6 dark launch: with chat_funnel_trace_capture OFF the cascade is
    # byte-identical to before; ON opens the same evaluation capture seam the
    # shadow/preview lanes use, and its summary rides the event row.
    capture: dict | None = {} if settings.chat_funnel_trace_capture else None
    out = await run_cascade(ctx, deps, requirements, trace, capture=capture)
    observability.log_trace(trace)
    await observability.persist_event(deps, ctx, trace,
                                      observability.trace_json(capture, ctx))
    return out if out is not None else requirements


async def run_nodes(ctx, deps, requirements, trace, *,
                    recall_min_score: float | None = None,
                    model_candidate_floor: float | None = None,
                    capture: dict | None = None):
    """Run the single-hop cascade body (see run_cascade for the shadow seam).
    Returns a certified TurnRequirements, or None after
    setting trace['fallback'] — the caller converts None into the original
    object. Raises only for faults, which the caller maps by trace['stage']."""
    from . import binder, guardrails, matcher, recall
    from .registry import active_view as registry_active_view
    from .registry.entry import chat_plane_candidate
    from .tool_intent import select_and_extract as tool_intent

    message = ctx.body.message or ""
    facts = TurnFacts.of(ctx)

    # ── Registry (reads the LIVE table, fingerprint-cached) ──────────────────────
    view = await registry_active_view(session_factory=deps.session_factory)
    if view is None or not view.entries:
        trace["fallback"] = REASON_REGISTRY_UNAVAILABLE  # no capabilities in the table yet
        return None
    trace["registry"] = view.fingerprint
    entries_by_id = {
        e.capability_id: e for e in view.entries if chat_plane_candidate(e)
    }

    # ── Node 1: Matcher (table-only; the negation guard applies BEFORE it ───────
    # can certify anything, ruling 8.1-a) ────────────────────────────────────────
    # E2 (final semantics 2026-09-26): the HIT lane NEVER touches the Recall
    # index — exact table evidence certifies independently of Recall
    # availability. load_index lives in the MISS/AMBIGUOUS branch below, so an
    # index fault or an unembedded corpus can only ever degrade the Recall lane
    # (RECALL_UNAVAILABLE there, unchanged).
    mres = matcher.match(message, facts, view)
    if mres.state != MATCH_MISS and guardrails.negated(message):
        mres = matcher.MatchResult(state=MATCH_MISS, registry_version=mres.registry_version)
    trace["matcher"] = f"{mres.state}:{mres.capability_id or mres.candidates or '-'}"
    if capture is not None:
        capture["matcher"] = {
            "state": mres.state, "capability_id": mres.capability_id,
            "candidates": list(mres.candidates or []),
            "matched_literal": mres.matched_literal,
        }

    # ── One candidate set, ONE convergence point: a HIT enters ToolIntentModel with the ─
    # same semantics as a Recall lane — the direct-certification special path is
    # deleted (chain ruling 2026-09-24). Recall runs only when the table missed.
    # Live-table ruling 2026-09-26 (RAW lane): every recall hit >= threshold is
    # kept AS IS at the raw stage — no MAX/AVG, no per-capability dedup THERE —
    # a capability may legitimately arrive several times through different
    # sentences. The no-dedup ruling scopes to Raw Recall only: the Capability
    # Candidate Aggregation below (Raw Recall -> ToolIntentModel) then collapses
    # the raw hits into ONE capability-level candidate per capability_id.
    candidates: list[Candidate] = []
    if mres.state == MATCH_AMBIGUOUS:
        for cid in mres.candidates:
            candidates.append(Candidate(cid, 0.0, origin="matcher_ambiguous"))
    elif mres.state == MATCH_HIT:
        candidates.append(Candidate(
            mres.capability_id, 1.0, matched_example=mres.matched_literal,
            origin="matcher_hit", query_kind="standard",
        ))
    if mres.state != MATCH_HIT:
        # E2: Recall availability is ONLY a MISS/AMBIGUOUS-lane dependency.
        trace["stage"] = "recall"
        index = await recall.load_index(deps.session_factory)
        if index is None:
            trace["fallback"] = REASON_RECALL_UNAVAILABLE  # corpus not embedded yet
            return None
        trace["index"] = index.version
        rres = await recall.recall(
            index, message, embedder=deps.embedder(),
            min_score=(settings.chat_funnel_min_score if recall_min_score is None
                       else recall_min_score),
        )
        if capture is not None:
            # the RAW lane: every candidate recall scored, pre any floor AND pre
            # aggregation — the offline threshold sweep recomputes buckets from
            # THIS list (Phase E); aggregation must never overwrite it.
            capture["recall_raw"] = [
                {"capability_id": c.capability_id, "score": c.score,
                 "origin": c.origin, "matched_example": c.matched_example,
                 "query_kind": c.query_kind, "language": c.language,
                 "query_id": c.query_id}
                for c in rres.candidates
            ]
        candidates.extend(rres.candidates)
    trace["recall_count"] = len(candidates)
    if candidates:
        top = max(candidates, key=lambda c: c.score)
        trace["recall_top"] = f"{top.capability_id}@{top.score:.3f}"
    # ── Capability Candidate Aggregation (final semantics 2026-09-26) ──────────
    # Between Raw Recall and ToolIntentModel: group the query-level candidates
    # by capability_id and keep ONE candidate per capability (see
    # :mod:`candidate_aggregation`); the model sees capability-level cards.
    cands = sorted(
        candidate_aggregation.aggregate_by_capability(candidates),
        key=lambda c: c.score, reverse=True)
    if capture is not None:
        capture["candidates"] = [
            {"capability_id": c.capability_id, "score": c.score, "origin": c.origin,
             "matched_example": c.matched_example,
             "kind": c.query_kind or "-", "language": c.language or "-",
             "query_id": c.query_id or "-"}
            for c in cands
        ]
    if model_candidate_floor is not None:
        cands = candidate_aggregation.apply_model_floor(cands, model_candidate_floor)
    # Empty candidate set -> Agent, no model hop (ruling 2026-09-26, supersedes
    # the 2026-09-25 "Action Detection is UNCONDITIONAL" note): with no Matcher
    # HIT/AMBIGUOUS card AND no Recall hit at/above the quality gate there is
    # nothing for the single hop to select from. NO_CANDIDATE is back as the
    # honest deepest-stage=recall exit; the byte-identical turn goes to Agent.
    if not cands:
        trace["fallback"] = REASON_NO_CANDIDATE
        return None

    # ── Node 2: ToolIntentModel — the ONE model call of the turn (select + extract) ────
    trace["stage"] = "tool_intent"
    jv = await tool_intent(message, cands, entries_by_id=entries_by_id,
                       llm=deps.llm, facts=facts)
    trace["tool_intent"] = f"{jv.decision}:{jv.capability_id or '-'}"
    if capture is not None:
        capture["tool_intent"] = {
            "decision": jv.decision, "capability_id": jv.capability_id,
            "confidence": jv.confidence, "arguments": jv.arguments,
            "rationale": jv.rationale,
        }
    if jv.decision != TOOL_INTENT_CONFIDENT:
        trace["fallback"] = (
            REASON_TOOL_INTENT_REJECT if jv.decision == TOOL_INTENT_REJECT
            else REASON_TOOL_INTENT_UNCERTAIN
        )
        return None

    # ── Capability → Binder validate → certified ACTION metadata ───────────────
    entry = entries_by_id.get(jv.capability_id)
    if entry is None:  # a verdict the active table no longer honors: refuse
        trace["fallback"] = REASON_VERSION_MISMATCH
        return None
    if not policy.kind_enabled(entry.intent_kind):  # P3: in the table, but not ON
        trace["fallback"] = REASON_KIND_DISABLED
        return None
    if capture is not None:
        capture["entry"] = {"intent_kind": entry.intent_kind,
                            "tool_binding": entry.tool_binding}
    trace["stage"] = "binder"
    bound = binder.validate(entry, jv.arguments)
    if capture is not None:
        capture["binder"] = bound.state
    if not bound.is_complete:
        trace["fallback"] = {
            "MISSING": REASON_BIND_MISSING,
            "AMBIGUOUS": REASON_BIND_AMBIGUOUS,
            "INVALID": REASON_BIND_INVALID,
        }.get(bound.state, REASON_BIND_MISSING)
        return None
    trace["stage"] = "certified"
    return certified(requirements, entry, bound.args, view.fingerprint,
                     stage="tool_intent")


def certified(requirements, entry, args, registry_fp, *,
              stage: str, integrity: str | None = None) -> TurnRequirements:
    """Certified turn: the same construction shape as the legacy branches
    (only action fields set; source facts ride through). ``funnel_registry_version``
    carries the LIVE Registry content fingerprint — the executor's TOCTOU
    re-validation (8.9) checks against the same fingerprint (migration 0014:
    single namespace, the legacy index-version stamp is gone)."""
    action = {
        "tool": entry.tool_binding, "args": args,
        "capability_id": entry.capability_id,
        "funnel_registry_version": registry_fp,
        "funnel_stage": stage,
        "funnel_kind": entry.intent_kind,
    }
    if integrity is not None:
        action["binding_integrity"] = integrity
    return TurnRequirements(
        needs_action=Signal.HIGH, requested_action=action,
        complexity=Complexity.LOW, confidence=Confidence.HIGH,
        private_only=requirements.private_only,
        external_ok=requirements.external_ok,
    )
