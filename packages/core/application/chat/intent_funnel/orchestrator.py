"""Funnel orchestration — the one active control flow over the nodes.

Registry → Matcher → (Recall on MISS/AMBIGUOUS) → Candidate Aggregation →
ToolIntentModel (ONE call) → Binder → certified TurnRequirements, every
non-COMPLETE outcome exiting to the Agent byte-identically (8.10).

Phase 4 seam (2026-10-01): when ``chat_cap_router_backend != off`` the decided
capability — a MATCH_HIT directly, or a MISS/AMBIGUOUS via ``Recall ->
Aggregation -> cap_router -> ONE|NONE`` — feeds the **Argument Path Router**
(:mod:`.argument_acquisition.path_router`), which derives the acquisition
strategy from the capability's declaration + injected inputs. A MATCH_HIT is
NEVER re-selected (no Recall, no Aggregation, no cap_router). This step wires
the ARP only: CONTEXT_DIRECT reuses the existing Binder/certified handoff, while
a MODEL acquisition need (QUERY_TO_QWEN / QUERY_PLUS_5_USER_TURNS / MIXED) is
not yet executable (no Qwen) and exits to the Agent; so does an
acquisition-undeclared or -MISSING capability. With the default ``off`` every
turn takes the fully-legacy hop, byte-identical.
This module
owns sequencing, the wall-clock budget and fail-open classification ONLY —
node algorithms, SQL, prompts, parsing and schema validation live in their own
packages; rollout gating and reason naming live in :mod:`.policy`; the trace
record, its log line and the event row live in :mod:`.observability`.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from core.config import settings

if TYPE_CHECKING:  # import cycle: understanding -> actions -> intent_funnel (see __init__)
    from core.application.chat.understanding import (
        Complexity,
        Confidence,
        Signal,
        TurnRequirements,
    )

from . import candidate_aggregation, observability, policy
from .contract import (
    MATCH_AMBIGUOUS,
    MATCH_HIT,
    MATCH_MISS,
    REASON_ACQUISITION_MISSING,
    REASON_ACQUISITION_MODEL_PENDING,
    REASON_ACQUISITION_UNDECLARED,
    REASON_BIND_AMBIGUOUS,
    REASON_BIND_INVALID,
    REASON_BIND_MISSING,
    REASON_CAP_ROUTER_NONE,
    REASON_CAP_ROUTER_UNAVAILABLE,
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
    from . import cap_router, guardrails, matcher, recall
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

    # ── New lane, MATCH_HIT: the Matcher already pinned the capability (§A.10 / ─
    # §I). HIT -> Argument Path Router DIRECTLY: NO re-selection — no Recall, no
    # Candidate Aggregation, no cap_router. It short-circuits BEFORE candidate
    # assembly so none of those nodes runs. ``backend=off`` keeps the legacy HIT
    # hop (select_and_extract) further below.
    backend = settings.chat_cap_router_backend
    if backend != cap_router.BACKEND_OFF and mres.state == MATCH_HIT:
        trace["stage"] = "capability"
        entry = entries_by_id.get(mres.capability_id)
        if entry is None:  # a HIT the active table no longer honors: refuse
            trace["fallback"] = REASON_VERSION_MISMATCH
            return None
        return _acquisition_hop(requirements, deps, entry, facts=facts,
                                view=view, trace=trace, capture=capture)

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

    # ── Node 2: capability SELECTION ────────────────────────────────────────────
    # Phase 4 split (2026-10-01): the NEW lane (backend != off) serves ONLY the
    # MISS/AMBIGUOUS selection here:
    #
    #     MISS/AMBIGUOUS -> Recall -> Aggregation -> K normalization
    #                    -> cap_router (stub | laya) -> ONE | NONE
    #
    # ... then the decided capability feeds the Argument Path Router (below). A
    # MATCH_HIT on the new lane already returned above (HIT -> ARP). The ``else``
    # is the backend=off compatibility/rollback lane: the FULL legacy single-call
    # ToolIntentModel hop for HIT *and* MISS/AMBIGUOUS, byte-identical (K
    # normalization is a new-lane business step and never runs on that lane).
    if backend != cap_router.BACKEND_OFF:
        trace["stage"] = "cap_router"
        selector = cap_router.selector_for(backend)
        if selector is None:  # unknown backend: never falls through
            trace["fallback"] = REASON_CAP_ROUTER_UNAVAILABLE
            return None
        # Business-layer K normalization (§26.2) — the SAME rule for stub and
        # laya: the selector only ever sees a list of at most K candidates, and a
        # K=1 turn is executed by the business layer directly (the selector is
        # never consulted, so no model is called and no confidence is invented).
        # Applied AFTER the backend is resolved, so an unknown backend can never
        # be laundered into a K=1 direct selection. backend=off never reaches here.
        norm = candidate_aggregation.normalize_top_k(cands)
        if norm.direct is not None:
            route = cap_router.CapabilityRoute(
                cap_router.ROUTE_SELECTED, norm.direct.capability_id,
                provenance="K=1 business-layer direct selection")
        else:
            try:
                route = await selector.select(message, list(norm.candidates),
                                              entries_by_id=entries_by_id, facts=facts)
            except cap_router.CapabilityRouterUnavailable:
                # ruling 2026-10-01: a selector that cannot serve exits to the
                # Agent; selection NEVER falls back to Qwen (the metric would be
                # polluted).
                trace["fallback"] = REASON_CAP_ROUTER_UNAVAILABLE
                return None
        trace["cap_router"] = f"{route.decision}:{route.capability_id or '-'}"
        if capture is not None:
            capture["cap_router"] = {
                "decision": route.decision, "capability_id": route.capability_id,
                "confidence": route.confidence, "provenance": route.provenance,
            }
        if not route.selected:
            trace["fallback"] = REASON_CAP_ROUTER_NONE
            return None
        # capability decided -> the Argument Path Router decides HOW its
        # arguments are acquired (CONTEXT_DIRECT reuses the Binder; a MODEL need
        # is not yet executable and exits to the Agent).
        entry = entries_by_id.get(route.capability_id)
        if entry is None:  # a route the active table no longer honors: refuse
            trace["fallback"] = REASON_VERSION_MISMATCH
            return None
        return _acquisition_hop(requirements, deps, entry, facts=facts,
                                view=view, trace=trace, capture=capture)

    # backend=off compatibility/rollback lane: the legacy single-call
    # ToolIntentModel selection hop, UNCHANGED (HIT and MISS/AMBIGUOUS alike).
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
    entry = entries_by_id.get(jv.capability_id)
    if entry is None:  # a verdict the active table no longer honors: refuse
        trace["fallback"] = REASON_VERSION_MISMATCH
        return None
    return _certify(requirements, entry, jv.arguments, facts=facts, view=view,
                    trace=trace, capture=capture)


def _certify(requirements, entry, args, *, facts, view, trace, capture):
    """Capability -> kind gate -> Binder validate -> certified ACTION metadata.

    Shared by the legacy lane and the ARP's CONTEXT_DIRECT. ``args`` is the
    argument DRAFT (legacy: the model's extraction; ARP: the legal system values
    supplied via ``acquisition_inputs``) — the Binder still resolves context
    slots from the turn facts and is the final gate (never modified here)."""
    from . import binder

    if not policy.kind_enabled(entry.intent_kind):  # P3: in the table, but not ON
        trace["fallback"] = REASON_KIND_DISABLED
        return None
    if capture is not None:
        capture["entry"] = {"intent_kind": entry.intent_kind,
                            "tool_binding": entry.tool_binding}
    trace["stage"] = "binder"
    bound = binder.validate(entry, args, facts)
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


def _acquisition_hop(requirements, deps, entry, *, facts, view, trace, capture):
    """Argument Path Router hop for a DECIDED capability (Phase 4 Step 2/3).

    The capability is already pinned (Matcher HIT or cap_router SELECTED); this
    node decides HOW its arguments are acquired — NEVER re-selecting it. The ARP
    inputs come from the injected seam (``deps.acquisition_inputs``); absent →
    the capability is acquisition-undeclared → Agent (§G).

    CONTEXT_DIRECT reuses the existing Binder/certified handoff, consuming ONLY
    the supplied system values (no merge, no new validation). MIXED (Step 3)
    merges the system values with the injected MODEL values (§B/§F) and hands the
    merged draft to the SAME Binder/certified handoff. The MODEL strategies that
    need Qwen (QUERY_TO_QWEN / QUERY_PLUS_5_USER_TURNS) are not wired and exit to
    the Agent. MISSING (no legal value) likewise exits to the Agent."""
    from .argument_acquisition import path_router
    from .argument_acquisition.contract import (
        STRATEGY_CONTEXT_DIRECT,
        STRATEGY_MISSING,
        STRATEGY_MIXED,
    )

    provider = getattr(deps, "acquisition_inputs", None)
    inputs = provider(entry.capability_id) if callable(provider) else None
    declaration = dict(inputs.declaration) if inputs else {}
    evidence = dict(inputs.evidence) if inputs else {}
    system_values = dict(inputs.system_values) if inputs else {}
    system_sources = dict(inputs.system_sources) if inputs else {}
    model_values = dict(inputs.model_values) if inputs else {}
    model_source = str(inputs.model_source or "") if inputs else ""
    decision = path_router.route(entry.parameters, declaration,
                                 evidence=evidence, system_values=system_values)
    if capture is not None:
        capture["acquisition"] = {
            "declared": decision.declared, "strategy": decision.strategy,
            "model_slots": list(decision.model_slots),
            "system_slots": list(decision.system_slots),
            "bundle_source": decision.bundle_source,
            "ready": decision.readiness.ready,
            "needs_acquisition": decision.readiness.needs_acquisition,
            "unsatisfiable": list(decision.readiness.unsatisfiable),
        }
    trace["acquisition"] = decision.strategy or "UNDECLARED"
    if not decision.declared:
        trace["fallback"] = REASON_ACQUISITION_UNDECLARED
        return None
    if decision.strategy == STRATEGY_MISSING:
        trace["fallback"] = REASON_ACQUISITION_MISSING
        return None
    if decision.strategy == STRATEGY_CONTEXT_DIRECT:
        # All required slots are deterministically ready and no MODEL acquisition
        # is needed. Consume ONLY the legal system values supplied via the seam;
        # the existing Binder resolves context slots and validates.
        return _certify(requirements, entry, system_values, facts=facts,
                        view=view, trace=trace, capture=capture)
    if decision.strategy == STRATEGY_MIXED:
        # MIXED (Step 3): the system side and the MODEL side merge into ONE draft
        # (§B) with per-slot provenance; the SAME existing Binder then validates
        # it — the system value wins on any collision and provenance records the
        # ACTUAL source each side was resolved from.
        from .argument_acquisition import merge as merge_mod

        merged = merge_mod.merge(system_values, model_values,
                                 declaration=declaration,
                                 system_sources=system_sources,
                                 model_source=model_source)
        if capture is not None:  # telemetry only; the action schema is unchanged
            capture["acquisition"]["provenance"] = [
                {"slot": p.slot, "source": p.source, "produced_by": p.produced_by}
                for p in merged.provenance
            ]
        return _certify(requirements, entry, merged.args, facts=facts,
                        view=view, trace=trace, capture=capture)
    # QUERY_TO_QWEN / QUERY_PLUS_5_USER_TURNS: a MODEL acquisition need whose
    # Qwen extractor is not wired -> Agent.
    trace["fallback"] = REASON_ACQUISITION_MODEL_PENDING
    return None


def certified(requirements, entry, args, registry_fp, *,
              stage: str, integrity: str | None = None) -> TurnRequirements:
    """Certified turn: the same construction shape as the legacy branches
    (only action fields set; source facts ride through). ``funnel_registry_version``
    carries the LIVE Registry content fingerprint — the executor's TOCTOU
    re-validation (8.9) checks against the same fingerprint (migration 0014:
    single namespace, the legacy index-version stamp is gone)."""
    # Runtime import (kept out of module top): understanding imports actions,
    # whose moved-name façade imports this package — a top-level import would
    # close the cycle. See the package __init__ note.
    from core.application.chat.understanding import (
        Complexity,
        Confidence,
        Signal,
        TurnRequirements,
    )

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
        # Web demand RIDES THROUGH (E2E-matrix ruling 2026-09-27): the entry veto
        # no longer blanket-refuses web turns, so the composite guard lives in
        # execution_plan._is_action_eligible, which needs the turn's true
        # needs_web to refuse "新建文件夹并查新闻"-style half-certifications.
        needs_web=requirements.needs_web,
        private_only=requirements.private_only,
        external_ok=requirements.external_ok,
    )
