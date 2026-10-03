"""Funnel orchestration — the one active control flow over the nodes.

Registry → Matcher → (Recall on MISS/AMBIGUOUS) → Candidate Aggregation →
ToolIntentModel (ONE call) → Binder → certified TurnRequirements, every
non-COMPLETE outcome exiting to the Agent byte-identically (8.10).

Phase 4 seam (2026-10-01): when ``chat_cap_router_backend != off`` the decided
capability — a MATCH_HIT directly, or a MISS/AMBIGUOUS via ``Recall ->
Aggregation -> K normalization -> cap_router -> SELECTED|REJECT|NONE`` — feeds
the **Argument Path Router** (:mod:`.argument_acquisition.path_router`), which
derives the acquisition strategy from the capability's declaration + injected
inputs. A MATCH_HIT is NEVER re-selected (no Recall, no Aggregation, no
cap_router). CONTEXT_DIRECT reuses the existing Binder/certified handoff; a
MODEL acquisition need (QUERY_TO_EXTRACTOR / QUERY_PLUS_5TURNS_TO_EXTRACTOR /
MIXED) is executed through the injected extractor. Every non-certified outcome
(REJECT, NONE, K<3 ineligibility, an acquisition-undeclared or -MISSING
capability, a selector that cannot serve) exits to the Agent. With the default
``off`` every turn takes the fully-legacy hop, byte-identical.
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
    REASON_CAP_ROUTER_INELIGIBLE,
    REASON_CAP_ROUTER_NONE,
    REASON_CAP_ROUTER_REJECT,
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


def _mark(capture: dict | None, key: str, t0: float) -> None:
    """Record a per-stage duration (ms) into the capture seam. A no-op when
    capture is off, so the production lane stays byte-identical; ``timings``
    rides the trace_json blob (a JSONB key, never a new column)."""
    if capture is None:
        return
    import time

    capture.setdefault("timings", {})[key] = int((time.monotonic() - t0) * 1000)


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
    import time

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
    # ── Argument-acquisition seam (turn-scoped) ──────────────────────────────────
    # The production provider is built ONCE here — it needs the live
    # ``entries_by_id``, the query and the turn facts — and only when the deps
    # seam supplies none (tests inject their own provider). ``history`` is the
    # resolved turn context's history; it is read by the extractor's context
    # bundle for the QUERY_PLUS_5TURNS strategy and never fabricated.
    history = list(getattr(ctx, "history", None) or [])
    acq_provider = getattr(deps, "acquisition_inputs", None)
    if acq_provider is None:
        from .argument_acquisition.provider import for_turn as _acq_for_turn

        acq_provider = _acq_for_turn(entries_by_id=entries_by_id,
                                     query=message, facts=facts)

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
        _t_acq = time.monotonic()
        out = await _acquisition_hop(requirements, deps, entry, query=message,
                                     facts=facts, view=view, trace=trace,
                                     capture=capture, provider=acq_provider,
                                     history=history)
        _mark(capture, "acquisition_ms", _t_acq)
        return out

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
    #                    -> cap_router (stub | cap_router) -> SELECTED | REJECT | NONE
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
        # Business-layer K normalization (§26.2) — the SAME rule for every
        # backend: the selector only ever sees a list of at most K candidates, and
        # a K=1 turn is executed by the business layer directly (the selector is
        # never consulted, so no model is called and no confidence is invented).
        # Applied AFTER the backend is resolved, so an unknown backend can never
        # be laundered into a K=1 direct selection. backend=off never reaches here.
        norm = candidate_aggregation.normalize_top_k(cands)
        # V2 4-slot HARD invariant accounting: the true slot count the model is
        # (or would be) handed — capability cards + the frozen REJECT card. Only
        # a 3-capability candidate set is V2-eligible (3 + REJECT = 4 slots).
        # K=1 is business-direct (0 slots handed over); K=2 is V2-INELIGIBLE and
        # must NEVER be padded with a fake 3rd capability nor sent as a 3-slot
        # payload. K<3 is a degradation metric only — the frozen V2 training
        # distribution is never touched.
        option_slots = 0 if norm.direct is not None else len(norm.candidates) + 1
        v2_ineligible_reason = None
        if norm.direct is not None:
            v2_ineligible_reason = "K=1"
        elif len(norm.candidates) < 3:
            v2_ineligible_reason = "K<3"
        trace["option_slots"] = option_slots
        if v2_ineligible_reason is not None:
            trace["v2_ineligible_reason"] = v2_ineligible_reason
        if norm.direct is not None:
            route = cap_router.CapabilityRoute(
                cap_router.ROUTE_SELECTED, norm.direct.capability_id,
                provenance="K=1 business-layer direct selection")
        else:
            if len(norm.candidates) < 3:
                # V2-INELIGIBLE degradation: exit to the Agent via the same
                # fail-open path an unavailable service uses. No model call, no
                # fabricated card — the turn is recorded as degraded, not served.
                if capture is not None:
                    capture["cap_router"] = {
                        "decision": None, "capability_id": None, "confidence": None,
                        "provenance": "", "option_slots": option_slots,
                        "v2_ineligible_reason": v2_ineligible_reason,
                    }
                trace["cap_router"] = f"INELIGIBLE:{len(norm.candidates)}"
                trace["fallback"] = REASON_CAP_ROUTER_INELIGIBLE
                return None
            try:
                _t_select = time.monotonic()
                route = await selector.select(message, list(norm.candidates),
                                              entries_by_id=entries_by_id, facts=facts)
                _mark(capture, "selection_ms", _t_select)
            except cap_router.CapabilityRouterUnavailable:
                # ruling 2026-10-01: a selector that cannot serve exits to the
                # Agent; selection NEVER falls back to another model (the metric
                # would be polluted).
                trace["fallback"] = REASON_CAP_ROUTER_UNAVAILABLE
                return None
        trace["cap_router"] = f"{route.decision}:{route.capability_id or '-'}"
        if capture is not None:
            capture["cap_router"] = {
                "decision": route.decision, "capability_id": route.capability_id,
                "confidence": route.confidence, "provenance": route.provenance,
                "option_slots": option_slots, "v2_ineligible_reason": v2_ineligible_reason,
            }
        if not route.selected:
            # REJECT is the NORMAL 4th decision → the system's REAL no-capability
            # path (Agent). NONE is a separate, distinct outcome. Neither ever
            # enters the argument chain.
            trace["fallback"] = (REASON_CAP_ROUTER_REJECT if route.rejected
                                 else REASON_CAP_ROUTER_NONE)
            return None
        # capability decided -> the Argument Path Router decides HOW its
        # arguments are acquired (CONTEXT_DIRECT reuses the Binder; a MODEL need
        # is executed through the injected extractor).
        entry = entries_by_id.get(route.capability_id)
        if entry is None:  # a route the active table no longer honors: refuse
            trace["fallback"] = REASON_VERSION_MISMATCH
            return None
        _t_acq = time.monotonic()
        out = await _acquisition_hop(requirements, deps, entry, query=message,
                                     facts=facts, view=view, trace=trace,
                                     capture=capture, provider=acq_provider,
                                     history=history)
        _mark(capture, "acquisition_ms", _t_acq)
        return out

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

    import time

    if not policy.kind_enabled(entry.intent_kind):  # P3: in the table, but not ON
        trace["fallback"] = REASON_KIND_DISABLED
        return None
    if capture is not None:
        capture["entry"] = {"intent_kind": entry.intent_kind,
                            "tool_binding": entry.tool_binding}
    trace["stage"] = "binder"
    _t_bind = time.monotonic()
    bound = binder.validate(entry, args, facts)
    _mark(capture, "binding_ms", _t_bind)
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


async def _acquisition_hop(requirements, deps, entry, *, query, facts, view, trace,
                           capture, provider=None, history=()):
    """Argument Path Router hop for a DECIDED capability (Phase 4 Step 2/3).

    The capability is already pinned (Matcher HIT or cap_router SELECTED); this
    node decides HOW its arguments are acquired — NEVER re-selecting it. The ARP
    inputs come from the injected seam (``acquisition_inputs``) when a test
    supplies one, else from the production provider built per turn; an absent /
    empty declaration → acquisition-undeclared → Agent (§G).

    CONTEXT_DIRECT reuses the existing Binder/certified handoff, consuming ONLY
    the supplied system values (no merge, no new validation). The MODEL
    strategies (QUERY_TO_EXTRACTOR / QUERY_PLUS_5TURNS_TO_EXTRACTOR / MIXED) run
    the injected ``argument_extractor`` over the sanctioned context bundle — the
    extracted values ARE the draft for the two QUERY strategies, while MIXED
    merges them with the system side (§B/§F) before the SAME Binder/certified
    handoff. MISSING (no legal value) and an unavailable extractor exit to the
    Agent."""
    from .argument_acquisition import context_bundle, path_router
    from .argument_acquisition.contract import (
        STRATEGY_CONTEXT_DIRECT,
        STRATEGY_MISSING,
        STRATEGY_MIXED,
        STRATEGY_QUERY_PLUS_5TURNS_TO_EXTRACTOR,
        STRATEGY_QUERY_TO_EXTRACTOR,
    )
    from .argument_acquisition.extractor import EXTRACTOR_NAME, ExtractionUnavailable

    import time

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
            # diagnostic: the values each side actually supplied (JSONB blob —
            # observability only, never read back by the request path)
            "system_values": dict(system_values),
            "system_sources": dict(system_sources),
            "model_values": dict(model_values),
            "model_source": model_source,
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

    _MODEL_STRATEGIES = (STRATEGY_QUERY_TO_EXTRACTOR,
                         STRATEGY_QUERY_PLUS_5TURNS_TO_EXTRACTOR, STRATEGY_MIXED)
    if decision.strategy in _MODEL_STRATEGIES:
        if not model_values:
            # No MODEL values injected (test seam) -> run the REAL extractor over
            # the sanctioned bundle (query alone, or query + last-5 user turns).
            extractor = getattr(deps, "argument_extractor", None)
            if extractor is None:
                trace["fallback"] = REASON_ACQUISITION_MODEL_PENDING
                return None
            bundle = context_bundle.build(query, history=history,
                                          source=decision.bundle_source)
            try:
                _t_extract = time.monotonic()
                model_values, model_source = await extractor(
                    query=query, entry=entry,
                    model_slots=decision.model_slots, bundle=bundle)
                _mark(capture, "extraction_ms", _t_extract)
            except ExtractionUnavailable:
                # Honest fail-closed: no fabricated value; the Agent owns the
                # clarification.
                trace["fallback"] = REASON_ACQUISITION_MODEL_PENDING
                return None
            if capture is not None:
                capture["acquisition"]["extractor"] = EXTRACTOR_NAME
        model_values = dict(model_values or {})
        model_source = str(model_source or "")
        if capture is not None:
            # the ACTUAL extracted MODEL values (post-extractor), for diagnostics
            capture["acquisition"]["model_values"] = dict(model_values)
            capture["acquisition"]["model_source"] = model_source
        if decision.strategy == STRATEGY_MIXED:
            # MIXED: the system side and the MODEL side merge into ONE draft (§B)
            # with per-slot provenance; the SAME existing Binder then validates it
            # — the system value wins on any collision and provenance records the
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
        # QUERY_TO_EXTRACTOR / QUERY_PLUS_5TURNS_TO_EXTRACTOR: the extracted MODEL
        # values ARE the draft; the Binder validates them.
        return _certify(requirements, entry, model_values, facts=facts,
                        view=view, trace=trace, capture=capture)
    # MISSING handled above; any other label is not a MODEL acquisition need.
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
