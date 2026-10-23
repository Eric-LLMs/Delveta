"""Funnel orchestration — the one active control flow over the nodes.

The production chain has exactly two lanes:

* ``Matcher HIT``   -> Argument Path Router -> Binder -> certified requirements;
  no Recall, no candidate aggregation, no selector. The Matcher already pinned
  the capability, so re-selecting it would be a second, competing decision.
* ``MISS/AMBIGUOUS`` -> Recall -> Candidate Aggregation -> K normalization ->
  cap_router -> Argument Path Router -> Binder -> certified requirements.

The **Argument Path Router** (:mod:`.argument_acquisition.path_router`) derives
the acquisition strategy from the capability's declaration + injected inputs.
CONTEXT_DIRECT reuses the existing Binder/certified handoff; a MODEL acquisition
need (QUERY_TO_EXTRACTOR / QUERY_PLUS_5TURNS_TO_EXTRACTOR / MIXED) is executed
through the injected extractor. Every non-certified outcome (NO_CANDIDATE,
REJECT, NONE, an acquisition-undeclared or -MISSING capability, a selector that
cannot serve) exits to the Agent with the user text byte-identical (8.10).

There is exactly ONE production selector (the deployed cap_router service) and
no backend switch: selection never falls back to a second model. A test injects
its own selector through ``deps.cap_router_selector``.

This module owns sequencing, the wall-clock budget and fail-open classification
ONLY — node algorithms, SQL, prompts, parsing and schema validation live in
their own packages; rollout gating and reason naming live in :mod:`.policy`;
the trace record, its log line and the event row live in :mod:`.observability`.
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
    REASON_ACQUISITION_DEFERRED,
    REASON_ACQUISITION_HANDLER_MISSING,
    REASON_ACQUISITION_MISSING,
    REASON_ACQUISITION_MODEL_PENDING,
    REASON_ACQUISITION_POLICY_CONFLICT,
    REASON_ACQUISITION_UNDECLARED,
    REASON_BIND_AMBIGUOUS,
    REASON_BIND_INVALID,
    REASON_BIND_MISSING,
    REASON_CAP_ROUTER_NONE,
    REASON_CAP_ROUTER_REJECT,
    REASON_CAP_ROUTER_UNAVAILABLE,
    REASON_CASCADE_ERROR,
    REASON_INTENT_FUNNEL_ALLOWLIST_REJECT,
    REASON_KIND_DISABLED,
    REASON_NO_CANDIDATE,
    REASON_RECALL_UNAVAILABLE,
    REASON_REGISTRY_UNAVAILABLE,
    REASON_VERSION_MISMATCH,
    Candidate,
    TurnFacts,
)
from .failure import (
    KIND_ACQUISITION_FAILURE,
    KIND_ADMISSION_REJECT,
    KIND_CONTRACT_MISMATCH,
    KIND_FUNNEL_ABSTAIN,
    STAGE_ACQUISITION,
    STAGE_ADMISSION,
    STAGE_BINDER,
    build_failure,
    missing_required_parameters,
    publish_failure,
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


def _present(raw: object) -> bool:
    """A value the acquisition layer would actually carry (presence only)."""
    if raw is None:
        return False
    if isinstance(raw, str):
        return bool(raw.strip())
    return raw != ""


def _handler_arg_sources(draft, final, *, model_slots, fallback_slots,
                         model_values, strategy) -> dict[str, str]:
    """Per-slot provenance for the handler lane (reporting only).

    ``model`` when the final value came from the authorized extractor, including
    a declared fallback slot it replaced; ``query_fallback`` when the verbatim
    sentence survived; ``context`` for a RULE_CONTEXT handler; ``handler`` for a
    deterministic handler value (e.g. the social ``r/<name>`` scope)."""
    from .cap_handler import STRATEGY_RULE_CONTEXT

    draft_values = dict(draft or {})
    final_values = dict(final or {})
    model_values = dict(model_values or {})
    out: dict[str, str] = {}
    for slot in final_values:
        if (
            slot in model_slots
            and slot in model_values
            and _present(model_values[slot])
            and (
                slot not in draft_values
                or not _present(draft_values[slot])
                or slot in fallback_slots
            )
        ):
            out[slot] = "model"
        elif (
            slot in fallback_slots
            and slot in draft_values
            and final_values[slot] == draft_values[slot]
        ):
            out[slot] = "query_fallback"
        elif strategy == STRATEGY_RULE_CONTEXT:
            out[slot] = "context"
        else:
            out[slot] = "handler"
    return out


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
    from .registry.entry import (
        funnel_cardable,
        is_intent_funnel_capability_allowed,
    )

    message = ctx.body.message or ""
    facts = TurnFacts.of(ctx)
    # Structured fallback evidence for THIS turn. Written only by the
    # acquisition / Binder nodes; published below when the turn is not certified.
    failure: dict = {}

    # ── Registry (reads the LIVE table, fingerprint-cached) ──────────────────────
    view = await registry_active_view(session_factory=deps.session_factory)
    if view is None or not view.entries:
        trace["fallback"] = REASON_REGISTRY_UNAVAILABLE  # no capabilities in the table yet
        return None
    trace["registry"] = view.fingerprint
    entries_by_id = {
        e.capability_id: e for e in view.entries
        if is_intent_funnel_capability_allowed(e)
    }
    # Card surface: EVERY routable action capability, admitted or not. The
    # training corpus shows all of them as ordinary options, so withholding a
    # card here would put production out of the model's input distribution and
    # force it to pick a wrong card. Capability admission
    # (``is_intent_funnel_capability_allowed``) is enforced AFTER the selector
    # decides: a pick outside it is rejected before argument extraction.
    card_entries_by_id = {
        e.capability_id: e for e in view.entries if funnel_cardable(e)
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
    # E2 (final semantics): the HIT lane NEVER touches the Recall
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

    # ── MATCH_HIT: the Matcher already pinned the capability (§A.10 / §I). ─────
    # HIT -> Argument Path Router DIRECTLY: NO re-selection — no Recall, no
    # Candidate Aggregation, no cap_router. It short-circuits BEFORE candidate
    # assembly so none of those nodes runs, and no selector is ever consulted.
    if mres.state == MATCH_HIT:
        trace["stage"] = "capability"
        entry = entries_by_id.get(mres.capability_id)
        if entry is None:  # a HIT the active table no longer honors: refuse
            trace["fallback"] = REASON_VERSION_MISMATCH
            return None
        _t_acq = time.monotonic()
        out = await _acquisition_hop(requirements, deps, entry, query=message,
                                     facts=facts, view=view, trace=trace,
                                     capture=capture, provider=acq_provider,
                                     history=history, failure=failure)
        _mark(capture, "acquisition_ms", _t_acq)
        if out is None and failure:
            publish_failure(ctx, failure)
            if capture is not None:
                capture["failure"] = dict(failure)
        return out

    # ── MISS/AMBIGUOUS lane: Recall -> Aggregation -> cap_router (the one
    # convergence point). A MATCH_HIT returned above, so it is never
    # re-selected here.
    # Live-table ruling (RAW lane): every recall hit >= threshold is
    # kept AS IS at the raw stage — no MAX/AVG, no per-capability dedup THERE —
    # a capability may legitimately arrive several times through different
    # sentences. The no-dedup ruling scopes to Raw Recall only: the Capability
    # Candidate Aggregation below then collapses the raw hits into ONE
    # capability-level candidate per capability_id.
    candidates: list[Candidate] = []
    if mres.state == MATCH_AMBIGUOUS:
        for cid in mres.candidates:
            candidates.append(Candidate(cid, 0.0, origin="matcher_ambiguous"))
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
    # Card-surface ruling (single closure point): keep every recall hit the
    # live table can render a card for (``funnel_cardable``). The
    # hidden-capability exclusion is deliberately NOT applied here — those
    # capabilities are cardable so the selector can identify the intent, and
    # capability admission is enforced on its DECISION instead (a pick outside
    # the admitted set becomes INTENT_FUNNEL_ALLOWLIST_REJECT before argument
    # extraction). This matches the LayaChoice input distribution, where every
    # action capability appears as an ordinary option.
    #
    # A capability ABSENT from the live table (the index running ahead of the
    # table) cannot be rendered — the card renderer skips it — so keeping it
    # would only burn a top-K slot and displace a real card. Drop it and
    # record the divergence count. No Recall SQL change: this business-layer
    # filter is the one place that closes the gap for every recall source.
    table_rows = {e.capability_id: e for e in view.entries}
    recall_hits: list[Candidate] = []
    dropped_unregistered = 0
    for hit in rres.candidates:
        row = table_rows.get(hit.capability_id)
        if row is None:
            dropped_unregistered += 1   # ghost: index ahead of the table
        elif funnel_cardable(row):
            recall_hits.append(hit)
        # present but not cardable (e.g. the research lane): not a fault,
        # simply not part of the card surface
    if dropped_unregistered:
        # Index/table divergence is a FAULT, not a business outcome: a row
        # the live table cannot render is a ghost. Refusing the turn keeps
        # the old divergence guard honest (it used to fire only when the
        # selector happened to pick the ghost; with cards now rendered from
        # the table a ghost can never be picked, so the guard must fire here
        # or it would silently disappear).
        trace["recall_dropped_unregistered"] = dropped_unregistered
        trace["fallback"] = REASON_VERSION_MISMATCH
        return None
    if capture is not None:
        # the RAW lane: every candidate recall scored, pre any floor AND pre
        # aggregation — the offline threshold sweep recomputes buckets from
        # THIS list (Phase E); aggregation must never overwrite it.
        capture["recall_raw"] = [
            {"capability_id": c.capability_id, "score": c.score,
             "origin": c.origin, "matched_example": c.matched_example,
             "query_kind": c.query_kind, "language": c.language,
             "query_id": c.query_id}
            for c in recall_hits
        ]
    candidates.extend(recall_hits)
    trace["recall_count"] = len(candidates)
    if candidates:
        top = max(candidates, key=lambda c: c.score)
        trace["recall_top"] = f"{top.capability_id}@{top.score:.3f}"
    # ── Capability Candidate Aggregation (final semantics) ──────────
    # Between Raw Recall and cap_router: group the query-level candidates
    # by capability_id and keep ONE candidate per capability (see
    # :mod:`candidate_aggregation`); the selector sees capability-level cards.
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
    # Empty candidate set -> Agent, no selector hop (ruling, supersedes
    # the "Action Detection is UNCONDITIONAL" note): with no Matcher
    # HIT/AMBIGUOUS card AND no Recall hit at/above the quality gate there is
    # nothing for the single hop to select from. NO_CANDIDATE is back as the
    # honest deepest-stage=recall exit; the byte-identical turn goes to Agent.
    if not cands:
        trace["fallback"] = REASON_NO_CANDIDATE
        return None

    # ── Node 2: capability SELECTION (the ONLY non-HIT lane) ────────────────────
    #     MISS/AMBIGUOUS -> Recall -> Aggregation -> K normalization
    #                    -> cap_router -> SELECTED | REJECT | NONE
    #
    # ... then the decided capability feeds the Argument Path Router (below). A
    # MATCH_HIT already returned above (HIT -> ARP) and never reaches this node.
    # There is exactly ONE production selector (the deployed cap_router
    # service); a test injects its own through ``deps.cap_router_selector``.
    trace["stage"] = "cap_router"
    selector = getattr(deps, "cap_router_selector", None) or cap_router.default_selector()
    # Business-layer K normalization (§26.2): the selector only ever sees a list
    # of at most K candidates. There is NO K=1 direct path and NO K=2
    # ineligibility: EVERY non-empty candidate set (K >= 1) is handed to the
    # selector, which may SELECT one of the shown cards or legitimately REJECT;
    # the business layer never selects a capability on the model's behalf and
    # never fabricates a confidence.
    raw_k = len(cands)
    norm = candidate_aggregation.normalize_top_k(cands)
    effective_k = len(norm.candidates)
    # Variable-length option accounting: the model is handed ``effective_k``
    # capability cards plus the frozen REJECT card appended by the card
    # renderer. The selector is consulted for EVERY K >= 1; there is no
    # special-cased slot count.
    option_slots = effective_k + 1
    trace["raw_k"] = raw_k
    trace["post_gate_k"] = raw_k
    trace["effective_k"] = effective_k
    trace["option_slots"] = option_slots
    if capture is not None:
        # K telemetry recorded BEFORE the selector call, so a turn that
        # times out or finds the service unavailable still exposes exactly
        # what it was handed (raw/effective K, option slots). Overwritten
        # below with the decision once the selector returns.
        capture["cap_router"] = {
            "decision": None, "capability_id": None, "confidence": None,
            "provenance": "", "raw_k": raw_k, "post_gate_k": raw_k,
            "effective_k": effective_k, "option_slots": option_slots,
        }
    try:
        _t_select = time.monotonic()
        route = await selector.select(message, list(norm.candidates),
                                      entries_by_id=card_entries_by_id, facts=facts)
        _mark(capture, "selection_ms", _t_select)
    except cap_router.CapabilityRouterUnavailable:
        # ruling: a selector that cannot serve exits to the
        # Agent; selection NEVER falls back to another model (the metric
        # would be polluted).
        trace["fallback"] = REASON_CAP_ROUTER_UNAVAILABLE
        return None
    trace["cap_router"] = f"{route.decision}:{route.capability_id or '-'}"
    # The event row's ``tool_intent`` column predates the split (migration
    # 0008); the schema is frozen, so the ONE selection verdict is mirrored
    # there too. It is the cap_router decision, never a second model hop.
    trace["tool_intent"] = trace["cap_router"]
    if capture is not None:
        capture["cap_router"] = {
            "decision": route.decision, "capability_id": route.capability_id,
            "confidence": route.confidence, "provenance": route.provenance,
            "raw_k": raw_k, "post_gate_k": raw_k, "effective_k": effective_k,
            "option_slots": option_slots,
        }
    if not route.selected:
        # REJECT is a NORMAL decision → the system's REAL no-capability path
        # (Agent). NONE is a separate, distinct outcome. Neither ever enters
        # the argument chain.
        trace["fallback"] = (REASON_CAP_ROUTER_REJECT if route.rejected
                             else REASON_CAP_ROUTER_NONE)
        return None
    # capability decided -> the Argument Path Router decides HOW its
    # arguments are acquired (CONTEXT_DIRECT reuses the Binder; a MODEL need
    # is executed through the injected extractor).
    entry = card_entries_by_id.get(route.capability_id)
    if entry is None:  # a route the active table no longer honors: refuse
        trace["fallback"] = REASON_VERSION_MISMATCH
        return None
    if not is_intent_funnel_capability_allowed(entry):
        # The selector identified a capability the Intent Funnel must not certify
        # (a hidden capability). Capability admission is the SOLE rule here:
        # terminate the Intent Funnel tool path NOW — no argument extraction, no
        # Binder, no Intent Funnel execution. This is a CORRECT identification, so
        # the turn falls open to the Agent through the NORMAL REJECT exit: the
        # returned requirements stay byte-identical and the Agent's own tool
        # roster / permissions decide what that independent plane may run. The
        # remaining cards are never used as a consolation pick.
        trace["fallback"] = REASON_INTENT_FUNNEL_ALLOWLIST_REJECT
        # Turn-scoped structured context handed to the Agent lane: WHY the Intent
        # Funnel declined to certify the picked capability. The control plane
        # sinks it into ``agent_context`` and the chat prompt renders it, so the
        # Agent understands the funnel abstained — it is not a system-wide ban
        # and it neither grants nor restricts the Agent's own tool surface.
        publish_failure(ctx, build_failure(
            kind=KIND_ADMISSION_REJECT,
            stage=STAGE_ADMISSION,
            reason=REASON_INTENT_FUNNEL_ALLOWLIST_REJECT,
            capability_id=entry.capability_id,
            tool_binding=entry.tool_binding,
            query=message,
            facts=facts,
        ))
        # Explicit audit record: reason + selected capability + original request +
        # routing context. Kept OFF the DB trace's no-query rule; this log line
        # is the operational record for a correct out-of-surface pick.
        logger.warning(
            "Intent Funnel allowlist reject reason=%s capability=%s tool=%s query=%r "
            "candidates=%s registry=%s",
            REASON_INTENT_FUNNEL_ALLOWLIST_REJECT, entry.capability_id,
            entry.tool_binding, message,
            [(c.capability_id, round(c.score, 4)) for c in norm.candidates],
            view.fingerprint,
        )
        return None
    _t_acq = time.monotonic()
    out = await _acquisition_hop(requirements, deps, entry, query=message,
                                 facts=facts, view=view, trace=trace,
                                 capture=capture, provider=acq_provider,
                                 history=history, failure=failure)
    _mark(capture, "acquisition_ms", _t_acq)
    if out is None and failure:
        publish_failure(ctx, failure)
        if capture is not None:
            capture["failure"] = dict(failure)
    return out



def _certify(requirements, entry, args, *, facts, view, trace, capture,
             failure=None, strategy="", arg_sources=None):
    """Capability -> kind gate -> Binder validate -> certified ACTION metadata.

    The ONE certification gate every acquisition strategy exits through.
    ``args`` is the argument DRAFT (whatever the capability handler, the
    Argument Path Router or the extractor produced) — the Binder still resolves
    context slots from the turn facts and is the final gate (never modified
    here). A non-certified exit also writes the structured fallback record the
    Agent receives (``failure`` is the turn-scoped accumulator)."""
    from . import binder

    import time

    if not policy.kind_enabled(entry.intent_kind):  # P3: in the table, but not ON
        trace["fallback"] = REASON_KIND_DISABLED
        if failure is not None:
            failure.update(build_failure(
                kind=KIND_FUNNEL_ABSTAIN, stage=STAGE_BINDER,
                reason=REASON_KIND_DISABLED,
                capability_id=entry.capability_id,
                tool_binding=entry.tool_binding, strategy=strategy,
                args=args, arg_sources=arg_sources,
                required=entry.parameters, facts=facts,
                detail=f"intent kind {entry.intent_kind!r} is not enabled",
            ))
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
        reason = {
            "MISSING": REASON_BIND_MISSING,
            "AMBIGUOUS": REASON_BIND_AMBIGUOUS,
            "INVALID": REASON_BIND_INVALID,
        }.get(bound.state, REASON_BIND_MISSING)
        trace["fallback"] = reason
        if failure is not None:
            diagnosis, prepared = binder.diagnose(entry, args, facts)
            failure.update(build_failure(
                kind=KIND_CONTRACT_MISMATCH, stage=STAGE_BINDER, reason=reason,
                capability_id=entry.capability_id,
                tool_binding=entry.tool_binding, strategy=strategy,
                args=prepared, arg_sources=arg_sources,
                missing=diagnosis.missing,
                invalid=diagnosis.invalid,
                extra=diagnosis.unknown,
                required=entry.parameters, facts=facts,
                detail=f"binder state: {bound.state}",
            ))
        return None
    trace["stage"] = "certified"
    # ``funnel_stage`` is part of the observable ACTION handoff (preview, shadow
    # and existing clients read it); its value is frozen for compatibility.
    return certified(requirements, entry, bound.args, view.fingerprint,
                     stage="tool_intent")


def _certify_capture(requirements, entry, deferred, *, facts, view, trace, capture):
    """Certify a DEFERRED vision ACTION: the capability is decided, the picture is owed.

    The capture request rides ``TurnRequirements.capture_request``; the action carries
    the final ``question`` but NO ``asset_id`` yet — the orchestrator patches it in
    from the client's uploaded capture before the executor runs. The kind gate is
    still enforced here (入表≠开闸 holds for the deferred lane too); the Binder is
    bypassed only because its one missing slot is the picture itself, which the
    capture resolution supplies. The funnel is NOT re-entered on resume.
    """
    from core.application.chat.understanding import (
        Complexity,
        Confidence,
        Signal,
        TurnRequirements,
    )

    if not policy.kind_enabled(entry.intent_kind):  # P3: in the table, but not ON
        trace["fallback"] = REASON_KIND_DISABLED
        return None
    if capture is not None:
        capture["entry"] = {"intent_kind": entry.intent_kind,
                            "tool_binding": entry.tool_binding}
    trace["stage"] = "certified"
    args: dict[str, object] = {}
    if deferred.question:
        args["question"] = deferred.question
    action = {
        "tool": entry.tool_binding, "args": args,
        "capability_id": entry.capability_id,
        "funnel_registry_version": view.fingerprint,
        "funnel_stage": "capture",
        "funnel_kind": entry.intent_kind,
    }
    return TurnRequirements(
        needs_action=Signal.HIGH, requested_action=action,
        capture_request=deferred.request(),
        complexity=Complexity.LOW, confidence=Confidence.HIGH,
        needs_web=requirements.needs_web,
        private_only=requirements.private_only,
        external_ok=requirements.external_ok,
    )


async def _acquisition_hop(requirements, deps, entry, *, query, facts, view, trace,
                           capture, provider=None, history=(), failure=None):
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
    import time

    from .argument_acquisition import context_bundle, path_router
    from .argument_acquisition.contract import (
        STRATEGY_CONTEXT_DIRECT,
        STRATEGY_MISSING,
        STRATEGY_MIXED,
        STRATEGY_QUERY_PLUS_5TURNS_TO_EXTRACTOR,
        STRATEGY_QUERY_TO_EXTRACTOR,
    )
    from .argument_acquisition.extractor import EXTRACTOR_NAME, ExtractionUnavailable
    from .cap_handler import (
        STRATEGY_DEFERRED_AGENT,
        STRATEGY_MODEL,
        STRATEGY_NO_ARGS,
        STRATEGY_RULE_CONTEXT,
        SlotPlanningHandler,
        handler_for,
        policy_for,
    )

    def _acq_fail(reason, *, strategy="", args=None, arg_sources=None,
                  missing=(), invalid=None, detail=""):
        """Record an acquisition exit ONCE: the trace keeps its node code and the
        turn gains the structured record the Agent receives. The record never
        authorizes anything; it is routing context only."""
        trace["fallback"] = reason
        if failure is not None:
            failure.update(build_failure(
                kind=KIND_ACQUISITION_FAILURE, stage=STAGE_ACQUISITION,
                reason=reason, capability_id=entry.capability_id,
                tool_binding=entry.tool_binding, strategy=strategy, query=query,
                args=args, arg_sources=arg_sources, missing=missing,
                invalid=invalid, required=entry.parameters,
                detail=detail, facts=facts,
            ))
        return None

    # ── Per-capability acquisition policy (cap_handler) ──────────────────────────
    # The strategy is DECLARED per capability_id and dispatched explicitly. It is
    # never inferred from ``parameters == {}``: an empty schema may mean NO_ARGS,
    # a hidden required input (DEFERRED_AGENT), or a policy that was never
    # declared (ACQUISITION_UNDECLARED) — three different business outcomes.
    acq_policy = policy_for(entry.capability_id)
    handler = handler_for(entry.capability_id)
    if acq_policy is not None:
        trace["acquisition_policy"] = acq_policy.strategy
        if capture is not None:
            capture["acquisition_policy"] = {
                "strategy": acq_policy.strategy,
                "detail": acq_policy.detail,
            }
        if acq_policy.strategy == STRATEGY_DEFERRED_AGENT:
            # Admitted, but the real tool contract cannot be satisfied from any
            # legal producer on this path. Never fabricate the missing input.
            return _acq_fail(
                REASON_ACQUISITION_DEFERRED, strategy=acq_policy.strategy,
                missing=missing_required_parameters(entry.parameters, {}),
                detail=acq_policy.detail,
            )
        if acq_policy.strategy == STRATEGY_NO_ARGS:
            if dict(entry.parameters or {}) or handler is not None:
                # A NO_ARGS declaration on a capability that DOES publish a
                # parameter schema (or mounts a handler) is a configuration
                # conflict: fail closed instead of bypassing acquisition.
                return _acq_fail(
                    REASON_ACQUISITION_POLICY_CONFLICT,
                    strategy=acq_policy.strategy,
                    detail=("NO_ARGS declared, but the capability publishes "
                            "parameters or mounts a handler"),
                )
            trace["acquisition"] = "NO_ARGS"
            if capture is not None:
                capture["acquisition"] = {
                    "declared": True, "strategy": STRATEGY_NO_ARGS,
                    "args": {},
                }
            return _certify(requirements, entry, {}, facts=facts,
                            view=view, trace=trace, capture=capture,
                            failure=failure, strategy=STRATEGY_NO_ARGS,
                            arg_sources={})
        if acq_policy.strategy not in (STRATEGY_MODEL, STRATEGY_MIXED,
                                       STRATEGY_RULE_CONTEXT):
            return _acq_fail(
                REASON_ACQUISITION_POLICY_CONFLICT, strategy=acq_policy.strategy,
                detail="the declared strategy has no dispatcher",
            )
        if handler is None:
            return _acq_fail(
                REASON_ACQUISITION_HANDLER_MISSING, strategy=acq_policy.strategy,
                missing=missing_required_parameters(entry.parameters, {}),
                detail="the capability declares a handler-backed strategy but no "
                       "handler is mounted",
            )
    elif handler is not None:
        # A handler without a declared policy is a wiring fault: the handler would
        # otherwise choose its own acquisition strategy implicitly.
        return _acq_fail(
            REASON_ACQUISITION_POLICY_CONFLICT,
            missing=missing_required_parameters(entry.parameters, {}),
            detail="a handler is mounted without a declared acquisition policy",
        )

    # ── Per-capability parameter handler (cap_handler) ───────────────────────────
    # A capability wired to a CapabilityHandler owns its own argument acquisition:
    # the handler's ``acquire()`` returns the DET / trusted-context draft, and its
    # OPTIONAL ``slot_plan()`` states, per slot, what still has to happen (§ the
    # four-state model: ACQUIRED / PENDING_MODEL / DEFAULTED / MISSING) — so the
    # caller NEVER infers "needs a model" from ``slot not in draft``. When a plan
    # authorizes >=1 slot the SHARED extractor runs ONCE over the query bundle
    # with the capability's Prompt (the default when it registers none), and only
    # valid, authorized model values are folded into the draft; an empty / invalid
    # / unavailable reply leaves the handler draft (the verbatim query + DET
    # values) intact, so the original request is never lost. The draft — with or
    # without the model's contribution — still passes through the SAME
    # _certify -> Binder -> ActionExecutor -> ToolRuntime handoff below. A handler
    # WITHOUT slot_plan (the asset-anchored vision / read / pdf handlers) plans
    # nothing and calls no model — byte-identical to before. Only a MODEL policy
    # ever reaches the extractor; RULE_CONTEXT stops at the handler draft.
    if handler is not None:
        capture_for = getattr(handler, "capture_for", None)
        deferred = None
        # A VIDEO viewer always needs a client frame: the vision tool cannot read a
        # video file, so a cloud asset id must not win over the capture. For every
        # other kind the asset is tried first (a cloud document is readable server-side).
        if (callable(capture_for)
                and str(getattr(facts, "viewer_kind", "") or "").lower() == "video"):
            deferred = capture_for(query=query, facts=facts)
        args = (None if deferred is not None
                else await handler.acquire(query=query, facts=facts))
        if args is None:
            # Deferred capture: the capability is ALREADY decided (vision) but its
            # picture must come from the PC client (local page / video frame). The
            # turn is certified now with a capture request; the orchestrator pauses
            # and resumes THIS turn once the client supplies the asset. The funnel is
            # never re-run and no argument is extracted a second time.
            if deferred is None and callable(capture_for):
                deferred = capture_for(query=query, facts=facts)
            if deferred is not None:
                trace["acquisition"] = "DEFERRED_CAPTURE"
                if capture is not None:
                    capture["acquisition"] = {
                        "declared": True, "strategy": "DEFERRED_CAPTURE",
                        "handler": type(handler).__name__,
                        "capture": deferred.request(),
                    }
                return _certify_capture(requirements, entry, deferred, facts=facts,
                                        view=view, trace=trace, capture=capture)
            return _acq_fail(
                REASON_ACQUISITION_MISSING, strategy=acq_policy.strategy,
                missing=missing_required_parameters(entry.parameters, {}),
                detail="the capability handler produced no draft and no deferred "
                       "capture is available",
            )
        initial_args = dict(args)
        trace["acquisition"] = "CAP_HANDLER"
        if capture is not None:
            capture["acquisition"] = {
                "declared": True, "strategy": "CAP_HANDLER",
                "policy": acq_policy.strategy,
                "handler": type(handler).__name__, "args": dict(args),
            }
        # ONLY a MODEL / MIXED policy may run the shared extractor. RULE_CONTEXT
        # keeps the deterministic handler draft and calls no model.
        plan = None
        model_values: dict = {}
        if acq_policy.strategy in (STRATEGY_MODEL, STRATEGY_MIXED):
            if not isinstance(handler, SlotPlanningHandler):
                return _acq_fail(
                    REASON_ACQUISITION_POLICY_CONFLICT,
                    strategy=acq_policy.strategy, args=args,
                    arg_sources={slot: "handler" for slot in args},
                    missing=missing_required_parameters(entry.parameters, args),
                    detail=f"{acq_policy.strategy} requires a slot plan, but the "
                           "handler does not implement SlotPlanningHandler",
                )
            plan = handler.slot_plan(query=query, facts=facts, draft=args)
        model_slots = tuple(getattr(plan, "model_slots", ()) or ())
        fallback_slots = tuple(getattr(plan, "fallback_slots", ()) or ())
        if model_slots:
            if capture is not None:
                capture["acquisition"]["slot_plan"] = {
                    "model_slots": list(model_slots),
                    "fallback_slots": list(fallback_slots),
                    "default_slots": list(getattr(plan, "default_slots", ()) or ()),
                }
            extractor = getattr(deps, "argument_extractor", None)
            if extractor is not None:
                from .argument_acquisition import context_bundle
                from .argument_acquisition.prompts import prompt_for
                from .argument_acquisition.slot_refine import (
                    accept_model_values,
                    validation_report,
                )

                bundle = context_bundle.build(query)  # query-only bundle
                try:
                    _t_extract = time.monotonic()
                    model_values, model_source = await extractor(
                        query=query, entry=entry, model_slots=model_slots,
                        bundle=bundle, prompt=prompt_for(entry.capability_id))
                    _mark(capture, "extraction_ms", _t_extract)
                except ExtractionUnavailable:
                    # No fabricated value: the handler draft (verbatim query + DET
                    # slots) stands, and the Binder remains the final gate.
                    model_values, model_source = {}, ""
                model_values = dict(model_values or {})
                args = accept_model_values(args, model_values,
                                           model_slots=model_slots,
                                           parameters=entry.parameters,
                                           fallback_slots=fallback_slots)
                if capture is not None:
                    capture["acquisition"]["extractor"] = EXTRACTOR_NAME
                    capture["acquisition"]["model_values"] = dict(model_values)
                    capture["acquisition"]["model_source"] = str(model_source or "")
                    # Report (never repair) the schema check on the RAW model
                    # values: the executed value is exactly what the model said.
                    capture["acquisition"]["validation"] = validation_report(
                        model_values, model_slots=model_slots,
                        parameters=entry.parameters)
        arg_sources = _handler_arg_sources(
            initial_args, args, model_slots=model_slots,
            fallback_slots=fallback_slots, model_values=model_values,
            strategy=acq_policy.strategy,
        )
        return _certify(requirements, entry, args, facts=facts,
                        view=view, trace=trace, capture=capture,
                        failure=failure, strategy=acq_policy.strategy,
                        arg_sources=arg_sources)

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
        return _acq_fail(
            REASON_ACQUISITION_UNDECLARED, strategy=decision.strategy,
            args=system_values, arg_sources=system_sources,
            missing=missing_required_parameters(entry.parameters, system_values),
            detail="no acquisition declaration exists for this capability",
        )
    if decision.strategy == STRATEGY_MISSING:
        return _acq_fail(
            REASON_ACQUISITION_MISSING, strategy=decision.strategy,
            args=system_values, arg_sources=system_sources,
            missing=(tuple(decision.readiness.unsatisfiable)
                     or tuple(missing_required_parameters(entry.parameters,
                                                          system_values))),
            detail="no legal value exists for a required slot",
        )
    if decision.strategy == STRATEGY_CONTEXT_DIRECT:
        # All required slots are deterministically ready and no MODEL acquisition
        # is needed. Consume ONLY the legal system values supplied via the seam;
        # the existing Binder resolves context slots and validates.
        return _certify(requirements, entry, system_values, facts=facts,
                        view=view, trace=trace, capture=capture,
                        failure=failure, strategy=decision.strategy,
                        arg_sources=system_sources)

    _MODEL_STRATEGIES = (STRATEGY_QUERY_TO_EXTRACTOR,
                         STRATEGY_QUERY_PLUS_5TURNS_TO_EXTRACTOR, STRATEGY_MIXED)
    if decision.strategy in _MODEL_STRATEGIES:
        existing_args = {**dict(model_values), **dict(system_values)}
        model_missing = tuple(
            slot for slot in (decision.model_slots or ())
            if not _present(existing_args.get(slot))
        )
        if not model_values:
            # No MODEL values injected (test seam) -> run the REAL extractor over
            # the sanctioned bundle (query alone, or query + last-5 user turns).
            extractor = getattr(deps, "argument_extractor", None)
            if extractor is None:
                return _acq_fail(
                    REASON_ACQUISITION_MODEL_PENDING,
                    strategy=decision.strategy, args=existing_args,
                    arg_sources={**system_sources,
                                 **{slot: model_source for slot in model_values}},
                    missing=model_missing,
                    detail="the selected acquisition strategy needs a MODEL "
                           "extraction, but no extractor is configured",
                )
            bundle = context_bundle.build(query, history=history,
                                          source=decision.bundle_source)
            try:
                _t_extract = time.monotonic()
                model_values, model_source = await extractor(
                    query=query, entry=entry,
                    model_slots=decision.model_slots, bundle=bundle)
                _mark(capture, "extraction_ms", _t_extract)
            except ExtractionUnavailable as exc:
                # Honest fail-closed: no fabricated value; the Agent owns the
                # clarification.
                return _acq_fail(
                    REASON_ACQUISITION_MODEL_PENDING,
                    strategy=decision.strategy, args=existing_args,
                    arg_sources=system_sources, missing=model_missing,
                    detail=f"the argument extractor could not serve: {exc}",
                )
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
                            view=view, trace=trace, capture=capture,
                            failure=failure, strategy=decision.strategy,
                            arg_sources={p.slot: p.source
                                         for p in merged.provenance})
        # QUERY_TO_EXTRACTOR / QUERY_PLUS_5TURNS_TO_EXTRACTOR: the extracted MODEL
        # values ARE the draft; the Binder validates them.
        return _certify(requirements, entry, model_values, facts=facts,
                        view=view, trace=trace, capture=capture,
                        failure=failure, strategy=decision.strategy,
                        arg_sources={slot: (model_source or "model")
                                     for slot in model_values})
    # MISSING handled above; any other label is not a MODEL acquisition need.
    return _acq_fail(
        REASON_ACQUISITION_MODEL_PENDING, strategy=decision.strategy,
        args=system_values, arg_sources=system_sources,
        missing=missing_required_parameters(entry.parameters, system_values),
        detail=f"strategy {decision.strategy!r} has no acquisition dispatcher",
    )


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
        # Web demand RIDES THROUGH (E2E-matrix ruling): the entry veto
        # no longer blanket-refuses web turns, so the composite guard lives in
        # execution_plan._is_action_eligible, which needs the turn's true
        # needs_web to refuse "新建文件夹并查新闻"-style half-certifications.
        needs_web=requirements.needs_web,
        private_only=requirements.private_only,
        external_ok=requirements.external_ok,
    )
