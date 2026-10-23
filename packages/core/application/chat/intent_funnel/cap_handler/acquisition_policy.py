"""Per-capability acquisition policy — the ONE explicit strategy declaration.

The orchestrator dispatches argument acquisition by ``capability_id`` through
THIS table. It never INFERS a strategy from ``parameters == {}``, from a missing
handler, or from "the slot is not in the draft": those inferences conflate four
different business states that this module keeps apart:

``NO_ARGS``          the capability genuinely takes no arguments (the Registry
                     schema is empty AND the real tool requires none). The turn
                     certifies ``{}``.
``RULE_CONTEXT``     a capability handler resolves EVERY slot deterministically
                     from trusted turn context, the viewer, an attachment or an
                     explicit rule in the user's own sentence. No model call.
``MODEL``            a capability handler owns the capability; the shared
                     extractor fills ONLY the MODEL-owned slots the handler
                     authorizes, and the handler draft is the protected /
                     fallback side.
``MIXED``            the handler has BOTH authoritative deterministic slots
                     (trusted turn context / an explicit structured token in
                     the sentence) AND MODEL-owned slots. The draft is kept,
                     only the remaining authorized slots are extracted, and a
                     deterministic value is never overwritten.
``DEFERRED_AGENT``   the capability is ADMITTED by the funnel but its REAL tool
                     contract needs an input with no legal producer on the
                     funnel path. The turn exits to the Agent with its OWN
                     reason (never silently certified, never given a fabricated
                     argument).

A capability absent from this table has NO declared policy; the orchestrator
reports ``ACQUISITION_UNDECLARED`` (policy not declared) unless an injected
provider explicitly declares the capability for the generic Argument Path Router.

This module maps and documents only: no acquisition logic, no handler
implementation, no tool execution.

The toolkit trio (``cap-summary`` / ``cap-mindmap`` / ``cap-slides``) is
DEFERRED_AGENT on the verified contract gap in ``_TOOLKIT_DEFERRED_DETAIL``:
the real ``summary_gen`` / ``mindmap_gen`` / ``slides_gen`` tools REQUIRE
``paths: string[]`` (``minItems: 1``), the live Registry declares no parameters
for them, the chat request carries no selected-workspace-file producer, and the
executor gate (``binder.legacy.validate_action``) accepts string slots only.
No argument may be invented to paper over that gap.
"""
from __future__ import annotations

from dataclasses import dataclass

# ── the declared strategies ─────────────────────────────────────────────────────
STRATEGY_NO_ARGS = "NO_ARGS"
STRATEGY_RULE_CONTEXT = "RULE_CONTEXT"
STRATEGY_MODEL = "MODEL"
STRATEGY_MIXED = "MIXED"
STRATEGY_DEFERRED_AGENT = "DEFERRED_AGENT"

STRATEGIES = frozenset({
    STRATEGY_NO_ARGS,
    STRATEGY_RULE_CONTEXT,
    STRATEGY_MODEL,
    STRATEGY_MIXED,
    STRATEGY_DEFERRED_AGENT,
})


@dataclass(frozen=True)
class AcquisitionPolicy:
    """One capability's acquisition strategy.

    ``detail`` documents WHY a strategy was chosen when the reason is not
    self-evident (today: the verified contract gap behind DEFERRED_AGENT). It is
    diagnostic text only — the runtime never branches on it.
    """

    strategy: str
    detail: str = ""


_TOOLKIT_DEFERRED_DETAIL = (
    "tool requires paths: string[] (minItems 1) with optional output_dir; "
    "slides also accepts count 3-20. The Registry declares no parameters, the "
    "chat request carries no selected-workspace-file source, and validate_action "
    "accepts string slots only. A toolkit-suite resolver is required before this "
    "capability can be acquired on the funnel path."
)

# ── the declaration ─────────────────────────────────────────────────────────────
# Every capability the funnel admits to argument acquisition is listed here.
# Handler-backed capabilities must have a matching entry in ``roster.HANDLERS``;
# the coverage test pins that alignment.
POLICIES: dict[str, AcquisitionPolicy] = {
    # MODEL-owned natural-language slots (handler + shared extractor).
    "cap-add-term": AcquisitionPolicy(STRATEGY_MODEL),
    "cap-create-folder": AcquisitionPolicy(STRATEGY_MODEL),
    "cap-rag-search": AcquisitionPolicy(STRATEGY_MODEL),
    "cap-web-search": AcquisitionPolicy(STRATEGY_MODEL),
    # Deterministic scope (an explicit r/<name> token pins subreddit/platform)
    # PLUS MODEL-owned natural-language slots (query cleaning, a named platform).
    "cap-social-search": AcquisitionPolicy(STRATEGY_MIXED),
    "cap-translate": AcquisitionPolicy(STRATEGY_MODEL),
    # Deterministic acquisition from trusted turn facts / the user's literal text.
    "cap-vision": AcquisitionPolicy(STRATEGY_RULE_CONTEXT),
    "cap-read-document": AcquisitionPolicy(STRATEGY_RULE_CONTEXT),
    "cap-read-file": AcquisitionPolicy(STRATEGY_RULE_CONTEXT),
    "cap-pdf-extract-text": AcquisitionPolicy(STRATEGY_RULE_CONTEXT),
    "cap-pdf-table-to-text": AcquisitionPolicy(STRATEGY_RULE_CONTEXT),
    # Admitted, but the real tool contract cannot be satisfied on this path.
    "cap-mindmap": AcquisitionPolicy(STRATEGY_DEFERRED_AGENT,
                                     _TOOLKIT_DEFERRED_DETAIL),
    "cap-slides": AcquisitionPolicy(STRATEGY_DEFERRED_AGENT,
                                    _TOOLKIT_DEFERRED_DETAIL),
    "cap-summary": AcquisitionPolicy(STRATEGY_DEFERRED_AGENT,
                                     _TOOLKIT_DEFERRED_DETAIL),
}


def policy_for(capability_id: str) -> AcquisitionPolicy | None:
    """The declared acquisition policy for ``capability_id``, or ``None`` when
    the capability has no policy on this lane."""
    return POLICIES.get(str(capability_id or ""))
