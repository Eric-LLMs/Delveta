"""cap_router card rendering — the ``B_noprov`` view the LayaChoice model reads.

LayaChoice-v1 was fine-tuned and benchmarked under ONE frozen capability-card
view, ``B_noprov`` (§26.4): the production card with the ``query examples:``
block and the provenance/``evidence:`` line removed. The model sees, per
candidate, ONLY what the frozen bundle carried:

    ### <capability_id>
    tool: <tool_binding>
    does: <description>
    negative examples:
    - <...>            (or "- (none curated)")
    params:
    - <slot> (<type>, <required|optional>[, max_len=N]): <desc>

It must NOT carry query examples, a provenance/``evidence:`` line, or a
``matched_example:`` line — those were deliberately withheld from training, and a
card that added them would be out of distribution.

Byte-identity is the point, so the schema/negative helpers are IMPORTED from the
production prompt module rather than re-typed: ``_params_block`` renders the
Registry parameter schema, and the negative-example cap is the same constant the
production card applies. This module renders the card and assembles the single
choice question; it never calls a model, a Binder or the Registry.
"""
from __future__ import annotations

from ..tool_intent.prompt import _MAX_NEGATIVE_EXAMPLES, _params_block

# The instructions string the frozen dataset was built and benchmarked under
# (scripts/laya_finetune/data/manifest.json -> "instructions"). It is part of the
# model's input distribution, so it is pinned here verbatim.
INSTRUCTIONS = "Which capability should handle the user's request?"


def render_card(entry) -> str:
    """The ``B_noprov`` card for one Registry entry, byte-for-byte identical to
    the frozen bundle's option text for that capability."""
    lines = [
        f"### {entry.capability_id}",
        f"tool: {entry.tool_binding}",
        f"does: {entry.description}",
    ]
    negatives = [f"- {str(x).strip()}" for x in (entry.negatives or ()) if str(x).strip()]
    negatives = negatives[:_MAX_NEGATIVE_EXAMPLES]
    lines.append("negative examples:\n"
                 + ("\n".join(negatives) if negatives else "- (none curated)"))
    lines.append(_params_block(entry))
    return "\n".join(lines)


def render_choice_question(candidates, entries_by_id: dict) -> dict:
    """The ONE choice question for a candidate set: ``criteria`` maps each
    capability_id (the option label the model answers with) to its ``B_noprov``
    card. Candidates the active table no longer honors are skipped — the model
    can only ever choose a capability it was shown."""
    criteria: dict[str, str] = {}
    for cand in candidates:
        entry = entries_by_id.get(cand.capability_id)
        if entry is None:
            continue
        criteria[entry.capability_id] = render_card(entry)
    return {"type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria}
