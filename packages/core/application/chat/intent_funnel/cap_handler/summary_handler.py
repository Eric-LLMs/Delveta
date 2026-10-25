"""SummaryHandler — argument acquisition for ``cap-summary``.

``cap-summary`` (tool binding ``summary_gen``) does NOT run the shared Qwen
extractor and does NOT run a page-scope model. The handler certifies only two
deterministic outcomes:

* a literal workspace path in the user's own sentence ("总结 ./reports/q3.md")
  becomes ``path`` — fully specified by the sentence, no id invented. A path
  that follows a save cue ("保存到 reports/q3.md") is a DESTINATION, not a
  source, so that turn keeps the ``auto`` marker and the tool validates the
  same literal text as its save target;
* everything else becomes the explicit ``auto`` marker. The ``summary_gen`` tool
  then runs the two-stage flow: a first LLM call decides WHICH source to
  summarize (the previous message / the open document and pages / the whole
  session), whether it can already answer, and the save intent; a second call
  runs ONLY when that decision needs material the first call did not have.

The Binder still injects ``asset_id`` from turn facts into any schema that
declares it, so the open document's identity rides the action WITHOUT being a
model output. ``auto`` is what tells the tool "the source was not decided
upstream — decide it from the turn"; an Agent-lane call that names an explicit
source stays on the deterministic path.

No ``slot_plan`` exists, so the orchestrator never reaches the extractor and
never asks for an output directory. Save behaviour is decided inside the tool's
first call and carried through the flow (see ``toolkit/plugins.py``).
"""
from __future__ import annotations

from .acquisition_failure import (
    CATEGORY_NO_CONTENT,
    STAGE_CONTENT_SELECTION,
    AcquisitionFailure,
    turn_context,
)
from .path_token import is_destination, literal_path
from .scope import resolve_asset_id


class SummaryHandler:
    """``cap-summary`` parameter handler (stateless)."""

    capability_id = "cap-summary"

    async def acquire(self, *, query: str, facts) -> dict[str, object] | AcquisitionFailure:
        """Return the deterministic draft, or an honest "nothing to summarize".

        A literal workspace path wins only when the turn carries no document
        identity (the same precedence the old handler used). Every other
        non-empty request is the ``auto`` marker: the tool's first LLM call owns
        the source decision.
        """
        message = str(query or "").strip()
        if not message:
            return AcquisitionFailure(
                category=CATEGORY_NO_CONTENT,
                stage=STAGE_CONTENT_SELECTION,
                detail="the request is empty, so no content source can be selected",
                context=turn_context(facts),
            )
        asset_id = resolve_asset_id(facts)
        path = None if asset_id else literal_path(message)
        if path is not None and is_destination(message, path):
            # A path AFTER a save/export cue is where the result is WRITTEN, not
            # what to read. The source stays the tool's first-call decision, and
            # the tool validates the same literal path as its save target.
            path = None
        if path is not None:
            return {"path": path}
        return {"auto": "1"}
