"""CreateFolderHandler — argument acquisition for ``cap-create-folder``.

Owns ONLY this capability's required schema slot, ``name``. ``cap-create-folder``
(tool binding ``create_folder``) creates ONE folder in the caller's cloud drive
(``DriveService.create_folder`` — My Drive root unless an internal parent path is
supplied). There is no asset model and no path plane on this capability: the
value is a single folder NAME, never a filesystem path.

Per the acquisition contract, ``name`` is a MODEL-owned natural-language slot and
this handler runs NO extraction rule (no quoted / ``叫`` / ``named`` / ``for the
X`` / ``X 文件夹`` regex ladder — that was implementation with no requirement
basis):

* the MODEL extracts the literal folder name from the user's sentence (quotes
  dropped, casing and spaces kept). A name the model does not resolve honestly
  lands MISSING — the Binder gate owns it; the model is never allowed to
  fabricate a name.

``acquire()`` returns the EMPTY ``{}`` draft (never ``None`` unless the query is
blank) so the model is authorized to extract ``name``. The model is authorized
for ``name`` ONLY: it must NEVER guess a root or a path (the optional
``parent_path`` stays tool-owned, defaulted to the drive root), and its value is
folded only after ``_certify``'s SAME kind gate + Binder validate.

``parent_path`` is NEVER emitted and NEVER asked of the model: the public
capability contract is the single ``name`` slot (``create_folder``'s optional
``parent_path`` is tool-owned, so the folder is created under the drive root — the
system default), so the draft is always exactly ``{}``.

``facts`` is accepted for the common handler contract but ignored ON PURPOSE — the
name comes from the sentence, never from the turn's asset context.
"""
from __future__ import annotations

from .slot_plan import SlotPlan


class CreateFolderHandler:
    """``cap-create-folder`` parameter handler (stateless).

    ``facts`` is accepted for the common handler contract but ignored ON PURPOSE:
    the name must come from the sentence, never from the turn's asset context.
    """

    capability_id = "cap-create-folder"

    async def acquire(self, *, query: str, facts) -> dict[str, object] | None:
        """Return the EMPTY ``{}`` draft (never a rule-extracted name) so the
        model is authorized to extract ``name``. Only a blank query returns
        ``None`` (no input at all)."""
        message = str(query or "").strip()
        if not message:
            return None
        return {}

    def slot_plan(self, *, query: str, facts, draft: dict) -> SlotPlan:
        """Authorize the model for ``name`` (the single slot). The model never
        guesses a root or path — an unresolved name stays MISSING for the Binder
        gate."""
        return SlotPlan(model_slots=("name",))
