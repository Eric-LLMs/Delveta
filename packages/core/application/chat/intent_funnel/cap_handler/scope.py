"""TurnFacts → capability-draft SCOPE — the shared asset + page-window sourcing.

Every file-content handler resolves its "scope" (WHICH asset, WHICH pages) the
same way, so the rules live here exactly once and cannot drift between
capabilities:

* ``asset_id`` — the ONLY legal source is :class:`~..contract.TurnFacts`, with the
  SAME precedence the Binder uses (``attachment_asset_id`` -> ``path_asset_id``
  -> ``viewer_asset_id``). No fact carries one -> ``""`` (the caller fails
  closed; a sentence-copied or model-invented id is never a source).
* ``page_window`` — the viewer's declared page RANGE (``viewer_page_from`` /
  ``viewer_page_to``), NEVER a page number parsed from the user's sentence. Both
  None -> the whole document (the slot is omitted). A complete forward range ->
  ``"3"`` / ``"3-5"``. A HALF-open (only one bound) or reversed range raises
  :class:`PageWindowError`: a page-scoped request must never silently widen to
  the whole document — the caller exits to the Agent instead.

``acquire_file_scope`` is the whole draft rule in one place, so ``read_document``,
``pdf_extract_text`` and ``pdf_table_to_text`` assemble identical arguments.
"""
from __future__ import annotations

# Mirrors binder._CONTEXT_SLOT_SOURCES["asset_id"] exactly (attachment -> path ->
# viewer), so the draft a handler emits is the value the Binder settles on.
_ASSET_FACT_FIELDS = ("attachment_asset_id", "path_asset_id", "viewer_asset_id")


def resolve_asset_id(facts) -> str:
    """The first non-empty asset id among the context facts, else ``""``."""
    if facts is None:
        return ""
    for field in _ASSET_FACT_FIELDS:
        value = str(getattr(facts, field, "") or "").strip()
        if value:
            return value
    return ""


class PageWindowError(ValueError):
    """A viewer page range that cannot be expressed as a legal spec (half-open or
    reversed). The caller fails closed — never a whole-document fallback."""


def page_window(facts) -> str | None:
    """The page spec for this turn's viewer range, or ``None`` to omit the slot.

    ``None`` means the whole document (both bounds unset). A complete forward
    range becomes ``"n"`` (from == to) or ``"from-to"``. A half-open or reversed
    range raises :class:`PageWindowError`.
    """
    start = getattr(facts, "viewer_page_from", None) if facts is not None else None
    end = getattr(facts, "viewer_page_to", None) if facts is not None else None
    if start is None and end is None:
        return None
    if start is None or end is None or start > end:
        raise PageWindowError(
            f"incomplete or reversed viewer page range ({start!r}, {end!r}); a "
            "page-scoped read must not widen to the whole document"
        )
    return str(start) if start == end else f"{start}-{end}"


def acquire_file_scope(facts) -> dict[str, object] | None:
    """The shared file-content argument draft — ``{asset_id[, pages]}`` or ``None``.

    ``None`` (no asset id, or a half-open/reversed page range) is the fail-closed
    signal: the caller exits to the Agent's missing/clarify handling rather than
    guess an asset or widen a page-scoped request to the whole document.
    """
    asset_id = resolve_asset_id(facts)
    if not asset_id:
        return None
    try:
        pages = page_window(facts)
    except PageWindowError:
        return None
    draft: dict[str, object] = {"asset_id": asset_id}
    if pages is not None:
        draft["pages"] = pages
    return draft
