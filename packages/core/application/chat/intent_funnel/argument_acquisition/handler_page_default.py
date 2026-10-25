"""Handler page default — the viewer's current page as a ``pages`` candidate.

ONE responsibility: derive the UI handler's default page value for a file
capability from settled turn facts. This is a deterministic VALUE PRODUCER, not
a sentence parser: it reads only the fact the client settled upstream, never
inspects the user's sentence, never calls a model and performs no I/O.

The approved acquisition rule (handler value -> LLM -> merge):

* the viewer must be showing the SAME document as the target (``asset_id``); a
  page belonging to another document is never reused;
* only the current page is used — the visible-range fields are deliberately NOT
  consulted (the rule names the current page, and a visible range is not the
  same thing as the pages the user asked to read);
* a missing / non-positive / non-numeric current page yields ``None`` (no
  value), never a fabricated one.
"""
from __future__ import annotations


def _positive_page(raw: object) -> int | None:
    """A 1-based page number, or ``None``. Booleans are not page numbers."""
    if raw is None or isinstance(raw, bool):
        return None
    try:
        page = int(raw)
    except (TypeError, ValueError):
        return None
    return page if page > 0 else None


def handler_page_default(facts, *, asset_id: str) -> str | None:
    """The viewer's current page as a tool-ready ``pages`` spec, or ``None``.

    ``asset_id`` is the already-resolved target document. The default is only
    produced when the viewer reports a valid page AND the viewer's asset is the
    SAME document; otherwise the handler has no value (the callers then follow
    their own missing/failure path).
    """
    if facts is None:
        return None
    target = str(asset_id or "").strip()
    if not target:
        return None
    viewer_asset = str(getattr(facts, "viewer_asset_id", "") or "").strip()
    if not viewer_asset or viewer_asset != target:
        return None
    page = _positive_page(getattr(facts, "viewer_current_page", None))
    return str(page) if page is not None else None
