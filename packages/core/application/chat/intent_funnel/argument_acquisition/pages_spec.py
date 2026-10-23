"""Canonical page-SPEC rendering — the ONE in-memory -> tool-spec conversion.

A page list produced by a deterministic rule or by the online page-scope resolver
must reach a file tool in the SAME grammar :func:`core.infrastructure.pdf_pages.
parse_pages_spec` accepts. This module owns that rendering so both acquisition
lanes cannot drift.

Two input FORMS are preserved, because the user's sentence states one or the
other and the executed argument must not silently rewrite it:

* a RANGE (``第1到3页``) renders as ``"1-3"``;
* a LIST (``第1页和第3页``) renders as ``"1,3"`` — even when the pages happen to
  be contiguous, because ``2,3`` and ``2-3`` are different requests to the user
  even though the tool expands them identically.

Validation is delegated to the strict parser — a canonical spec is ALWAYS one
the tool itself will accept.
"""
from __future__ import annotations

from core.infrastructure.pdf_pages import parse_pages_spec


def canonical_range(start: int, end: int) -> str:
    """Render an inclusive page RANGE as the tools' canonical spec."""
    start, end = int(start), int(end)
    if start <= 0 or end <= 0:
        raise ValueError(f"pages must be 1-based, got {start!r}-{end!r}")
    if start > end:
        raise ValueError(f"reversed page range {start}-{end}: expected ascending bounds")
    spec = str(start) if start == end else f"{start}-{end}"
    parse_pages_spec(spec)
    return spec


def canonical_list(pages) -> str:
    """Render an explicit page LIST as a canonical ascending comma spec."""
    ordered = sorted({int(p) for p in pages})
    if not ordered or any(p <= 0 for p in ordered):
        raise ValueError('pages must be a non-empty 1-based spec, e.g. "2,5"')
    spec = ",".join(str(p) for p in ordered)
    parse_pages_spec(spec)
    return spec


def canonical_pages_spec(raw: str) -> str:
    """Validate an already-written spec with the strict parser, then canonicalize.

    Used by the online resolver, whose reply states a spec rather than a form:
    contiguous pages collapse to a range, everything else stays a comma list.
    """
    pages = parse_pages_spec(raw)
    if len(pages) == 1:
        return str(pages[0])
    if pages == list(range(pages[0], pages[-1] + 1)):
        return f"{pages[0]}-{pages[-1]}"
    return ",".join(str(p) for p in pages)
