"""Strict 1-based page/slide SPEC parsing — the ONE shared implementation.

A ``pages`` argument (``"3"`` / ``"2,5"`` / ``"1-3"``) is a window over a
page-addressable file. Every file-content tool that accepts one parses it with
THIS function — ``read_document`` (PDF/PPTX), ``pdf_extract_text`` and
``pdf_table_to_text`` (PDF) — so a spec means exactly one thing across tools and
no tool imports another tool's private helper.

The parser is deliberately strict and fail-closed: an empty/blank spec, a
malformed or empty token, a zero/negative page, a reversed range, or a span over
the cap raises :class:`ValueError`. A page-scoped request must never be silently
widened back to the whole document.
"""
from __future__ import annotations

import re

# Hard cap on one page-scoped request: the parser refuses specs above this BEFORE
# building the expanded list, so a pathological spec ("1-1000000000") can never allocate.
MAX_REQUESTED_PAGES = 16

# A pages token is one number or an inclusive range; anything else is a hard parse error.
_PAGE_TOKEN = re.compile(r"^\d+(?:-\d+)?$")


def parse_pages_spec(pages: str) -> list[int]:
    """Strict 1-based page/slide spec parser: ``"3"``, ``"2,5"``, ``"1-3"``, ``" 1, 3-5 , 8 "``.

    Dedup + ascending sort; single-point ranges normalize (``"1-1"`` → ``[1]``). Raises
    :class:`ValueError` on an empty/blank spec, malformed or empty tokens (``"1-"``,
    ``"1,,3"``, ``"abc"``), zero/negative numbers, reversed ranges (``"5-2"``), a single
    span wider than :data:`MAX_REQUESTED_PAGES` (checked BEFORE expansion), or a unique
    page count over the cap (checked incrementally — an oversized list is never built).
    """
    if not pages or not pages.strip():
        raise ValueError('pages must be a non-empty 1-based spec, e.g. "3", "2,5", "1-3"')
    acc: set[int] = set()

    def _add(page: int) -> None:
        acc.add(page)
        if len(acc) > MAX_REQUESTED_PAGES:
            raise ValueError(f"pages spec exceeds the {MAX_REQUESTED_PAGES}-page per-call limit")

    for token in pages.split(","):
        token = token.strip()
        if not token or not _PAGE_TOKEN.match(token):
            raise ValueError(
                f'invalid pages token "{token}": expect "<n>" or "<a-b>" of 1-based page numbers'
            )
        if "-" in token:
            start, end = (int(part) for part in token.split("-", 1))
            if start <= 0 or end <= 0:
                raise ValueError(f"pages must be 1-based, got {token!r}")
            if start > end:
                raise ValueError(f"reversed page range {token!r}: expected ascending bounds")
            if end - start + 1 > MAX_REQUESTED_PAGES:
                raise ValueError(
                    f"page range {token!r} spans {end - start + 1} pages, over the "
                    f"{MAX_REQUESTED_PAGES}-page per-call limit"
                )
            for page in range(start, end + 1):
                _add(page)
        else:
            page = int(token)
            if page <= 0:
                raise ValueError(f"pages must be 1-based, got {token!r}")
            _add(page)
    return sorted(acc)
