"""ReadDocumentHandler — argument acquisition for ``cap-read-document``.

Owns ONLY this capability's parameter preparation; it never loads document bytes
and never parses a PDF/slide — the real file read and extraction stay entirely
inside the shared ``read_document`` tool. The two schema slots of
``cap-read-document``:

* ``asset_id`` — the ONLY legal source is :class:`~..contract.TurnFacts`, resolved
  with the SAME precedence the Binder uses (``attachment_asset_id`` ->
  ``path_asset_id`` -> ``viewer_asset_id``). There is NO fail-open path: when no
  fact carries an asset id the handler returns ``None`` and the turn exits to the
  Agent's existing missing/clarify handling — never a guessed or fabricated id.
* ``pages`` — emitted ONLY on an EXPLICIT page reference in the sentence, then
  normalized to a spec the tool's ``_parse_pages_spec`` accepts (``"3"``,
  ``"2,5"``, ``"1-3"``). The guard is the page WORD itself (``页`` / ``page`` /
  ``p.`` / ``pp.``) or the ``第`` marker: a number is a page only when the
  sentence says so. Years ("2024 年财报"), standards/versions ("ISO 9001"),
  product names ("GPT-4") and statistics carry no page word and are therefore
  NEVER read as page numbers. When the sentence names no page (or asks for the
  whole document) the slot is OMITTED and the tool applies its own default
  handling — the handler never substitutes a guessed page range.

Note: there is NO ``query`` / ``question`` slot on this capability — neither the
Registry schema nor the ``read_document`` tool declares one, so none is emitted
(an unknown slot would fail the Binder as ``BIND_INVALID``). The user's question
is not an argument here: the tool reads the document, and the turn's own LLM
composes the answer from the extracted text.

The returned draft (``{slot: value}``) is passed through the SAME Binder /
ActionExecutor / ToolRuntime handoff every other capability uses.
"""
from __future__ import annotations

import re

# Mirrors binder._CONTEXT_SLOT_SOURCES["asset_id"] exactly (attachment -> path ->
# viewer), so the draft this handler emits is the value the Binder settles on.
_ASSET_FACT_FIELDS = ("attachment_asset_id", "path_asset_id", "viewer_asset_id")

# Explicit page references only. Every alternative is anchored to a page WORD
# (页 / page / p. / pp.), so a bare number that merely follows a year/brand/stat
# can never match. Alternatives run most-specific first; each match sets exactly
# one group pair below.
_PAGES_RE = re.compile(
    r"第\s*(\d+)\s*页\s*(?:到|至|~|—|–|-)\s*第?\s*(\d+)\s*页"   # 第3页到第5页
    r"|第\s*(\d+)\s*(?:到|至|~|—|–|-)\s*(\d+)\s*页"             # 第10到15页 / 第3-5页
    r"|第\s*(\d+)\s*页"                                          # 第3页
    r"|(\d+)\s*(?:到|至|~|—|–|-)\s*(\d+)\s*页"                  # 3到5页 / 3-5页
    r"|(\d+)\s*页"                                               # 3页
    r"|\bpages?\s*(\d+)\s*(?:-|–|—|~|to)\s*(\d+)"               # pages 3-5 / page 10 to 15
    r"|\bpages?\s*(\d+)"                                         # page 4
    r"|\bpp?\.\s*(\d+)\s*(?:-|–|—|~)\s*(\d+)"                   # pp. 3-5
    r"|\bpp?\.\s*(\d+)",                                         # p.3
    re.IGNORECASE,
)
# Capturing-group layout of _PAGES_RE: start/end pairs, then singles.
_PAGE_PAIRS = ((1, 2), (3, 4), (6, 7), (9, 10), (12, 13))
_PAGE_SINGLES = (5, 8, 11, 14)


def _resolve_asset_id(facts) -> str:
    """The first non-empty asset id among the context facts, else ``""``."""
    if facts is None:
        return ""
    for field in _ASSET_FACT_FIELDS:
        value = str(getattr(facts, field, "") or "").strip()
        if value:
            return value
    return ""


def _extract_pages(message: str) -> str | None:
    """The explicit page/slide spec in ``message`` (``"3"`` / ``"2,5"`` / ``"1-3"``),
    or ``None`` when the sentence names no page."""
    fragments: list[str] = []
    for match in _PAGES_RE.finditer(message):
        for start_group, end_group in _PAGE_PAIRS:
            if match.group(start_group) is not None:
                fragments.append(f"{match.group(start_group)}-{match.group(end_group)}")
                break
        else:
            for group in _PAGE_SINGLES:
                if match.group(group) is not None:
                    fragments.append(match.group(group))
                    break
    return ",".join(fragments) if fragments else None


class ReadDocumentHandler:
    """``cap-read-document`` parameter handler (stateless)."""

    capability_id = "cap-read-document"

    async def acquire(self, *, query: str, facts) -> dict[str, object] | None:
        """Return the ``read_document`` argument draft, or ``None`` when no document
        is in context (fail-closed — the Agent owns the clarification)."""
        asset_id = _resolve_asset_id(facts)
        if not asset_id:
            return None
        message = str(query or "").strip()
        draft: dict[str, object] = {"asset_id": asset_id}
        pages = _extract_pages(message)
        if pages is not None:
            draft["pages"] = pages
        return draft
