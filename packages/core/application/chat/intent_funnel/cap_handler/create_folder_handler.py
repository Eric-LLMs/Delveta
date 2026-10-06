"""CreateFolderHandler — argument acquisition for ``cap-create-folder``.

Owns ONLY this capability's required schema slot, ``name``. ``cap-create-folder``
(tool binding ``create_folder``) creates ONE folder in the caller's cloud drive
(``DriveService.create_folder`` — My Drive root unless an internal parent path is
supplied). There is no asset model and no path plane on this capability: the
value is a single folder NAME, never a filesystem path.

The name is resolved DETERMINISTICALLY — no model call, no Qwen. It may come ONLY
from a literal folder name in the user's own sentence, found by a priority ladder
(first hit wins), never echoed from the whole sentence:

1. a QUOTED span — ASCII ``"…"`` / ``'…'`` or the full-width ``“…”`` / ``‘…’`` /
   ``「…」`` / ``『…』`` pairs (the enclosing quotes are dropped, the inside copied
   verbatim, so ``"2026 年度计划"`` keeps its space);
2. an explicit naming lead — ``叫 X`` / ``命名为 X`` / ``named X`` / ``called X``
   — captured up to the trailing delimiter (``的文件夹`` / ``文件夹`` / ``的`` /
   punctuation / end); a multi-token name is NEVER truncated at whitespace
   (``2026-Q1 报表`` stays whole, unlike the legacy model extraction);
3. the English ``for the X`` clause — the trailing noun phrase to end of sentence;
4. a Chinese modifier before the folder word — ``X 文件夹`` / ``X 的文件夹``.

The extracted value is copied VERBATIM — original casing and spaces survive
(``Release Notes`` keeps its space, nothing is lowercased or translated).

Fail-closed — returns ``None`` (the turn exits to the Agent with
``ACQUISITION_MISSING``) — when: no rule matches; the only candidate is a generic
stop word (``folder`` / ``文件夹`` / ``一个`` / ``please`` …); the query is blank;
or the candidate contains a slash (``/`` / ``\\``) — a slash means the sentence
carried a PATH, not a single folder name, and this handler does not guess.

``parent_path`` is NEVER emitted: the public capability contract is frozen to the
single ``name`` slot (``create_folder``'s optional ``parent_path`` is tool-owned),
so the draft is always exactly ``{"name": <literal>}``.

``facts`` is accepted for the common handler contract but ignored ON PURPOSE — the
name comes from the sentence, never from the turn's asset context.
"""
from __future__ import annotations

import re

# ── rule 1: quoted spans — the quotes are dropped, the inside copied verbatim ─────
_QUOTED_PAIRS = (
    ('"', '"'),
    ("'", "'"),
    ("“", "”"),
    ("‘", "’"),
    ("「", "」"),
    ("『", "』"),
)

# ── rule 2: explicit naming leads ────────────────────────────────────────────────
# The non-greedy capture stops at the trailing delimiter via the lookahead, so a
# multi-token name ("2026-Q1 报表") is kept WHOLE — never cut at a space.
_NAMED_RE = re.compile(
    r"(?:叫|命名为|named|called)\s*(.+?)(?=\s*(?:的)?文件夹|\s*的|$)",
    re.IGNORECASE,
)

# ── rule 3: English "for the X" ─────────────────────────────────────────────────
_FOR_THE_RE = re.compile(r"for the\s+(.+?)\s*$", re.IGNORECASE)

# ── rule 4: Chinese "X 文件夹" / "X 的文件夹" ──────────────────────────────────────
_ZH_FOLDER_RE = re.compile(
    r"(?:建个|建一个|新建一个|新建|创建一个|创建|加个|建)\s*(.+?)\s*(?:的)?文件夹"
)

# A candidate that is only intent/stop words is NOT a folder name.
_GENERIC_RE = re.compile(
    r"文件夹|folder|新建|创建|建个|建一个|一个|帮我|please", re.IGNORECASE
)
_GENERIC_EXACT = frozenset(
    {"a", "an", "the", "new", "folder", "please", "的", "了", "个", "这个", "那个"}
)

_EDGE_PUNCT = " \t\r\n。.!?，,;；:：\"'“”‘’「」『』"
_SLASH_CHARS = ("/", "\\", "／", "＼")


def _quoted(message: str) -> str | None:
    for open_q, close_q in _QUOTED_PAIRS:
        start = message.find(open_q)
        if start == -1:
            continue
        end = message.find(close_q, start + 1)
        if end > start + 1:
            return message[start + 1:end].strip() or None
    return None


def _named(message: str) -> str | None:
    match = _NAMED_RE.search(message)
    return match.group(1).strip() or None if match else None


def _for_the(message: str) -> str | None:
    match = _FOR_THE_RE.search(message)
    return match.group(1).strip() or None if match else None


def _zh_folder(message: str) -> str | None:
    match = _ZH_FOLDER_RE.search(message)
    return match.group(1).strip() or None if match else None


def _clean(name: str | None) -> str | None:
    """Trim edge punctuation; reject empty, generic-only, or slash-bearing names."""
    if name is None:
        return None
    name = name.strip(_EDGE_PUNCT).strip()
    if not name:
        return None
    if name.lower() in _GENERIC_EXACT or _GENERIC_RE.search(name):
        return None
    if any(ch in name for ch in _SLASH_CHARS):
        return None
    return name


class CreateFolderHandler:
    """``cap-create-folder`` parameter handler (stateless).

    ``facts`` is accepted for the common handler contract but ignored ON PURPOSE:
    the name must come from the sentence, never from the turn's asset context.
    """

    capability_id = "cap-create-folder"

    async def acquire(self, *, query: str, facts) -> dict[str, object] | None:
        """Return the ``create_folder`` argument draft (``{"name": <literal>}``),
        or ``None`` when the sentence carries no literal folder name."""
        message = str(query or "").strip()
        if not message:
            return None
        for extractor in (_quoted, _named, _for_the, _zh_folder):
            name = _clean(extractor(message))
            if name is not None:
                return {"name": name}
        return None
