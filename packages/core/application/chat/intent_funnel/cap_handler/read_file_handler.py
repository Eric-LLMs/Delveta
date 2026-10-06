"""ReadFileHandler — argument acquisition for ``cap-read-file``.

Owns ONLY this capability's required schema slot, ``path``. ``cap-read-file``
(tool binding ``read_file``) addresses ONE file by PATH across two planes — a
workspace-relative path, or a drive path under ``My Drive/`` / ``我的云盘/`` —
and the runtime dispatches on the path shape (``fs_tools._DRIVE_PATH_RE``). There
is NO asset model on this capability: the path is a filesystem/workspace path,
never an ``asset_id``, and it is never reachable from a viewer/attachment.

The value is resolved DETERMINISTICALLY — no model call, no Qwen. ``path`` may
come ONLY from a literal path token in the user's own sentence:

* a DRIVE path — the ``My Drive/`` / ``我的云盘/`` prefix (matched
  case-insensitively) plus the following path text, copied VERBATIM so the
  caller's casing and spaces survive (``my drive/annual report 2025.pdf`` stays
  lowercase, ``My Drive/Agent Harness Engineering A Survey.pdf`` keeps its
  spaces);
* a WORKSPACE path — the first whitespace token that carries a ``/`` segment or a
  ``<name>.<ext>`` filename (``./data/train.csv`` keeps its ``./`` prefix,
  ``报告.docx`` is never transliterated, ``src/main/java/App.java`` keeps its
  directories).

Six hard prohibitions — each violation returns ``None`` (fail-closed) instead:

* never inject :attr:`~...contract.TurnFacts.viewer_asset_id` (or any asset /
  attachment field) as ``path`` — the path comes from the sentence only;
* never call Qwen to generate or guess a ``path``;
* never translate a path (``报告.docx`` must not become ``report.docx``);
* never normalize case (no ``.lower()`` / ``.upper()`` on the path value);
* never strip a ``./`` (or any) prefix;
* never emit ``max_chars`` — this handler produces the ``path`` slot and nothing
  else, so the tool's own default governs truncation.

A pronoun referent (``读取这个文档`` / ``read this file``), a vague spoken
sentence (``read that file we talked about``), a sentence with no literal path
token, or a blank query all return ``None`` — the turn exits to the Agent
(``ACQUISITION_MISSING``) rather than fabricate a path.
"""
from __future__ import annotations

import re

# The drive plane is a path PREFIX inside the sentence: "My Drive/..." /
# "我的云盘/...". Matched case-insensitively (My Drive / my drive / MY DRIVE) with
# a half- or full-width slash; the match NEVER rewrites the text — the returned
# span is the caller's original literal (casing and spaces intact).
_DRIVE_PREFIX_RE = re.compile(r"(?:My Drive|我的云盘)\s*[／/]\s*\S", re.IGNORECASE)

# A workspace path token: a whitespace-delimited token that carries a directory
# slash, or is a "<name>.<ext>" filename. Everything is copied verbatim.
_PATH_SLASH_CHARS = ("/", "／")
_FILE_EXT_RE = re.compile(r"^\S+\.[A-Za-z0-9]{1,8}$")


def _drive_path(message: str) -> str | None:
    """The drive-plane path (prefix + verbatim remainder), or ``None``.

    The prefix anchors the path; everything from it to the end of the sentence is
    the path (a Drive filename may contain spaces, so there is no whitespace cut).
    """
    match = _DRIVE_PREFIX_RE.search(message)
    if match is None:
        return None
    return message[match.start():].strip() or None


def _workspace_path(message: str) -> str | None:
    """The first literal workspace path token, or ``None``.

    "First" is by position, so the earliest path token in the sentence wins.
    """
    for token in message.split():
        if any(ch in token for ch in _PATH_SLASH_CHARS) or _FILE_EXT_RE.match(token):
            return token
    return None


class ReadFileHandler:
    """``cap-read-file`` parameter handler (stateless).

    ``facts`` is accepted for the common handler contract but ignored ON PURPOSE:
    the path must come from the sentence, never from the turn's asset context
    (injecting ``viewer_asset_id`` here would be asset pollution).
    """

    capability_id = "cap-read-file"

    async def acquire(self, *, query: str, facts) -> dict[str, object] | None:
        """Return the ``read_file`` argument draft (``{"path": <literal>}``), or
        ``None`` when the sentence carries no literal path (fail-closed)."""
        message = str(query or "").strip()
        if not message:
            return None
        path = _drive_path(message)
        if path is None:
            path = _workspace_path(message)
        if path is None:
            return None
        return {"path": path}
