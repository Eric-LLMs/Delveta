"""Literal path tokens in the user's own sentence — the deterministic extractor.

A path value on the funnel path may come ONLY from literal text the user wrote:
this module owns the grammar once, so ``cap-read-file`` (a path slot) and
``cap-summary`` (a workspace-file input) cannot drift apart. The output
DIRECTORY is not extracted here: it is a MODEL-owned slot, and this module only
recognises the save cue that authorizes that slot for the turn.

Two shapes are recognised, copied VERBATIM (no translation, no case change, no
prefix stripping):

* a DRIVE path — the ``My Drive/`` / ``我的云盘/`` prefix (case-insensitive) plus
  the following path text, which may contain spaces;
* a WORKSPACE path — the first whitespace token carrying a directory slash or a
  ``<name>.<ext>`` filename.

Nothing here guesses: a pronoun referent, a vague sentence or a missing token
yields ``None`` and the caller fails closed to the Agent.
"""
from __future__ import annotations

import re

# The drive plane is a path PREFIX inside the sentence: "My Drive/..." /
# "我的云盘/...". Matched case-insensitively (My Drive / my drive / MY DRIVE) with a
# half- or full-width slash; the match NEVER rewrites the text — the returned
# span is the caller's original literal (casing and spaces intact).
_DRIVE_PREFIX_RE = re.compile(r"(?:My Drive|我的云盘)\s*[／/]\s*\S", re.IGNORECASE)

# A workspace path token: a whitespace-delimited token that carries a directory
# slash, or is a "<name>.<ext>" filename. Everything is copied verbatim.
_PATH_SLASH_CHARS = ("/", "／")
_FILE_EXT_RE = re.compile(r"^\S+\.[A-Za-z0-9]{1,8}$")

# An explicit export destination: "save to ..." / "保存到 ..." and its close
# variants. The cue says the user WANTS a saved result; it never supplies the
# directory itself (that value is extracted by the shared model extractor).
_DEST_CUE_RE = re.compile(
    r"(?:保存(?:到|至)|存(?:到|至)|写(?:到|至)|输出(?:到|至)|导出(?:到|至)|放到"
    r"|save\s+(?:it\s+)?(?:to|in|into)"
    r"|write\s+(?:it\s+)?(?:to|in|into)"
    r"|export\s+(?:it\s+)?(?:to|into)"
    r"|output\s+(?:it\s+)?(?:to|into))\s*[:：]?\s*",
    re.IGNORECASE,
)


def drive_path(message: str) -> str | None:
    """The drive-plane path (prefix + verbatim remainder), or ``None``.

    The prefix anchors the path; everything from it to the end of the sentence is
    the path (a Drive filename may contain spaces, so there is no whitespace cut).
    """
    match = _DRIVE_PREFIX_RE.search(message)
    if match is None:
        return None
    return message[match.start():].strip() or None


def workspace_path(message: str) -> str | None:
    """The first literal workspace path token, or ``None``.

    "First" is by position, so the earliest path token in the sentence wins.
    """
    for token in message.split():
        if any(ch in token for ch in _PATH_SLASH_CHARS) or _FILE_EXT_RE.match(token):
            return token
    return None


def literal_path(message: str) -> str | None:
    """The drive path when present, else the first workspace path token."""
    return drive_path(message) or workspace_path(message)


def has_destination_cue(message: str) -> bool:
    """True when the sentence explicitly asks to save/export to a destination.

    This is the per-turn AUTHORIZATION for the ``output_dir`` model slot — not a
    value source. The directory itself is extracted by the shared model
    extractor; a cue with no usable directory there is an extraction failure the
    Agent owns, never a silent fall-back to chat-only output.
    """
    return _DEST_CUE_RE.search(message) is not None
