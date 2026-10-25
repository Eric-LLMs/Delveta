"""The Cloud Drive destination for a chat turn's write.

One rule set, shared by every chat-facing drive write (``create_folder``,
``create_file`` and the summary tool's save step). Resolution order:

1. an explicit destination the TOOL was called with;
2. a destination the USER named in the sentence — a narrow, explicit grammar
   (the current folder / the drive root / one named folder), never free-form
   path inference;
3. the folder the desktop client is working in for this turn (``drive_cwd``,
   resolved once at context-build in :mod:`core.application.chat.context`).

Nothing here invents a destination. A named folder that is not visible, or that
matches more than one folder, raises :class:`DestinationError` instead of
writing somewhere else. A workspace destination is admitted only after the
existing Cloud Drive workspace check (:meth:`DriveService.ensure_workspace_member`),
so the chat path cannot widen access the drive itself refuses.

A turn with no ``drive_cwd`` (Local workspace mode, legacy client) resolves to
``None``: the tool keeps its pre-existing behaviour byte-for-byte.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from agent.engine.context import current_turn
from core.application.drive_service import DriveError, DriveService

# A drive path the user wrote: "My Drive/a/b" / "我的云盘/a/b".
_DRIVE_PATH = re.compile(
    r"(?:My Drive|我的云盘)\s*[／/]\s*([^\s\"'，。；：！？、,;:<>|／/][^\s\"'，。；：！？、,;:<>|]*)"
)
# "My Drive" with no sub-path ⇒ the personal root ("放到 My Drive 下面").
_MY_DRIVE_ROOT = re.compile(r"(?:My Drive|我的云盘)\s*(?:根目录|根|下面|下边|下|里面|里|中)?", re.I)
# "current folder / this directory / here" ⇒ the turn's working folder.
_CWD_PHRASE = re.compile(
    r"(?:当前|这个|本|此)\s*(?:目录|文件夹)"
    r"|(?:放到|放在|保存到|保存在|存到|输出到|写到|写入|生成到|生成在|创建在)\s*这里"
)
# "在 <name> 文件夹下面 …" / "在 <name> 下面创建 …" ⇒ ONE named folder.
_NAMED_FOLDER = re.compile(
    r"在\s*([^\s，。；：、,;:][^，。；：、,;:]{0,58}?)\s*文件夹\s*(?:下面|下边|下|里面|里|中)"
    r"|在\s*([^\s，。；：、,;:]{1,58}?)\s*(?:下面|下边)\s*"
    r"(?:创建|新建|建立|生成|保存|输出|放到|放|写|存)"
)

# Tool arguments an agent may pass for "the folder this goes in".
EXPLICIT_FOLDER_SLOTS = ("parent_path", "folder_path", "output_dir")

# The personal root written as a bare alias ("My Drive" on its own) is still the
# root, not a folder to look up by name.
_ROOT_ALIASES = ("My Drive", "我的云盘")

_WRITE_MIME = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
    ".json": "application/json",
    ".csv": "text/csv",
    ".yml": "text/yaml",
    ".yaml": "text/yaml",
}


class DestinationError(ValueError):
    """The requested destination cannot be resolved to exactly one folder.

    Raised BEFORE any write, so the caller may report it as a preflight failure.
    """


@dataclass(frozen=True)
class DriveDestination:
    """A resolved folder inside the caller's Cloud Drive (``folder_path`` "" = root)."""

    workspace_id: str | None
    folder_path: str
    label: str

    def join(self, name: str) -> str:
        """The display path of ``name`` inside this folder."""
        return f"{self.label} / {name}" if self.label else name


def _turn_context() -> dict:
    turn = current_turn()
    return dict(getattr(turn, "context", None) or {})


def current_cwd() -> DriveDestination | None:
    """The Cloud Drive folder this turn works in, or ``None`` outside Cloud mode."""
    raw = _turn_context().get("drive_cwd")
    if not isinstance(raw, dict):
        return None
    path = str(raw.get("path") or "").strip().strip("/")
    ws = str(raw.get("workspace_id") or "").strip() or None
    label = str(raw.get("label") or "").strip() or (
        f"My Drive / {path.replace('/', ' / ')}" if path else "My Drive")
    return DriveDestination(workspace_id=ws, folder_path=path, label=label)


def declared_error() -> str:
    """Why the client's declared working folder was refused ("" = none)."""
    return str(_turn_context().get("drive_error") or "").strip()


def _uuid(value: Any) -> UUID | None:
    if value in (None, ""):
        return None
    return value if isinstance(value, UUID) else UUID(str(value))


def _sentence_destination(query: str) -> tuple[str, str] | None:
    """``(kind, value)`` for an explicitly named destination, or ``None``.

    Deliberately narrow: a bare file name ("abc.txt"), an unprefixed folder name
    or a relative path is NOT a destination — those default to the working
    folder. Only the three shapes below are treated as an override.
    """
    text = str(query or "")
    m = _DRIVE_PATH.search(text)
    if m:
        return ("path", m.group(1).strip())
    if _MY_DRIVE_ROOT.search(text):
        return ("root", "")
    if _CWD_PHRASE.search(text):
        return ("cwd", "")
    m = _NAMED_FOLDER.search(text)
    if m:
        name = (m.group(1) or m.group(2) or "").strip()
        if name:
            return ("name", name)
    return None


def _split_scope(raw: str) -> tuple[bool, str]:
    """``(pinned_to_my_drive, remainder)`` for a destination the user/tool wrote."""
    value = raw.strip().rstrip("/")
    for prefix in ("My Drive/", "我的云盘/"):
        if value.startswith(prefix):
            return True, value[len(prefix):].strip()
    if value in _ROOT_ALIASES:
        return True, ""
    return False, raw.strip()


def _label_for(folder: dict, ws_names: dict[str, str]) -> str:
    path = str(folder.get("path") or "")
    ws = folder.get("workspace_id")
    base = ws_names.get(str(ws)) if ws else "My Drive"
    base = base or "Workspace"
    return f"{base} / {path.replace('/', ' / ')}" if path else base


async def _assert_member(drive: DriveService, user_id: Any, workspace_id: str | None) -> None:
    """The drive's own workspace ACL; a personal folder needs no membership."""
    ws = _uuid(workspace_id)
    if ws is None:
        return
    try:
        await drive.ensure_workspace_member(user_id, ws)
    except DriveError as exc:
        raise DestinationError(str(exc)) from None


async def _resolve_named(
    drive: DriveService, user_id: Any, name: str,
) -> DriveDestination:
    """ONE visible folder (or workspace) whose name is exactly ``name``."""
    name = name.strip().strip("/")
    if not name:
        raise DestinationError("the destination folder name is empty")
    folders = await drive.list_folders(user_id)
    hits = [f for f in folders if str(f.get("name") or "") == name]
    ws_names = {str(w["id"]): str(w["name"]) for w in await drive.list_workspaces(user_id)}
    if len(hits) > 1:
        where = "; ".join(_label_for(f, ws_names) for f in hits)
        raise DestinationError(
            f"'{name}' matches more than one folder ({where}); say which one")
    if len(hits) == 1:
        folder = hits[0]
        await _assert_member(drive, user_id, folder.get("workspace_id"))
        return DriveDestination(
            workspace_id=(str(folder["workspace_id"]) if folder.get("workspace_id") else None),
            folder_path=str(folder.get("path") or ""),
            label=_label_for(folder, ws_names),
        )
    workspaces = [w for w in ws_names.items() if w[1] == name]
    if len(workspaces) == 1:
        ws_id, ws_name = workspaces[0]
        return DriveDestination(workspace_id=ws_id, folder_path="", label=ws_name)
    raise DestinationError(f"no folder named '{name}' is visible in your Cloud Drive")


async def _resolve_explicit(
    drive: DriveService, user_id: Any, raw: str, cwd: DriveDestination,
) -> DriveDestination:
    """A destination the tool passed explicitly.

    A path-shaped value keeps the historical meaning (a path inside the working
    folder's scope; a ``My Drive/`` prefix pins the personal scope). A bare word
    is a folder NAME and must resolve uniquely — the same rule the sentence
    follows, never a guess.
    """
    personal, value = _split_scope(raw)
    value = value.strip().strip("/")
    if not value:
        return DriveDestination(None, "", "My Drive")
    if personal or "/" in value:
        workspace_id = None if personal else cwd.workspace_id
        await _assert_member(drive, user_id, workspace_id)
        label = (f"My Drive / {value.replace('/', ' / ')}" if personal
                 else cwd.join(value))
        return DriveDestination(workspace_id=workspace_id, folder_path=value, label=label)
    return await _resolve_named(drive, user_id, value)


async def resolve_destination(
    drive: DriveService, user_id: Any, *, query: str, explicit: str = "",
) -> DriveDestination | None:
    """The destination for this turn's write, or ``None`` when no Cloud turn applies.

    ``None`` means "the caller keeps its own local/default behaviour": neither the
    client nor the user named a Cloud Drive place, so nothing here overrides it.
    """
    error = declared_error()
    cwd = current_cwd()
    if cwd is None and not error:
        return None
    if error:
        raise DestinationError(error)
    assert cwd is not None

    explicit = str(explicit or "").strip()
    if explicit:
        return await _resolve_explicit(drive, user_id, explicit, cwd)

    named = _sentence_destination(query)
    if named is not None:
        kind, value = named
        if kind == "root":
            return DriveDestination(None, "", "My Drive")
        if kind == "path":
            return await _resolve_explicit(drive, user_id, f"My Drive/{value}", cwd)
        if kind == "name":
            return await _resolve_named(drive, user_id, value)
    return cwd


def write_mime_type(name: str) -> str:
    """MIME type for a text file created through chat (extension-based)."""
    lowered = str(name or "").lower()
    for ext, mime in _WRITE_MIME.items():
        if lowered.endswith(ext):
            return mime
    return "text/plain"
