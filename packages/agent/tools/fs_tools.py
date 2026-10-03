"""Core resident filesystem / shell tools for the agent kernel.

``read_file`` (READ), ``edit_file`` (WRITE), and ``bash`` (WRITE + NETWORK) are the
microkernel's resident file-system tools. All LOCAL file access is rooted at a workspace
directory (path traversal is rejected), and the permission class is declared explicitly
so the :class:`~agent.security.sandbox.Sandbox` gates them: the default READ-only session denies
``edit_file`` / ``bash`` unless the host grants WRITE / NETWORK.

Two file planes, one capability (drive-edit ruling): ``read_file`` /
``edit_file`` address a file by PATH and dispatch on its plane — a path under
``My Drive/`` (or ``我的云盘/``) is a Drive asset resolved to its real asset_id and
handled by the injected Drive service (edits follow its copy-on-write semantics);
anything else is a workspace file. The tools never materialize a Drive copy.

``bash`` delegates to a :class:`~agent.tools.bash_sandbox.BashSandbox` (host or docker) and
applies a best-effort workspace-escape guard; the sandbox is the real isolation boundary.
"""
from __future__ import annotations

import asyncio
import os
import re
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

from agent.engine.decisions import ToolExecution, text_block
from agent.tools.bash_sandbox import BashSandbox, assert_no_escape, get_bash_sandbox
from agent.tools.definition import ToolDefinition, ToolOutput, define_tool
from agent.tools.tool_permissions import ToolPermission
from core.config import settings
from core.infrastructure.request_context import get_request_user_id

if TYPE_CHECKING:  # runtime injection only — the kernel never imports the service
    from core.application.drive_service import DriveService

# "My Drive/scope.md" / "我的云盘/notes/a.md" — explicit personal-drive root only,
# the same shape the chat turn-facts recognizer uses (context._PATH_TOKEN).
_DRIVE_PATH_RE = re.compile(r"^\s*(?:My Drive|我的云盘)\s*[／/]\s*(\S.*)$")


async def _drive_asset(drive: "DriveService", raw_path: str) -> tuple[UUID, UUID]:
    """Resolve a drive-plane path → ``(user_id, asset_id)``.

    Raises (never falls back to the workspace plane) when the turn has no user or the
    named path holds no READY asset — an unresolvable Drive file is an honest error,
    not a silently-different local file.
    """
    if drive is None:
        raise ValueError("drive plane not available in this runtime")
    user_id = get_request_user_id()
    if user_id is None:
        raise ValueError("drive file access requires a request user")
    rel = _DRIVE_PATH_RE.match(raw_path).group(1).strip()
    found = await drive.resolve_personal_path(user_id, f"My Drive/{rel}")
    if not found:
        raise ValueError(f"no readable Drive file at: My Drive/{rel}")
    return user_id, UUID(found["asset_id"])


def _resolve(workspace: Path, raw_path: str) -> Path:
    """Resolve ``raw_path`` inside ``workspace``; raise ValueError on escape."""
    root = workspace.resolve()
    candidate = (root / raw_path).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"path escapes workspace: {raw_path}")
    return candidate


def _read(path: Path, max_chars: int) -> str:
    """Sync body of ``read_file`` (runs in a worker thread to keep the loop free)."""
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    if max_chars and path.stat().st_size > max_chars:
        return text[:max_chars] + "\n…(truncated)"
    return text


def read_file_tool(workspace: Path, drive: "DriveService | None" = None) -> ToolDefinition:
    async def execute(args: dict, exec: ToolExecution) -> str:
        raw = args["path"]
        max_chars = int(args.get("max_chars", 0) or 0)
        if _DRIVE_PATH_RE.match(raw):
            user_id, asset_id = await _drive_asset(drive, raw)
            text = await drive.read_text(user_id, asset_id)
            if max_chars and len(text) > max_chars:
                return text[:max_chars] + "\n…(truncated)"
            return text
        path = _resolve(workspace, raw)
        return await asyncio.to_thread(_read, path, max_chars)

    return define_tool(
        name="read_file",
        description=(
            "Read a text file and return its contents. The path selects the plane: a "
            "workspace-relative path reads a local file; \"My Drive/<path>\" (or "
            "\"我的云盘/<path>\") reads that cloud-drive file by its real asset."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": (
                    "Workspace-relative file path, or a drive path \"My Drive/<path>\".")},
                "max_chars": {"type": "integer", "description": "Optional read cap."},
            },
            "required": ["path"],
        },
        output=ToolOutput(
            schema={"type": "string"}, render=lambda args, value: [text_block(value)]
        ),
        execute=execute,
        permission={ToolPermission.READ},
        is_concurrency_safe=True,
    )


def _apply_edit(path: Path, old: str | None, new: str) -> tuple[str, bool]:
    """Compute the edited content in a worker thread; returns ``(content, replaced)``."""
    if old is not None and path.is_file():
        current = path.read_text(encoding="utf-8", errors="replace")
        if old not in current:
            raise ValueError("old_text not found in the file — nothing replaced")
        return current.replace(old, new, 1), True
    return new, False


def _atomic_write(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically: temp file in the same dir + ``os.replace``.

    Never leaves a truncated file if the process dies mid-write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def edit_file_tool(workspace: Path, drive: "DriveService | None" = None) -> ToolDefinition:
    async def execute(args: dict, exec: ToolExecution) -> str:
        raw = args["path"]
        old = args.get("old_text")
        new = args.get("new_text", "")
        if _DRIVE_PATH_RE.match(raw):
            user_id, asset_id = await _drive_asset(drive, raw)
            result = await drive.edit_text(
                user_id, asset_id, old_text=old, new_text=new,
            )
            asset = result["asset"]
            if result["mode"] == "in_place":
                verb = "edited in place" if old is not None else "overwritten"
                return f"{raw}: {verb} (asset {asset_id})."
            return (
                f"{raw}: the file is shared, so the edit was saved as a new copy "
                f"\"{asset['name']}\" (asset {asset['id']}); readers of the original "
                f"(asset {result['original_asset_id']}) are unaffected."
            )
        path = _resolve(workspace, raw)
        content, replaced = await asyncio.to_thread(_apply_edit, path, old, new)
        await asyncio.to_thread(_atomic_write, path, content)
        if replaced:
            return f"replaced 1 occurrence in {raw}"
        return f"wrote {raw}"

    return define_tool(
        name="edit_file",
        description=(
            "Edit an existing text file; the path selects the plane: a workspace-relative "
            "path edits a local file, \"My Drive/<path>\" (or \"我的云盘/<path>\") edits that "
            "cloud-drive file (shared files are copy-on-write edited, never mutated under "
            "other readers). Provide old_text to replace a specific snippet, or new_text "
            "only to (over)write the whole file."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": (
                    "Workspace-relative file path, or a drive path \"My Drive/<path>\".")},
                "old_text": {"type": "string", "description": "Exact text to replace (optional)."},
                "new_text": {"type": "string", "description": "Replacement / new file content."},
            },
            "required": ["path"],
        },
        output=ToolOutput(
            schema={"type": "string"}, render=lambda args, value: [text_block(value)]
        ),
        execute=execute,
        destructive=True,
        permission={ToolPermission.WRITE},
    )


def bash_tool(workspace: Path, sandbox: BashSandbox) -> ToolDefinition:
    async def execute(args: dict, exec: ToolExecution) -> str:
        command = args["command"]
        # Cap the model-supplied timeout: an unbounded value would let a prompt park a
        # process forever. Default to (and clamp at) the configured sandbox timeout.
        timeout = min(
            int(args.get("timeout", settings.bash_sandbox_timeout)),
            settings.bash_sandbox_timeout,
        )
        assert_no_escape(workspace, command)
        try:
            return await sandbox.run(command, timeout)
        except TimeoutError:
            raise TimeoutError(f"command timed out after {timeout}s")

    return define_tool(
        name="bash",
        description=(
            "Run a shell command in the host environment. High-risk: requires the session "
            "to have granted WRITE + NETWORK permissions."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to run."},
                "timeout": {"type": "integer", "description": "Timeout in seconds."},
            },
            "required": ["command"],
        },
        output=ToolOutput(
            schema={"type": "string"}, render=lambda args, value: [text_block(value)]
        ),
        execute=execute,
        destructive=True,
        permission={ToolPermission.WRITE, ToolPermission.NETWORK},
    )


def register_fs_tools(
    runtime,
    workspace: Path,
    sandbox: BashSandbox | None = None,
    drive: "DriveService | None" = None,
    exclude: "Iterable[str]" = (),
) -> list[ToolDefinition]:
    """Register the resident fs/shell tools onto ``runtime``; returns the definitions.

    ``drive`` (when provided) enables the Drive plane of read_file/edit_file —
    paths under "My Drive/" resolve to real assets; bash stays workspace-only.

    ``exclude`` names tools to skip registering (e.g. ``{"edit_file"}`` on the
    Chat-process composition). The tool is then absent from the runtime roster,
    so it never appears in the prompt catalog, is not searchable by ``tool_search``,
    cannot be mounted, and is invisible to the LLM's tool array — the implementation
    and its destructive/permission semantics are untouched, simply not wired in.
    Worker / Research planes omit ``exclude`` and keep the original behavior.
    """
    sandbox = sandbox or get_bash_sandbox()
    hidden = set(exclude or ())
    tools = [
        read_file_tool(workspace, drive),
        edit_file_tool(workspace, drive),
        bash_tool(workspace, sandbox),
    ]
    registered: list[ToolDefinition] = []
    for tool in tools:
        if tool.name in hidden:
            continue
        runtime.register(tool)
        registered.append(tool)
    return registered
