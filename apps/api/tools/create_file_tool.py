"""``create_file``: create ONE text file in the caller's cloud drive.

The write half of the chat's "创建一个文件 abc.txt" flow. Destination priority is
the shared rule in :mod:`api.tools.drive_destination`: an explicit
``folder_path`` the tool was called with, else an explicitly named folder in the
user's sentence, else the Cloud Drive working folder of the turn, else the My
Drive root (the pre-existing default).

Bytes go through ``DriveService.save_artifact`` — the same entry the toolkit uses
to drop a generated artifact into a folder: unique-name suffixing (never
overwrites), READY asset row, drive audit log. Nothing is written until the
destination and the name have passed those checks.
"""
from __future__ import annotations

from agent import (
    Context,
    ToolExecution,
    ToolOutput,
    ToolPermission,
    ToolRuntime,
    define_tool,
    text_block,
)
from api.tools.drive_destination import (
    DestinationError,
    resolve_destination,
    write_mime_type,
)
from api.tools.turn_material import turn_query
from core.application.drive_service import DriveError, DriveService
from core.infrastructure.request_context import get_request_user_id

# A chat turn creates a note, never a data dump; a bigger body belongs in a
# toolkit artifact (summary / slides) which streams its own bytes.
MAX_CONTENT_CHARS = 1_000_000


def register(runtime: ToolRuntime, ctx: Context, llm) -> None:
    async def create_file(args: dict, exec: ToolExecution) -> str:
        uid = get_request_user_id()
        if uid is None:
            raise ValueError("preflight: create_file requires an authenticated user")
        name = (args.get("name") or "").strip().strip("/")
        if not name or "/" in name:
            raise ValueError("preflight: file name must be a single non-empty name")
        if len(name) > 200:
            raise ValueError("preflight: file name is too long (max 200)")
        content = args.get("content")
        content = "" if content is None else str(content)
        if len(content) > MAX_CONTENT_CHARS:
            raise ValueError(
                f"preflight: file content is too large (max {MAX_CONTENT_CHARS} chars)")

        drive = DriveService(ctx.resolve("session_factory"))
        workspace_id = None
        folder_path = None
        destination = None
        try:
            destination = await resolve_destination(
                drive, uid, query=turn_query(),
                explicit=(args.get("folder_path") or "").strip())
            if destination is not None:
                workspace_id = destination.workspace_id
                folder_path = destination.folder_path or None
            asset = await drive.save_artifact(
                uid, name, write_mime_type(name), content.encode("utf-8"),
                folder_path=folder_path, workspace_id=workspace_id,
            )
        except DestinationError as exc:
            raise ValueError(f"preflight: {exc}") from None
        except DriveError as exc:
            raise ValueError(f"preflight: {exc}") from None
        where = destination.join(asset.name) if destination else asset.name
        return f"Created file '{where}' in your drive."

    runtime.register(
        define_tool(
            name="create_file",
            description="Create ONE text file (markdown / txt / csv / json) in the "
            "user's cloud drive. It lands in the folder the user is working in "
            "unless they named another folder or `folder_path` says otherwise. A "
            "busy name is auto-suffixed, never overwritten.",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "maxLength": 200,
                             "description": "File name including its extension "
                             "(e.g. 'abc.txt', 'report.md')."},
                    "content": {
                        "type": "string",
                        "description": "The file body. Omit for an empty file.",
                    },
                    "folder_path": {
                        "type": "string",
                        "description": "Optional destination folder (a path or a "
                        "folder name). Omit to use the user's current folder.",
                    },
                },
                "required": ["name"],
            },
            output=ToolOutput(
                schema={"type": "string"}, render=lambda args, value: [text_block(value)]
            ),
            execute=create_file,
            permission={ToolPermission.WRITE},
        )
    )
