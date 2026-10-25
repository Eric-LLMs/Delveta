"""``create_folder``: create a folder in the caller's cloud drive.

Destination priority (business rule, shared with every chat-facing drive write —
see :mod:`api.tools.drive_destination`):

1. an explicit ``parent_path`` the tool was called with;
2. a folder the USER named in the sentence ("在 Documents 文件夹下面 …");
3. the Cloud Drive folder the desktop client is working in (``drive_cwd``);
4. with no Cloud context at all, the pre-existing default (My Drive root).

Registers the SAME service entry the REST endpoint uses (``DriveService.create_folder`` —
validation, membership check, unique-suffix, repo write, audit log), so the LLM keeps the
capability as fallback for the chat fast path's ``create_folder`` allowlist entry. Every
``DriveError`` this call can raise happens BEFORE any write (name/parent validation,
workspace membership, collision) — the body re-raises them as ``"preflight: …"`` value
errors, the channel the ACTION executor reads as "provably not executed ⇒ safe to
escalate".
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
from api.tools.drive_destination import DestinationError, resolve_destination
from api.tools.turn_material import turn_query
from core.application.drive_service import DriveError, DriveService
from core.infrastructure.request_context import get_request_user_id


def register(runtime: ToolRuntime, ctx: Context, llm) -> None:
    async def create_folder(args: dict, exec: ToolExecution) -> str:
        uid = get_request_user_id()
        if uid is None:
            raise ValueError("preflight: create_folder requires an authenticated user")
        name = (args.get("name") or "").strip()
        if not name:
            raise ValueError("preflight: folder name must be a single non-empty name")
        workspace_id = None
        parent_path = (args.get("parent_path") or "").strip() or None
        drive = DriveService(ctx.resolve("session_factory"))
        try:
            # Cloud turns resolve the working folder (or the destination the user
            # named); local turns keep the historical My Drive-root behaviour.
            destination = await resolve_destination(
                drive, uid, query=turn_query(),
                explicit=(args.get("parent_path") or "").strip())
            if destination is not None:
                workspace_id = destination.workspace_id
                parent_path = destination.folder_path or None
            folder = await drive.create_folder(uid, workspace_id, parent_path, name)
        except DestinationError as exc:
            raise ValueError(f"preflight: {exc}") from None
        except DriveError as exc:
            # All DriveError paths of create_folder are pre-write ⇒ no side effect.
            raise ValueError(f"preflight: {exc}") from None
        return f"Created folder '{folder['path']}' in your drive."

    runtime.register(
        define_tool(
            name="create_folder",
            description="Create a folder in the user's cloud drive. Returns the created "
            "folder path. Fails with an error if the name is invalid or a write is denied; "
            "a busy name is auto-suffixed, never overwritten.",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "maxLength": 120,
                             "description": "Folder name (single name, no '/')."},
                    "parent_path": {
                        "type": "string",
                        "description": "Optional existing folder path to create it under "
                        "(default: drive root).",
                    },
                },
                "required": ["name"],
            },
            output=ToolOutput(
                schema={"type": "string"}, render=lambda args, value: [text_block(value)]
            ),
            execute=create_folder,
            # Explicit WRITE: this tool mutates the cloud drive. Declared here so the
            # sandbox's ASK/DENY gate never depends on an inference from a parameter
            # name happening to contain "path" (e.g. `parent_path`) — a rename would
            # otherwise silently downgrade the tool to READ and bypass approval.
            permission={ToolPermission.WRITE},
        )
    )
