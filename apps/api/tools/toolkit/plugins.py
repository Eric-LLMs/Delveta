"""The three toolkit tools as Cordis-style :class:`Plugin` objects.

Each tool is one plugin that owns exactly one :class:`ToolDefinition` (``summary_gen`` /
``mindmap_gen`` / ``slides_gen``). The plugin wires a fresh :class:`ToolKitPipeline` for its
tool and exports the pipeline as a named service (``provides``) — the plugin lifecycle (mount
via the manager, dispose via unregister, hooks via the shared EventBus) is exactly the Cordis
pattern the kernel already uses.

``summary_gen`` accepts exactly ONE content source, because a summary always has one:

* ``paths`` / ``path`` — workspace files (the original contract, also the Agent lane);
* ``asset_id`` — a document the user attached or has open, read through the shared
  authorized reader (``api.tools.document_content``: drive ACL → storage → extraction);
  ``pages`` narrows it to pages/slides;
* ``context`` — the conversation itself ("总结我们刚才讨论的内容"). Nothing on disk is
  read for this lane.

``context`` WINS over an ``asset_id`` the Binder injects from turn facts (the slot is
context-sourced, so an open document rides along even when the user asked about the chat);
it is never combined with ``paths``/``path``, and no other two sources may be combined.
With no ``output_dir`` the rendered Markdown is returned in the chat and nothing is
written; with one, the pipeline persists the file.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

from agent.engine.decisions import ToolExecution, text_block
from agent.plugins.base import Plugin
from agent.tools.definition import ToolOutput, define_tool
from agent.tools.tool_permissions import ToolPermission
from api.tools.document_content import UnsupportedDocument, read_asset_document
from core.infrastructure.pdf_pages import parse_pages_spec

from .pipeline import ToolKitPipeline
from .sources import WorkspaceSource, from_text
from .summary_auto import run_auto_summary

logger = logging.getLogger(__name__)

_DESCRIPTIONS = {
    "summary": (
        "Generate a grounded Markdown research summary (executive summary, key points, "
        "per-section analysis, Q&A) from exactly ONE content source: workspace files "
        "(`paths` / `path`), a document the user attached to the chat or has open in the "
        "viewer (`asset_id`, optionally narrowed by `pages`), or the conversation itself "
        "(`context`). `auto` is set by the intent funnel, which then decides the source "
        "from the turn. The summary is returned in the chat unless the user explicitly "
        "asked to save it."
    ),
    "mindmap": (
        "Generate a Mermaid mind map (.mmd, depth <= 4) from one or more workspace files, "
        "converging on a single central topic."
    ),
    "slides": (
        "Generate a slide deck from one or more workspace files: canonical 16:9 PDF "
        "(Typst-compiled) plus Marp Markdown and .pptx compat exports and the deck.json "
        "model; one core idea per slide with grounded citations."
    ),
}

_PLUGIN_DESCRIPTIONS = {
    "summary": "Grounded summary generator (toolkit_summary).",
    "mindmap": "Mermaid mind map generator (toolkit_mindmap).",
    "slides": "Slide deck generator, PDF + Marp + .pptx (toolkit_slides).",
}

_PATHS_PROPERTY = {
    "type": "array",
    "minItems": 1,
    "items": {"type": "string"},
    "description": "Workspace-relative file paths (single or multiple).",
}

_OUTPUT_DIR_PROPERTY = {
    "type": "string",
    "description": "Optional workspace-relative output directory "
    "(default <toolkit_output_dir>/<tool>).",
}

# The summary-specific source slots. Every one is a string: the funnel/executor lane
# binds strings only, and each value is resolved before the tool body runs.
_SUMMARY_PROPERTIES = {
    "paths": _PATHS_PROPERTY,
    "path": {
        "type": "string",
        "description": "ONE workspace-relative file path. Use `paths` for several.",
    },
    "asset_id": {
        "type": "string",
        "description": "Cloud-drive asset id of a document the user attached or has open "
        "in the viewer ([Attached: ...] / Viewer Access Context note).",
    },
    "pages": {
        "type": "string",
        "description": 'When the source is an asset_id document: 1-based page/slide spec '
        '("3", "2,5", "1-3"). Omit to summarize the whole document. Only PDF and '
        "PowerPoint slide-based formats (PPTX/POTX/PPSX) support pages.",
    },
    "context": {
        "type": "string",
        "description": "The conversation transcript to summarize instead of a file "
        "(e.g. \"summarize what we just discussed\").",
    },
    "output_dir": _OUTPUT_DIR_PROPERTY,
    "auto": {
        "type": "string",
        "description": "Internal marker set by the intent funnel: decide the source "
                       "from the turn material (previous message / open document / "
                       "session) and the save intent in the first call. Do not set "
                       "manually.",
    },
}


def _tool_parameters(tool: str) -> dict:
    if tool == "summary":
        # No slot is globally required: the ONE source comes from whichever legal
        # producer the turn has, and the body rejects an empty/multi-source call
        # with a message that names the alternatives. The EMPTY ``required`` list
        # is load-bearing: the executor gate treats a missing ``required`` key as
        # "every slot required".
        return {"type": "object", "properties": dict(_SUMMARY_PROPERTIES),
                "required": []}
    params: dict = {
        "type": "object",
        "properties": {"paths": _PATHS_PROPERTY, "output_dir": _OUTPUT_DIR_PROPERTY},
        "required": ["paths"],
    }
    if tool == "slides":
        params["properties"]["count"] = {
            "type": "integer",
            "minimum": 3,
            "maximum": 20,
            "description": "Number of slides to generate (3-20, default 8).",
        }
    return params


def _render_result(args: dict, value: dict) -> list:
    lines = [f"Generated **{value['tool']}** output."]
    lines += [f"- {f}" for f in value["files"]]
    return [text_block("\n".join(lines))]


class _SummaryContent(NamedTuple):
    """The ONE resolved content source, in the form its lane needs.

    ``paths`` is the workspace lane: the pipeline still validates and extracts those
    files (they are only known by path here). ``sources`` is the already-read lane —
    an authorized document or the conversation itself, whose bytes the caller's own
    reader produced. Exactly one of the two is non-empty.
    """

    paths: tuple[str, ...] = ()
    sources: tuple[WorkspaceSource, ...] = ()
    stem: str = "summary"


async def _summary_content(args: dict, ctx, llm) -> _SummaryContent:
    """Resolve the ONE content source into the lane it belongs to.

    Precedence is narrow and deliberate: ``context`` (the conversation) outranks an
    ``asset_id`` the Binder sources from turn facts — a document that happens to be
    open must never be read for a "summarize our chat" request — and is rejected
    together with a workspace path. Every other source is exclusive: the capability
    handler emits exactly one, so a call naming two is a caller error, not a case to
    silently resolve.

    Raises :class:`ValueError` with a caller-readable message when the call names
    no source, conflicting sources, an unreadable asset, a page spec on a format
    without a page axis, an image, or a document with no extractable text.
    Nothing here invents a path, an asset id or content.
    """
    raw_paths = args.get("paths") or []
    path = str(args.get("path") or "").strip()
    asset_id = str(args.get("asset_id") or "").strip()
    context = str(args.get("context") or "").strip()

    if not isinstance(raw_paths, list) or not all(
            isinstance(p, str) and p.strip() for p in raw_paths):
        raise ValueError("paths must be a non-empty list of workspace file paths")
    paths = [p for p in raw_paths]
    if paths and path:
        raise ValueError("give either paths or path, not both")

    pages_raw = args.get("pages")
    if pages_raw is not None and not isinstance(pages_raw, str):
        raise ValueError('pages must be a string like "3" or "1-3"')

    if context:
        if paths or path:
            raise ValueError(
                "context summarizes the conversation and cannot be combined with "
                "paths/path; choose one content source")
        if pages_raw is not None:
            raise ValueError(
                "pages applies only to a document, never to the conversation context")
        return _SummaryContent(sources=(from_text("conversation", context),),
                               stem="conversation")

    if paths or path:
        if pages_raw is not None:
            raise ValueError(
                "pages applies only to an asset_id document read; omit it for "
                "workspace files (the whole file is summarized)")
        if asset_id:
            raise ValueError(
                "summary_gen needs exactly ONE content source; got a workspace path "
                "and an asset_id")
        selected = paths or [path]
        return _SummaryContent(paths=tuple(selected),
                               stem=(Path(selected[0]).stem or "summary"))

    if not asset_id:
        raise ValueError(
            "summary_gen needs exactly ONE content source: paths (workspace files), "
            "path (one workspace file), asset_id (an attached/viewer document), or "
            "context (the conversation)")
    if ctx is None:
        raise ValueError(
            "summary_gen cannot read an attached document without a runtime context")

    pages = parse_pages_spec(pages_raw) if pages_raw is not None else None
    try:
        doc = await read_asset_document(ctx, llm, asset_id, pages=pages,
                                        caller="summary_gen")
    except UnsupportedDocument as exc:
        raise ValueError(
            f"cannot extract text from '{exc.name}': {exc.reason} — tell the user this "
            "file could not be summarized; do not substitute another document") from None
    if doc.is_image:
        raise ValueError(
            f"'{doc.name}' is an image; use the vision tool to analyze it instead")
    if not doc.text.strip():
        raise ValueError(
            f"'{doc.name}' contains no extractable text (it may be a scanned, "
            "image-only document), so it cannot be summarized")
    return _SummaryContent(sources=(from_text(doc.name, doc.text),),
                           stem=(Path(doc.name).stem or "summary"))


def build_toolkit_plugin(tool: str, llm, events=None, workspace=None,
                         ctx=None) -> Plugin:
    """Build one Cordis-style plugin for ``tool`` (summary / mindmap / slides).

    ``ctx`` is the tool runtime context; ``summary_gen`` needs it to read an
    authorized ``asset_id`` through the shared document reader. The other tools
    never touch it.
    """
    pipeline = ToolKitPipeline(llm, tool, events=events, workspace=workspace)

    async def _execute_summary(args: dict) -> str:
        """Return the Markdown summary itself, or a one-line saved-file note.

        ``auto`` is the intent-funnel marker: the source was NOT decided upstream,
        so the two-stage flow in :mod:`summary_auto` owns it (first call decides
        target / readiness / save intent, second call reads only the chosen
        material). An explicit source on the call always takes the deterministic
        path below, so the Agent lane is unchanged.

        The chat-window lane (``persist=False``) returns the rendered Markdown so the
        ACTION result carries the actual summary. The explicit ``output_dir`` lane
        writes the artifact and returns its location, keeping the (potentially long)
        body out of the tool result.
        """
        auto = str(args.get("auto") or "").strip()
        explicit_source = any(str(args.get(key) or "").strip()
                              for key in ("paths", "path", "context"))
        if auto and not explicit_source:
            return await run_auto_summary(
                args=args, ctx=ctx, llm=llm, pipeline=pipeline)
        content = await _summary_content(args, ctx, llm)
        output_dir = str(args.get("output_dir") or "").strip()
        persist = bool(output_dir)
        if content.paths:
            # Workspace files still go through validate + ingest: their text is not
            # in hand yet, and path confinement is the pipeline's job.
            result = await pipeline.run(list(content.paths), output_dir or None,
                                        persist=persist)
        else:
            result = await pipeline.run_sources(list(content.sources),
                                                output_dir or None,
                                                persist=persist, stem=content.stem)
        if not persist:
            return result.text
        lines = [result.summary, *(f"- {f}" for f in result.files)]
        return "\n".join(lines)

    async def execute(args: dict, exec: ToolExecution) -> dict | str:
        if tool == "summary":
            return await _execute_summary(args)
        params = {k: v for k, v in args.items() if k not in ("paths", "output_dir")}
        result = await pipeline.run(args["paths"], output_dir=args.get("output_dir"), **params)
        # The ToolOutput schema is ``{"type": "object"}``: hand the model a structured
        # dict (tool / files / summary) rather than the internal result dataclass.
        return {"tool": result.tool, "files": result.files, "summary": result.summary}

    tool_def = define_tool(
        name=f"{tool}_gen",
        description=_DESCRIPTIONS[tool],
        parameters=_tool_parameters(tool),
        output=(
            ToolOutput(schema={"type": "string"},
                       render=lambda args, value: [text_block(value)])
            if tool == "summary"
            else ToolOutput(schema={"type": "object"}, render=_render_result)
        ),
        execute=execute,
        is_concurrency_safe=True,
        permission={ToolPermission.READ, ToolPermission.WRITE},
    )

    return Plugin(
        name=f"toolkit_{tool}",
        description=_PLUGIN_DESCRIPTIONS[tool],
        tools=[tool_def],
        # Unique capability name per tool (the Context rejects duplicate providers, and all
        # three pipelines would otherwise collide on a shared "pipeline" capability).
        provides={f"{tool}_pipeline": pipeline},
    )
