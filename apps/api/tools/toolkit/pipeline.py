"""The shared 5-stage toolkit pipeline: the lifecycle engine behind every tool.

Each run passes through five explicit stages, each with a named Cordis-style hook pair so
other plugins can intercept or observe a stage by registering on the shared :class:`EventBus`
(the same bus the agent's tool lifecycle uses):

    toolkit/before-validate  (waterfall; raise to abort)   toolkit/after-validate  (observer)
    toolkit/before-ingest    (waterfall; raise to abort)   toolkit/after-ingest    (observer)
    toolkit/before-generate  (waterfall; raise to abort)   toolkit/after-generate  (observer)
    toolkit/before-render    (waterfall; raise to abort)   toolkit/after-render    (observer)
    toolkit/before-persist   (waterfall; raise to abort)   toolkit/after-persist   (observer)

Stage order and responsibilities:

1. **validate** — path safety (workspace escape rejected), existence, per-file size cap.
2. **ingest**   — text extraction; the FULL raw text always goes downstream (never a
   digest). ``toolkit_max_input_tokens`` is a capacity check only.
3. **generate** — structured JSON via JSON mode, jsonschema-validated, one retry that
   carries the concrete schema errors back into the prompt. When the complete input
   exceeds the one-shot capacity, the EXPLICIT big-document multi-call flow runs instead:
   one raw-grounded call per batch (:mod:`sources.plan_big_document`) + deterministic
   merge — never a summary as sole input.
4. **render**   — structured JSON → Mermaid / Marp / summary Markdown / .pptx (never raw
   model-written diagram markup).
5. **persist**  — atomic write into the workspace output dir with collision-proof names.
   SKIPPED (``persist=False``) when the caller wants the rendered text itself — the
   chat-window summary lane — so nothing is written to disk on that path.

Every stage raises a :class:`ToolKitError` subclass; ``run`` maps any other failure to a
generic :class:`ToolKitError` so callers get one readable message.
"""
from __future__ import annotations

import asyncio
import logging
import os
import secrets
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from agent.tools.fs_tools import _atomic_write, _resolve
from core.config import settings
from core.infrastructure import media as media_lib

from . import outputs
from .errors import GenerationError, PersistError, SourceError, ToolKitError
from .prompts import SYSTEM_PROMPTS, build_user_prompt
from .sources import (
    WorkspaceSource,
    load_sources,
    plan_big_document,
    remap_citations,
    total_input_tokens,
)

logger = logging.getLogger(__name__)

HOOK_PREFIX = "toolkit"
TOOLS = ("summary", "mindmap", "slides")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Binary twin of ``fs_tools._atomic_write`` (temp file + os.replace, no torn file)."""
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


@dataclass
class ToolKitResult:
    """The outcome of one pipeline run.

    ``files`` is non-empty only when the persist stage ran. ``text`` carries the
    rendered primary artifact for a run that did NOT persist (the chat-window
    summary lane), so the caller can return it without touching the filesystem.
    """

    tool: str
    files: list[str] = field(default_factory=list)   # absolute paths of written artifacts
    summary: str = ""                                 # one-line human summary
    stats: dict = field(default_factory=dict)         # deck per-stage instrumentation
    text: str = ""                                    # rendered Markdown when not persisted


class ToolKitPipeline:
    """A configured pipeline for one tool (``summary`` / ``mindmap`` / ``slides``)."""

    def __init__(
        self,
        llm,
        tool: str,
        *,
        workspace: Path | None = None,
        events=None,
        output_dir: Path | str | None = None,
    ) -> None:
        if tool not in TOOLS:
            raise ValueError(f"unknown toolkit tool: {tool!r}")
        self.llm = llm
        self.tool = tool
        self.workspace = Path(workspace or settings.workspace_dir)
        # Per-tool default output root under the configured toolkit output dir.
        self.output_dir = (
            Path(output_dir)
            if output_dir is not None
            else Path(settings.toolkit_output_dir) / tool
        )
        self.events = events

    # ── Cordis-style hook firing ──
    async def _hook(self, name: str, *payload) -> None:
        """Waterfall hook: a registered listener can raise to abort the stage."""
        if self.events is not None:
            await self.events.waterfall(f"{HOOK_PREFIX}/{name}", *payload, base=None)

    async def _observe(self, name: str, payload) -> None:
        """Observer hook: listeners run read-only; their errors are logged, never fatal."""
        if self.events is not None:
            await self.events.serial(f"{HOOK_PREFIX}/{name}", payload)

    # ── public entry ──
    async def run(self, paths: list[str], output_dir: str | None = None, *,
                  persist: bool = True, **params) -> ToolKitResult:
        """Run the full lifecycle over workspace files; raises :class:`ToolKitError`.

        ``persist=False`` renders and returns ``result.text`` instead of writing an
        artifact (the explicit output directory is then unused and not resolved).
        """
        try:
            return await self._run(paths, output_dir, params, persist=persist)
        except ToolKitError:
            raise
        except Exception as exc:
            raise ToolKitError(f"{self.tool} generation failed: {exc}") from exc

    async def run_sources(self, sources: list[WorkspaceSource],
                          output_dir: str | None = None, *,
                          persist: bool = True, stem: str = "artifact",
                          **params) -> ToolKitResult:
        """Run generation/render/(persist) over ALREADY-EXTRACTED text.

        For a source that has no workspace path — an authorized drive document or
        the conversation itself — the validate/ingest stages do not apply, because
        the caller already read the bytes through its own authorized reader. The
        generate/render/persist stages are the SAME code ``run`` uses.
        """
        try:
            if not sources:
                raise SourceError("no content to summarize")
            return await self._run_sources(sources, output_dir, params,
                                           persist=persist, stem=stem)
        except ToolKitError:
            raise
        except Exception as exc:
            raise ToolKitError(f"{self.tool} generation failed: {exc}") from exc

    async def _run(self, paths: list[str], output_dir: str | None, params: dict, *,
                   persist: bool) -> ToolKitResult:
        await self._hook("before-validate", paths)
        resolved, out_dir = await self.stage_validate(
            paths, output_dir, resolve_output=persist)
        await self._observe("after-validate", {
            "paths": [str(p) for p in resolved],
            "output_dir": str(out_dir) if out_dir is not None else ""})

        await self._hook("before-ingest", [str(p) for p in resolved])
        sources = await self.stage_ingest(resolved)
        await self._observe("after-ingest", sources)

        return await self._finish(
            sources, params, persist=persist,
            stem=resolved[0].stem if resolved else "artifact",
            out_dir=out_dir if persist else None)

    async def _run_sources(self, sources: list[WorkspaceSource],
                           output_dir: str | None, params: dict, *,
                           persist: bool, stem: str) -> ToolKitResult:
        # No validate/ingest hooks: there is no path to validate and no bytes to
        # extract — the caller's reader owns both, and it already succeeded.
        return await self._finish(
            sources, params, persist=persist, stem=stem,
            out_dir=self._resolve_output_dir(output_dir) if persist else None)

    async def _finish(self, sources: list[WorkspaceSource], params: dict, *,
                      persist: bool, stem: str, out_dir: Path | None) -> ToolKitResult:
        """generate → render → persist (or return the rendered text)."""
        self._deck_stats = {}

        # One-shot capacity check (never a compression trigger — see sources.py).
        batches = plan_big_document(sources)
        if batches:
            logger.warning(
                "toolkit %s: input of %d tokens exceeds the %d-token one-shot capacity — "
                "entering the EXPLICIT big-document multi-call flow: %d raw-grounded "
                "call(s), structurally merged; the document is never replaced by a summary",
                self.tool, total_input_tokens(sources),
                settings.toolkit_max_input_tokens, len(batches))
        bigdoc_calls = len(batches) if batches else None

        await self._hook("before-generate", sources)
        data = await self.stage_generate(sources, params, batches=batches)
        await self._observe("after-generate", data)

        await self._hook("before-render", data)
        rendered = await self.stage_render(data)
        await self._observe("after-render", rendered)

        if not persist:
            # Nothing is written: the persist hooks never fire, so an observer can
            # never mistake a chat-window result for a persisted artifact.
            return self._rendered_result(rendered, bigdoc_calls=bigdoc_calls)

        await self._hook("before-persist", rendered)
        result = await self.stage_persist(rendered, out_dir, stem=stem,
                                          bigdoc_calls=bigdoc_calls)
        await self._observe("after-persist", result)
        # Per-stage instrumentation (deck only): real LLM calls, provider token counts,
        # seconds and repair events per pass — carried on the result for the job record.
        if getattr(self, "_deck_stats", None) and isinstance(result, ToolKitResult):
            result.stats = self._deck_stats
        return result

    def _rendered_result(self, rendered: dict[str, object], *,
                         bigdoc_calls: int | None = None) -> ToolKitResult:
        note = (
            f" Large input: {bigdoc_calls} raw grounding calls "
            "(explicit big-document flow)."
            if bigdoc_calls else ""
        )
        primary = rendered.get("summary.md") or ""
        return ToolKitResult(
            tool=self.tool, files=[],
            summary=f"Rendered {self.tool} output (not saved).{note}",
            stats=dict(getattr(self, "_deck_stats", None) or {}),
            text=str(primary),
        )

    # ── stage 1: validate ──
    def _resolve_output_dir(self, output_dir: str | None) -> Path:
        """The caller's explicit directory, else the per-tool default; workspace-confined.

        Resolution happens only for a run that will persist — a chat-window run
        writes nothing, so an unusable destination must not fail it.
        """
        target = self.output_dir if output_dir is None else Path(output_dir)
        try:
            return _resolve(self.workspace, str(target))
        except ValueError as exc:
            raise SourceError(f"output dir escapes workspace: {output_dir}") from exc

    async def stage_validate(self, paths: list[str], output_dir: str | None, *,
                             resolve_output: bool = True,
                             ) -> tuple[list[Path], Path | None]:
        """Path safety + existence + size gates; returns ``(resolved_paths, out_dir)``.

        ``resolve_output=False`` (a run that will NOT persist) skips output-dir
        resolution: the chat-window lane writes nothing, so an unusable
        destination must never fail it.
        """
        if not paths:
            raise SourceError("no files selected")
        resolved: list[Path] = []
        for raw in paths:
            try:
                path = _resolve(self.workspace, raw)
            except ValueError as exc:
                raise SourceError(str(exc)) from exc
            if not path.is_file():
                raise SourceError(f"not a file: {raw}")
            if path.stat().st_size > settings.toolkit_max_file_bytes:
                max_mb = settings.toolkit_max_file_bytes // (1024 * 1024)
                raise SourceError(f"file too large (max {max_mb} MB): {raw}")
            resolved.append(path)

        return resolved, (self._resolve_output_dir(output_dir)
                          if resolve_output else None)

    # ── stage 2: ingest ──
    async def stage_ingest(self, resolved: list[Path]) -> list[WorkspaceSource]:
        """Extract text; the FULL text goes to generation verbatim (no digest/trimming)."""
        return await load_sources(self.workspace, resolved, self.llm)

    # ── stage 3: generate ──
    async def stage_generate(
        self, sources: list[WorkspaceSource], params: dict, *,
        batches: list[list[WorkspaceSource]] | None = None,
    ) -> dict:
        """Structured generation with schema validation and corrective retries.

        ``batches`` (from :func:`sources.plan_big_document`) engages the EXPLICIT
        big-document multi-call flow: one grounding call per batch, each on that batch's
        RAW text, plus a deterministic structural merge. Below capacity (``batches`` None)
        the complete raw text goes to the generator in one call — always verbatim, never
        digested.

        ``slides`` runs the grounded visual presentation engine: multimodal ingest
        (:mod:`.deck.ingest`, zero LLM) → the generation engine selected by
        ``generation_mode`` (default ``slides_generation_mode``): ``direct`` =
        :mod:`.deck.generator` (one semantic call + local compiler), ``legacy`` =
        the presentation-brief workflow (:mod:`.deck.workflow_driver`). Both emit
        the canonical :class:`~.deck.schema.PresentationBrief`. Other tools keep
        the single schema-validated call.

        A per-task custom prompt (``params["prompt"]``, from the generation dialog) is
        appended to the tool's default system prompt, never replacing it — the default
        carries the JSON/schema constraints that keep the pipeline working, and the user's
        own requirements layer on top. An empty/missing prompt uses the default alone.
        """
        if self.tool == "slides":
            from .deck.generator import run_direct_generation
            from .deck.ingest import build_document_representation, clean_source_name
            from .deck.schema import PresentationControls
            from .deck.workflow_driver import run_presentation_workflow

            controls = PresentationControls(
                target_audience=str(params.get("audience") or ""),
                presentation_goal=str(params.get("goal") or ""),
                target_slide_count=int(params["count"]) if params.get("count") else 8,
                language=str(params.get("language") or ""),
                format_mode=str(params.get("format_mode") or "detailed"),
                user_guidance=(params.get("prompt") or "").strip(),
            )
            deck_id = secrets.token_hex(4)
            doc_rep = await build_document_representation(
                sources, workspace=self.workspace, deck_id=deck_id)
            deck_stats: dict = {}
            # Attach BEFORE the await: the engine fills this dict in place, so a
            # mid-run failure still leaves the partial per-node stats observable.
            self._deck_stats = deck_stats
            # Engine switch: "direct" = one semantic LLM call + the local
            # compiler (default); "legacy" = the Brief chain (TEXT→VISUAL→REDUCE→
            # SYNTHESIZE) over the generic workflow core, kept whole as fallback.
            mode = str(params.get("generation_mode")
                       or settings.slides_generation_mode or "direct").lower()
            if mode == "legacy":
                # The brief chain chunks and grounds internally (per-block Pass A calls
                # on RAW text), so the batch plan is not consumed here.
                brief = await run_presentation_workflow(
                    self.llm, doc_rep, controls, deck_id=deck_id, stats_out=deck_stats)
            else:
                brief = await run_direct_generation(
                    self.llm, doc_rep, controls, deck_id=deck_id, stats_out=deck_stats)
            return {
                "brief": brief.model_dump(mode="json"),
                "document_title": doc_rep.document_title,
                "presentation_goal": controls.presentation_goal,
                "source_names": [clean_source_name(s.name) for s in sources],
                # the render stage re-embeds referenced slices by asset id (§9.3)
                "visual_assets": [a.model_dump(mode="json")
                                  for a in doc_rep.visual_assets],
            }

        system = SYSTEM_PROMPTS[self.tool]
        custom = (params.get("prompt") or "").strip()
        if custom:
            system = f"{system}\n\n{custom}"
        if not batches:
            prompt = build_user_prompt(self.tool, sources, params)
            return await self._generate_validated(prompt, system)

        partials: list[dict] = []
        n = len(batches)
        for i, batch in enumerate(batches, 1):
            prompt = (
                f"(Big-document batch {i} of {n}: generate this part of the {self.tool} "
                f"strictly from the sources below.)\n\n"
                + build_user_prompt(self.tool, batch, params)
            )
            data = await self._generate_validated(prompt, system)
            # Batch-relative [name:line] citations → absolute original lines before merge.
            partials.append(remap_citations(data, {s.name: s.line_offset for s in batch}))
        return outputs.merge_tool_outputs(self.tool, partials)

    async def _generate_validated(self, prompt: str, system: str) -> dict:
        """One JSON generation + schema validation + single corrective retry."""
        data = await self._complete_json(prompt, system)

        errors = outputs.validate(outputs.SCHEMAS[self.tool], data)
        if errors:
            retry_prompt = (
                "Your previous reply failed JSON schema validation:\n"
                + "\n".join(f"- {e}" for e in errors[:6])
                + "\n\nFix the reply to conform to the schema. Reply with JSON only.\n\n"
                + prompt
            )
            data = await self._complete_json(retry_prompt, system)
            errors = outputs.validate(outputs.SCHEMAS[self.tool], data)
        if errors:
            raise GenerationError("; ".join(errors[:6]))
        return data

    async def _complete_json(self, prompt: str, system: str) -> dict:
        """JSON mode when the client supports it; else a tolerant ``complete`` + parse.

        Toolkit calls run on (near-)full-context inputs: the global per-call timeout is
        too tight for them, so ``toolkit_llm_timeout_s`` bounds each attempt instead.
        """
        timeout = settings.toolkit_llm_timeout_s
        fn = getattr(self.llm, "complete_json", None)
        if fn is not None:
            try:
                return await fn(prompt, system, timeout=timeout)
            except Exception as exc:  # noqa: BLE001 - JSON mode is best-effort; fall back
                logger.info("complete_json unavailable (%s); falling back to tolerant parse", exc)
        raw = await self.llm.complete(prompt, system, timeout=timeout)
        data = outputs.extract_json(raw)
        if data is None:
            raise GenerationError("model response was not valid JSON")
        return data

    # ── stage 4: render ──
    async def stage_render(self, data: dict) -> dict[str, object]:
        """Structured JSON → final display formats (never raw model-written markup).

        Slides render from the canonical :class:`~.deck.schema.PresentationBrief`
        through the Visual Compiler (:mod:`.deck.compiler`): the brief + the sliced
        assets re-emit ``deck.pdf`` with zero LLM calls (§1.2.6), and the compat
        exports (Marp Markdown, .pptx inputs) derive from the same brief.
        ``deck.json`` is the brief itself. The RenderReport gate stays loud —
        nothing silently degrades; template fallbacks surface as ``layout_warnings``.
        """
        if self.tool == "slides":
            import json as _json
            import tempfile
            from pathlib import Path as _P

            from .deck.render import brief_to_marp, brief_to_pptx, render_brief_pdf
            from .deck.schema import PresentationBrief, VisualAsset

            brief = PresentationBrief.model_validate(data["brief"])
            assets = [VisualAsset.model_validate(a)
                      for a in data.get("visual_assets") or []]
            with tempfile.TemporaryDirectory(prefix="deck_") as td:
                res = await asyncio.to_thread(
                    render_brief_pdf, brief, assets, _P(td),
                    document_title=data.get("document_title", ""),
                    source_names=list(data.get("source_names") or []))
            # Render instrumentation (zero LLM): the page contract stated with the
            # cover counted separately from the content slides, and the compiler's
            # explicit degradations — on success they had no home before, which
            # made "a grammar that never landed" invisible in the job record.
            if getattr(self, "_deck_stats", None):
                self._deck_stats["E/render"] = {
                    "calls": 0, "llm_seconds": 0.0,
                    "local_seconds": None,
                    "pages_expected": res.report.pages_expected,
                    "pages_actual": res.report.pages_actual,
                    "cover_pages": 1,
                    "content_slides": len(brief.slides),
                    "layout_warnings": list(res.report.layout_warnings),
                }
            if not res.report.ok:
                raise GenerationError(
                    "deck PDF render failed: "
                    + f"pages {res.report.pages_actual}/{res.report.pages_expected}; "
                    + "; ".join(res.report.typst_warnings[:3]))
            return {
                "deck.pdf": res.pdf,
                "deck.json": _json.dumps(brief.model_dump(mode="json"),
                                         ensure_ascii=False),
                "deck.md": brief_to_marp(
                    brief, document_title=data.get("document_title", "")),
                "deck.pptx": brief_to_pptx(
                    brief, assets,
                    document_title=data.get("document_title", ""),
                    source_names=list(data.get("source_names") or [])),
            }
        return outputs.render(self.tool, data)

    # ── stage 5: persist ──
    async def stage_persist(
        self, rendered: dict[str, object], out_dir: Path, *, stem: str,
        bigdoc_calls: int | None = None,
    ) -> ToolKitResult:
        """Write every rendered artifact atomically; collision-proof names, workspace-confined.

        ``bigdoc_calls`` (set when the explicit big-document multi-call flow ran) is
        surfaced in the one-line summary so the job record states the flow used.
        """
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise PersistError(f"cannot create output dir {out_dir}: {exc}") from exc

        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(3)
        files: list[str] = []
        for logical, content in rendered.items():
            ext = logical.rsplit(".", 1)[-1]
            filename = f"{stem}_{stamp}.{ext}"
            path = out_dir / filename
            try:
                if isinstance(content, (bytes, bytearray)):
                    await asyncio.to_thread(_atomic_write_bytes, path, bytes(content))
                elif ext == "pptx":
                    # legacy tuple-input path (doc_slides); deck.pptx is real bytes now
                    await asyncio.to_thread(media_lib.build_text_pptx, content, path, title=stem)
                else:
                    await asyncio.to_thread(_atomic_write, path, content)
            except (OSError, ValueError) as exc:  # write error or a broken .pptx
                raise PersistError(f"failed to write {filename}: {exc}") from exc
            files.append(str(path))

        note = (
            f" Large input: {bigdoc_calls} raw grounding calls "
            "(explicit big-document flow)."
            if bigdoc_calls else ""
        )
        return ToolKitResult(
            tool=self.tool,
            files=files,
            summary=f"Generated {self.tool} output ({len(files)} file{'s' if len(files) != 1 else ''}).{note}",
            # Live-call and agent callers both get the rendered Markdown, so a
            # persisted run can still show what it wrote without a read-back.
            text=str(rendered.get("summary.md") or ""),
        )
