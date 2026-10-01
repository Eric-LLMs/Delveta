"""Toolkit pipeline factory — standalone pipelines for the async worker job.

:func:`pipeline_for` builds a pipeline for one toolkit tool (``/toolkit/generate``)
that shares the exact same lifecycle engine as the registered Cordis plugins, but
with no EventBus hooks wired.
"""
from __future__ import annotations

from .pipeline import ToolKitPipeline


def pipeline_for(tool: str, llm) -> ToolKitPipeline:
    """Standalone pipeline for the worker job (no EventBus hooks wired)."""
    return ToolKitPipeline(llm, tool)
