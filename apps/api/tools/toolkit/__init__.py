"""Toolkit content generation: workspace files → slides / mindmap / summary.

Public surface (implementations in their own modules):

- ``register_toolkit_plugins`` (:mod:`.registration`) — builds and registers the
  three Cordis plugins onto the agent's plugin manager.
- ``pipeline_for`` (:mod:`.factory`) — standalone pipeline for the async worker
  job (``/toolkit/generate``), sharing the same lifecycle engine.

This module only re-exports the public API (see CLAUDE.md, ``__init__.py``
Responsibility Rule).
"""
from __future__ import annotations

from .factory import pipeline_for
from .pipeline import TOOLS, ToolKitPipeline
from .plugins import build_toolkit_plugin
from .registration import register_toolkit_plugins

__all__ = [
    "TOOLS", "ToolKitPipeline", "build_toolkit_plugin",
    "pipeline_for", "register_toolkit_plugins",
]
