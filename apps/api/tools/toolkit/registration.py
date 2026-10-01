"""Toolkit plugin registration — hook the three Cordis plugins to the runtime.

:func:`register_toolkit_plugins` builds the ``toolkit_slides`` /
``toolkit_mindmap`` / ``toolkit_summary`` plugins and registers them onto the
agent's plugin manager, so the model can call ``slides_gen`` / ``mindmap_gen``
/ ``summary_gen``. Their pipelines are wired to the runtime EventBus.
"""
from __future__ import annotations

from agent.plugins.manager import PluginManager

from .pipeline import TOOLS
from .plugins import build_toolkit_plugin


def register_toolkit_plugins(manager: PluginManager, ctx, llm, workspace=None) -> None:
    """Register the three toolkit plugins, hooking their pipelines to the runtime EventBus."""
    events = manager.runtime.events
    for tool in TOOLS:
        manager.register(build_toolkit_plugin(tool, llm, events=events, workspace=workspace))
