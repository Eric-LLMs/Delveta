"""Built-in agent tools — public surface.

Each ``*_tool.py`` / ``*_tools.py`` module in this package exposes
``register(runtime, ctx, llm)``; registration scans the directory and calls it,
so adding a tool is just dropping in a new file — no central registry edit.
Implementation lives in :mod:`.registration`; this module only re-exports the
public API (see CLAUDE.md, ``__init__.py`` Responsibility Rule).
"""
from __future__ import annotations

from .registration import register_builtin_tools

__all__ = ["register_builtin_tools"]
