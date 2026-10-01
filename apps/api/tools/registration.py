"""Built-in agent tool registration — directory discovery + registrar dispatch.

Each ``*_tool.py`` / ``*_tools.py`` module in this package exposes
``register(runtime, ctx, llm)``; :func:`register_builtin_tools` scans the
directory and calls it, so adding a tool is just dropping in a new file — no
central registry edit. Discovery covers both naming conventions: singular for
one-file-per-tool modules and plural for modules that bundle several sibling
tools (``pdf_tools``/``document_tools``) — a glob that matched only
``*_tool.py`` silently left those bundles unregistered (Ghost Tool: registry
caps pointing at tools no runtime ever had).

Domain tools (rag_search/translate/web_search) live in this package rather than
in the plugin registry because they depend on gateway-scoped resources: the
retrieval capability seam, the shared LLM client, and the web-search provider.
"""
from __future__ import annotations

import importlib
from pathlib import Path

from agent import Context, ToolRuntime

_REGISTRAR = "register"  # each *tool module exports `register(runtime, ctx, llm) -> None`


def register_builtin_tools(runtime: ToolRuntime, ctx: Context, llm) -> None:
    """Discover and register every ``*_tool`` module in this package."""
    package_dir = Path(__file__).parent
    module_files = sorted({*package_dir.glob("*_tool.py"), *package_dir.glob("*_tools.py")})
    for module_file in module_files:
        # ``__package__`` (``api.tools``) is the scan root, not this module's
        # ``__name__`` — the discovered modules are siblings in the package.
        module_name = f"{__package__}.{module_file.stem}"
        module = importlib.import_module(module_name)
        registrar = getattr(module, _REGISTRAR, None)
        if registrar is not None:
            registrar(runtime, ctx, llm)
