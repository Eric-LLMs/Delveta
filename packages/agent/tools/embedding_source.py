"""Attribute embedding calls to the tool call that issued them.

The embedder is shared: ``rag_search`` (chunk embedding) and ``memory_search``
(query embedding) both reach the same ``/embed`` endpoint, so a raw embedder log
line cannot tell which call paid for it. Each search tool installs this tag around
its own body only. The tag is a ``ContextVar``, so a concurrent sibling tool call
(its own asyncio task) never sees it, and ``finally`` restores the previous value
even when retrieval raises.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from core.infrastructure.request_context import (
    reset_request_embedding_source,
    set_request_embedding_source,
)


@contextmanager
def embedding_source(source: str, *, call_id: str = "", call_index: int = 0) -> Iterator[None]:
    """Tag every embedding issued inside this block with ``source``/``call_id``/``call_index``."""
    token = set_request_embedding_source(
        {"source": source, "call_id": call_id, "call_index": call_index}
    )
    try:
        yield
    finally:
        reset_request_embedding_source(token)
