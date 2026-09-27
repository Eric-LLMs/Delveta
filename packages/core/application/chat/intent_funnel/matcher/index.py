"""The Matcher's exact-lookup index: build from the Registry view, cached by fingerprint."""
from __future__ import annotations

from core.application.chat.intent_funnel.registry.entry import chat_plane_candidate

from .normalize import normalize

# fingerprint -> compiled index; the content key makes staleness impossible:
# any corpus change is a different fingerprint (and thus entry) by construction.
_INDEX_CACHE: dict[str, dict] = {}
_CACHE_MAX = 16


def build_index(view) -> dict:
    """normalized sentence -> {capability_id}, built from ``intent_corpus`` ONLY.

    Only chat-plane candidates are indexed (enabled AND active AND not owned
    by another lane's kind) — ruling 4: a disabled capability is not a
    candidate for ANY node, deterministic included."""
    key = view.fingerprint
    cached = _INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    exact: dict[str, set[str]] = {}
    for e in view.entries:
        if not chat_plane_candidate(e):
            continue
        for lit in e.intent_corpus:
            exact.setdefault(normalize(lit), set()).add(e.capability_id)
    if len(_INDEX_CACHE) >= _CACHE_MAX:
        _INDEX_CACHE.clear()
    _INDEX_CACHE[key] = exact
    return exact
