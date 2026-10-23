"""TurnFacts → capability-draft IDENTITY — the shared asset sourcing.

Every file-content handler resolves WHICH asset it acts on the same way, so the
rule lives here exactly once and cannot drift between capabilities:

* ``asset_id`` — the ONLY legal source is :class:`~..contract.TurnFacts`, with the
  SAME precedence the Binder uses (``attachment_asset_id`` -> ``path_asset_id``
  -> ``viewer_asset_id``). No fact carries one -> ``""`` (the caller fails
  closed; a sentence-copied or model-invented id is never a source).

The page axis is deliberately NOT here: it is acquired by the combined
rule + online-LLM lane in
:mod:`..argument_acquisition.page_scope_settlement`. The viewer's open page/range
is never substituted for a page the user did not state — a whole-document
request stays whole-document even when a viewer page exists.
"""
from __future__ import annotations

# Mirrors binder._CONTEXT_SLOT_SOURCES["asset_id"] exactly (attachment -> path ->
# viewer), so the draft a handler emits is the value the Binder settles on.
_ASSET_FACT_FIELDS = ("attachment_asset_id", "path_asset_id", "viewer_asset_id")


def resolve_asset_id(facts) -> str:
    """The first non-empty asset id among the context facts, else ``""``."""
    if facts is None:
        return ""
    for field in _ASSET_FACT_FIELDS:
        value = str(getattr(facts, field, "") or "").strip()
        if value:
            return value
    return ""
