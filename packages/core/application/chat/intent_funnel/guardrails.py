"""Funnel common layer — the system-wide safety guards (ruling 8.1-a).

These guards are deliberately CODE, not Registry data: the configuration table
is "meant to be read and edited by humans" and does not fit negation/context
veto logic. The Matcher applies :func:`negated` before certifying a HIT — the
single call site: an explicit negation ("不要新建文件夹") must never certify a
write. There is no entry veto any more: a turn is routed by the Matcher /
Recall cascade itself, never re-routed before those nodes see it.
"""
from __future__ import annotations

from core.application.chat.actions import is_negated_request


def negated(message: str) -> bool:
    """The single global negation guard (8.1-a): "不要新建文件夹" is not a request."""
    return is_negated_request(message or "")
