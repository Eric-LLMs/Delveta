"""Matcher query normalization — canonical text for the exact lookup.

Deliberately minimal (live-table ruling): the Matcher certifies a
HIT only when the turn's sentence IS a curated corpus phrasing, so the
normalization is the display-fidelity floor (surrounding space, case) shared
by the index build and the lookup. No semantic rewriting belongs here — that
is Recall's job.
"""
from __future__ import annotations


def normalize(s: str) -> str:
    return s.strip().casefold()
