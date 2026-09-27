"""Matcher node — public surface (implementation in this package's modules)."""
from __future__ import annotations

from ..contract import MatchResult
from .index import build_index
from .matcher import match

__all__ = ["match", "build_index", "MatchResult"]
