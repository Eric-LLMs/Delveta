"""Pytest configuration for the in-process real-chain E2E suite (Phase 4-A / A-1).

Registers the ``live`` marker, which tags the A-2 live-stack tests (real
persistence / real network). A-1 only ships the in-process suite, so any
``live``-marked test is skipped here by existence (there are none yet).
"""
from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "live: A-2 live-stack E2E (real persistence / real network). Skipped in A-1.",
    )
