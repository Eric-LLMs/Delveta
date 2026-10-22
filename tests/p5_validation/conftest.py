"""Phase-5 lane validation is RETIRED, not silently green.

The whole Layer-A/B/C matrix in this directory drives the L0 classifier
(``resolve_requirements``) to produce HIGH-confidence requirement sets for the
DIRECT / VIEWER / LOCAL_RAG / COMPOSITE lanes. L0 was deleted: capability
selection now belongs to the Intent Funnel (Matcher HIT, or Recall +
LayaChoice-v2), and those experimental lanes have no requirement producer any
more. The artifacts stay on disk for reference; their cases are skipped rather
than deleted so the retirement is visible in every test run.

Re-enabling any lane requires a funnel-produced requirement set first; these
cases would then have to be re-authored against that producer.
"""
from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config, items):
    skip = pytest.mark.skip(
        reason="Phase-5 lane matrix retired with L0: no requirement producer "
               "for DIRECT/VIEWER/LOCAL_RAG/COMPOSITE",
    )
    for item in items:
        if item.path.parent.name == "p5_validation":
            item.add_marker(skip)
