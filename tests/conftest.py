"""Collection boundaries for retired test artifacts.

The single-path ruling removed the four legacy chat lanes and the
``PolicyContext`` routing plan. The test scripts that exercised those paths are
kept VERBATIM in the tree as historical evidence: they are neither deleted nor
rewritten to fit the current architecture. Their imports no longer resolve, so
pytest cannot collect them at all.

``collect_ignore`` lists EXACTLY those unimportable files (never a directory).
It does not hide a live failure: every other test module is still collected and
run normally, and the retired files remain readable on disk. If one of the
surfaces they test ever returns, delete its entry here.
"""
from __future__ import annotations

# Removed executor modules (direct / viewer / retrieval / composite).
collect_ignore = [
    "test_chat_composite_executor.py",
    "test_chat_direct_executor.py",
    "test_chat_retrieval_executor.py",
    "test_chat_viewer_executor.py",
    # p5 validation modules: they import ``_p5_harness``, which imports the
    # removed ``PolicyContext``. The harness itself is not collected by pytest.
    "p5_validation/test_p5_smoke.py",
    "p5_validation/test_p5a_routing.py",
    "p5_validation/test_p5b_action_phrasing.py",
    "p5_validation/test_p5b_action_prod.py",
    "p5_validation/test_p5b_lifecycle_security.py",
    "p5_validation/test_p5b_rag_prod.py",
    "p5_validation/test_p5c_legacy_comparison.py",
]
