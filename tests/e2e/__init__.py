"""In-process real-chain E2E suite (Phase 4-A / A-1).

Drives a certified ACTION end-to-end through the REAL ToolRuntime into a real tool
body, and verifies the outcome DETERMINISTICALLY (真实 Tool Body 执行 + 确定性效果
验证). The outer world only (LLM port, external service seams, DB) is faked; the
router, orchestrator, funnel, matcher, acquisition, binder, executor, ``_run_tool``,
runtime, sandbox, approval and the tool body itself are all production code.

This is an in-process suite: it asserts a real, deterministic effect at the tool
boundary — it does NOT persist to a real database (that is A-2 Live's job).
"""
