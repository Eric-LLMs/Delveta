"""The side-effect boundary contract at the real dispatch seam (Phase 4-A / A-1).

Pins ``ActionExecutor._dispatch`` (action.py) at the four outcomes that decide
whether an effect is allowed to exist:

  * PREFLIGHT failure  — the body proved nothing executed ⇒ escalate to the Agent,
    which receives the original user text, with ZERO side effect;
  * POST-WRITE failure — the body was entered and its write recorded ⇒ STATE
    UNKNOWN terminal, exactly ONE effect, NEVER a blind Agent retry;
  * decided DENIAL     — ``{"ok": False}`` ⇒ an honest terminal message, zero effect;
  * Binder BLOCK       — a required slot the handler cannot source ⇒ the executor is
    never reached (no side effect), the Agent owns the clarification.
"""
from __future__ import annotations

from tests.e2e.harness import CAP_BY_ID, build_stack, sse_for


async def test_preflight_failure_escalates_with_no_effect(monkeypatch):
    """DriveError before any write ⇒ preflight ⇒ the Agent takes the ORIGINAL text."""
    cap = CAP_BY_ID["cap-create-folder"]
    stack = build_stack(monkeypatch, drive_mode="preflight")

    await sse_for(stack, cap.message)

    assert stack.spy.folders_created == []          # provably no side effect
    assert stack.port.steps >= 1                    # escalated to the Agent
    assert stack.recorder.call_count == 1           # the failed tool result was recorded
    assert stack.recorder.calls[0]["is_error"] is True
    # The Agent's conversation carries the user's ORIGINAL sentence, byte-for-byte.
    flat = " ".join(str(m.get("content", "")) for m in stack.port.requests[-1])
    assert cap.message in flat


async def test_post_write_failure_is_terminal_single_effect(monkeypatch):
    """A failure AFTER the write ⇒ STATE UNKNOWN terminal, one effect, no retry."""
    cap = CAP_BY_ID["cap-create-folder"]
    stack = build_stack(monkeypatch, drive_mode="post-write")

    res = await sse_for(stack, cap.message)

    assert "could not be confirmed" in (res.answer or "")
    assert len(stack.spy.folders_created) == 1      # exactly ONE effect, never two
    assert stack.port.steps == 0                    # the Agent was never re-run


async def test_decided_denial_is_terminal_no_effect(monkeypatch):
    """A denied approval ⇒ honest terminal denial, zero side effect."""
    cap = CAP_BY_ID["cap-create-folder"]
    stack = build_stack(monkeypatch, broker_mode="deny")

    res = await sse_for(stack, cap.message)

    assert (res.answer or "").startswith("Could not complete that request: ")
    assert stack.spy.folders_created == []
    assert stack.port.steps == 0


async def test_binder_block_never_reaches_the_seam(monkeypatch):
    """A required slot no source fills ⇒ Binder blocks, Agent clarifies.

    The model fills only ``term`` (``domain`` is omitted — the sentence names no
    resolvable domain), so ``add_term``'s certified draft leaves the required
    ``domain`` MISSING and the Binder yields BIND_MISSING.
    """
    stack = build_stack(
        monkeypatch, query_overrides={"cap-add-term": "add quantum to the glossary"},
        slot_values={"cap-add-term": {"term": "quantum"}},   # the model omits domain
    )

    res = await sse_for(stack, "add quantum to the glossary")

    assert stack.recorder.call_count == 0           # the tool body was never entered
    assert stack.spy.terms_added == []
    assert stack.port.steps >= 1                    # the Agent owns the clarification
    assert res.approvals == []
