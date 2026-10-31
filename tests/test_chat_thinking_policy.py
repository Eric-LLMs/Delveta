"""Global thinking policy at the API boundary.

The request flag is tri-state so "omitted" and "explicitly false" stay distinguishable:
omitted means the global default (OFF), and only an explicit ``false`` from the client
is a user opt-in to reasoning. FOCUS turns force OFF regardless.
"""
from apps.api.schemas import ChatRequest
from core.application.chat.context import _effective_disable_thinking


def test_omitted_flag_stays_none():
    body = ChatRequest(message="展开说说 MFU")
    assert body.disable_thinking is None


def test_explicit_values_survive_validation():
    assert ChatRequest(message="x", disable_thinking=True).disable_thinking is True
    assert ChatRequest(message="x", disable_thinking=False).disable_thinking is False


def test_focus_forces_thinking_off():
    assert _effective_disable_thinking(None, True) is True
    assert _effective_disable_thinking(False, True) is True


def test_other_turns_pass_the_request_through():
    assert _effective_disable_thinking(None, False) is None
    assert _effective_disable_thinking(True, False) is True
    assert _effective_disable_thinking(False, False) is False
