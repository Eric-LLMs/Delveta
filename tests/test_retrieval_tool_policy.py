"""The retrieval policy the model is shown: soul.md plus the three tool descriptions.

These are contracts on the text itself — the wording is what makes general answers and
follow-ups skip retrieval, keeps ``web_search`` from becoming a RAG fallback, and stops
one retrieval target from being split into parallel paraphrases.
"""
from pathlib import Path

from agent.engine.runtime import ToolRuntime
from agent.memory.service import memory_search_tool

from apps.api.tools.rag_search_tool import register as register_rag
from apps.api.tools.web_search_tool import register as register_web

_SOUL = Path(__file__).resolve().parents[1] / "apps" / "api" / "soul.md"


class _Ctx:
    def resolve(self, key):
        return None


def _descriptions() -> dict[str, str]:
    runtime = ToolRuntime()
    register_rag(runtime, _Ctx(), None)
    register_web(runtime, _Ctx(), None)
    runtime.register(memory_search_tool(object()))
    return {name: runtime.get(name).description for name in ("rag_search", "memory_search", "web_search")}


def test_rag_search_is_scoped_to_the_users_own_material():
    description = _descriptions()["rag_search"]
    assert "OWN knowledge base" in description
    assert "does not hold general knowledge" in description


def test_memory_search_defers_to_the_current_conversation():
    description = _descriptions()["memory_search"]
    assert "SAVED long-term memory" in description
    assert "already in your context" in description


def test_web_search_is_not_a_rag_fallback():
    description = _descriptions()["web_search"]
    assert "real-time or external information" in description
    assert "NOT a fallback" in description


def test_soul_states_one_core_query_and_no_parallel_paraphrases():
    soul = _SOUL.read_text(encoding="utf-8")
    assert "Start with the ONE query" in soul
    assert "Chinese/English versions, synonyms or paraphrases" in soul
    assert "Always use the returned" in soul


def test_soul_stops_the_zero_hit_escalation_chain():
    soul = _SOUL.read_text(encoding="utf-8")
    assert "do not escalate to" in soul
    assert "not a fallback after `rag_search`" in soul


def test_soul_no_longer_pushes_retrieval_by_default():
    soul = _SOUL.read_text(encoding="utf-8")
    assert "search the corpus before guessing" not in soul
    assert "Ground answers in retrieved material" not in soul
    assert "retrieval is an option, not a default step" in soul


def test_delveta_manual_rule_survives():
    soul = _SOUL.read_text(encoding="utf-8")
    assert "Questions about using Delveta itself" in soul
    assert "a built-in product manual is part of the corpus" in soul
