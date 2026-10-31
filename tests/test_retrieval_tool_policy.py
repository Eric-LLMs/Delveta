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


# ── time-sensitive questions: search proactively, honestly, and only there ────────
# The model used to answer "I cannot check real-time weather" and ask whether to
# search. The contract is text the model is shown; these assertions pin the four rules
# the product depends on (search now, never confirm first, never invent, no overreach).

def _flat_soul() -> str:
    return " ".join(_SOUL.read_text(encoding="utf-8").split())


def test_soul_requires_a_same_turn_search_for_time_sensitive_questions():
    soul = _flat_soul()
    assert "Time-sensitive information" in soul
    assert "call `web_search` in this same turn" in soul
    assert "weather, prices, news, live scores, latest releases" in soul


def test_soul_forbids_asking_permission_before_a_search():
    soul = _flat_soul()
    assert "never ask the user whether they want an online search" in soul
    assert "do not say you cannot check real-time data without having tried" in soul


def test_soul_requires_honesty_when_the_search_fails_or_is_stale():
    soul = _flat_soul()
    assert "If the search fails or the results are clearly out of date, say so plainly" in soul
    assert "never invent numbers, dates, or headlines" in soul


def test_soul_keeps_ordinary_knowledge_out_of_the_search_rule():
    soul = _flat_soul()
    assert "Ordinary knowledge, explanations, summaries" in soul
    assert "not a reason to search everything" in soul


# ── search synthesis: raw results are material, the answer is the Agent's job ─────
# A weather turn once pasted every site's numbers, then averaged them into a made-up
# range and leaked URLs. These rules decide what the model is allowed to present.

def test_soul_synthesizes_instead_of_listing_each_source():
    soul = _flat_soul()
    assert "Search results are raw material, not the answer" in soul
    assert "never a per-site walkthrough, a source-by-source comparison" in soul
    assert "state one clear answer" in soul


def test_soul_refuses_to_average_conflicting_sources():
    soul = _flat_soul()
    assert "do not vote, average, splice or guess" in soul
    assert "cannot be confirmed reliably" in soul
    assert "Never dress an unconfirmed value up as fact" in soul


def test_soul_bans_internal_process_and_raw_urls():
    soul = _flat_soul()
    assert 'No "Thoughts" or analysis headings' in soul
    assert "no raw URLs in the body" in soul
    assert "the italic sources entry already exposes the links" in soul


def test_soul_allows_per_site_detail_only_when_the_user_asks_to_compare():
    soul = _flat_soul()
    assert "only when the user explicitly asks to compare sources" in soul
    assert "Do not ask the user to restate a date, place, or constraint" in soul
    assert "never ask them what today's date is" in soul
    assert "say the data is undated instead" in soul


def test_soul_never_lets_the_answer_write_its_own_source_list():
    # A model-written "来源：" line duplicates the italic entry the interface renders
    # and drags domain names into the body — both were observed on a real turn.
    soul = _flat_soul()
    assert "Name a source by its publication or site name, never by its domain or a URL" in soul
    assert "Do not end the answer with any sources line of your own" in soul
    assert 'not "来源：…", not an italic footnote' in soul


def test_soul_does_not_relabel_stale_data_as_todays():
    # A weather turn pasted a 30th-of-the-month page as "today (30th)".
    soul = _flat_soul()
    assert "when the freshest reliable result is not from today, name the date it describes" in soul
    assert "instead of presenting it as today's" in soul


def test_soul_no_longer_pushes_retrieval_by_default():
    soul = _SOUL.read_text(encoding="utf-8")
    assert "search the corpus before guessing" not in soul
    assert "Ground answers in retrieved material" not in soul
    assert "retrieval is an option, not a default step" in soul


def test_delveta_manual_rule_survives():
    soul = _SOUL.read_text(encoding="utf-8")
    assert "Questions about using Delveta itself" in soul
    assert "a built-in product manual is part of the corpus" in soul
