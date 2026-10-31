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


def test_web_search_description_ships_the_time_anchor():
    # The result list starts with retrieval metadata; the description must say so, or
    # the model has no reference for "today" and no warning that dates are missing.
    description = _descriptions()["web_search"]
    assert "retrieval metadata" in description
    assert "reference for today/now" in description
    assert "Most results carry no publication date" in description


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
    assert "Search results are evidence, not the answer" in soul
    assert "Do not walk through the results site by site" in soul
    assert "write one coherent answer that leads with the direct answer" in soul


def test_soul_resolves_conflicts_with_evidence_instead_of_refusing():
    # The generic policy: judge, use the better-supported fact, answer the confirmable
    # parts. Refusing the whole question because one number differs is not allowed.
    soul = _flat_soul()
    assert "When sources conflict, first compare their quality and time validity" in soul
    assert "Answer the parts that can be confirmed" in soul
    assert "Never average, vote, splice or invent a value" in soul
    assert "the question is not automatically unanswerable" in soul


def test_soul_bans_internal_process_and_bare_urls():
    soul = _flat_soul()
    assert 'No "Thoughts" or analysis headings' in soul
    assert "Never print a bare URL or domain" in soul


def test_soul_links_a_source_name_inline_from_a_real_result_url():
    # The body carries the source NAME as a Markdown link; the URL stays hidden in the
    # href and may only be one the results actually returned.
    soul = _flat_soul()
    assert "make the name a Markdown link to that page" in soul
    assert "with the URL copied exactly as the result gives it" in soul
    assert "never build a URL from a domain name" in soul
    assert "write its name as plain text" in soul


def test_soul_allows_per_site_detail_only_when_the_user_asks_to_compare():
    soul = _flat_soul()
    assert "only when the user explicitly asks to compare sources" in soul
    assert 'Use the retrieval time supplied with the search results as your reference for "today"' in soul
    assert "do not ask them to restate a date, place or constraint" in soul


def test_soul_never_lets_the_answer_write_its_own_source_list():
    # A model-written "来源：" line duplicates the italic entry the interface renders
    # and drags domain names into the body — both were observed on a real turn.
    soul = _flat_soul()
    assert 'no sources list or "来源 / Sources" line of your own' in soul
    assert "the interface renders that entry itself" in soul
    assert "Never print a bare URL or domain" in soul


def test_soul_searches_again_on_every_new_message():
    # In a session with earlier failed weather searches the model read "do not retry"
    # as applying across turns, made ZERO tool calls, and repeated its own refusal.
    soul = _flat_soul()
    assert "Every new user message that asks for current information is a NEW search" in soul
    assert "a previous failed or conflicting search is never a reason to skip this turn's search" in soul
    assert "the user asking again is itself the new target" in soul
    assert "This limit is per turn only" in soul


def test_soul_does_not_relabel_stale_data_as_todays():
    # A weather turn pasted a 30th-of-the-month page as "today (30th)".
    soul = _flat_soul()
    assert "never relabel an older page as today's" in soul
    assert "whether a date on the page is its publication date, the event's date, or the date its data describes" in soul
    assert 'Use the retrieval time supplied with the search results as your reference for "today"' in soul


def test_soul_no_longer_pushes_retrieval_by_default():
    soul = _SOUL.read_text(encoding="utf-8")
    assert "search the corpus before guessing" not in soul
    assert "Ground answers in retrieved material" not in soul
    assert "retrieval is an option, not a default step" in soul


def test_delveta_manual_rule_survives():
    soul = _SOUL.read_text(encoding="utf-8")
    assert "Questions about using Delveta itself" in soul
    assert "a built-in product manual is part of the corpus" in soul
