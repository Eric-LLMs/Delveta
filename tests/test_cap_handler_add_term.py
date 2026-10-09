"""cap_handler — AddTermHandler and the add_term WRITE hardening.

Five layers are pinned here:

* the HANDLER DET contract (unit): ``cap-add-term``'s two public slots are ``term``
  and ``domain``, resolved DET-FIRST. ``term`` comes from a literal span (a quoted
  pair; an unquoted ``把/将`` token; an unquoted English token after an insert verb);
  a deictic reference with no antecedent yields NO term. ``domain`` is the named
  vocabulary, copied VERBATIM (translation, slugification and multi-token truncation
  are all defects): its leading determiner (``my`` / ``我的``) is stripped, a trailing
  bare ``词库`` token and a trailing English ``domain`` word are dropped, while a
  carrier with a modifier stays whole (``金融 词汇库`` keeps its space).
* the UNIFIED-ACQUISITION draft contract: a DET-obtained value is ACQUIRED and kept;
  ``acquire()`` returns whatever the rules resolved — both, one, or the EMPTY ``{}``
  when they resolve neither (a DET miss is NOT a bail). Only a BLANK query returns
  ``None``. ``definition`` is NEVER emitted. ``slot_plan()`` authorizes the model for
  exactly the slots DET did NOT fill (rule 2), so a termless sentence is handed to
  the extractor rather than short-circuited to the Agent.
* the TOOL hardening: ``add_term`` explicitly declares ``{WRITE}`` — without it
  the auto-classifier defaults this INSERT tool to READ.
* the FUNNEL wiring: ``handler_for("cap-add-term")`` is the registered handler and
  short-circuits the generic acquisition chain; a COMPLETE draft certifies, a partial
  or empty draft whose required slot no source filled is blocked by the Binder
  (``BIND_MISSING`` — the turn escalates to the Agent to clarify, the tool never
  runs) — with the extractor seam absent (the hermetic lane) a DET miss therefore
  exits ``BIND_MISSING``, NOT ``ACQUISITION_MISSING`` (the handler no longer bails).
* the model-assisted path (see ``test_orchestrator_slot_extraction``): a DET-miss
  draft is completed by the shared extractor and only the authorized slots are
  folded, then the SAME ``_certify -> Binder`` gate runs.

The gold spans are the 44 ``cap-add-term`` cases of the frozen Formal-500
argument benchmark (``logs/_argbench/dataset_formal500.jsonl``) — the OLD
DET-only baseline, reconciled byte-for-byte below (a DET miss is the empty draft;
the frozen file still records those as ``None``/no-args, which is exactly what the
``{} -> model -> still-missing -> BIND_MISSING -> Agent`` chain finally produces).
"""
from __future__ import annotations

import logging
import re
import types
from pathlib import Path

import pytest
from agent.tools.definition import classify_permissions
from agent.tools.tool_permissions import ToolPermission
from api.tools import add_term_tool
from core.application.chat.intent_funnel import matcher as matcher_mod
from core.application.chat.intent_funnel import tool_intent as tool_intent_mod
from core.application.chat.intent_funnel.argument_acquisition import AcquisitionInputs
from core.application.chat.intent_funnel.cap_handler import (
    HANDLERS,
    AddTermHandler,
    handler_for,
)
from core.application.chat.intent_funnel.contract import (
    MATCH_HIT,
    REASON_ACQUISITION_MISSING,
    REASON_BIND_MISSING,
    MatchResult,
)
from core.application.chat.intent_funnel.registry import content_fingerprint
from core.application.chat.intent_funnel.registry.entry import (
    CapabilityEntry,
    QueryRecord,
    RegistryLiveView,
    derive_language,
)
from core.application.chat.understanding import TurnRequirements
from core.config import settings

FUNNEL_LOGGER = "core.application.chat.intent_funnel.funnel"

GOLD_PATH = "logs/_argbench/dataset_formal500.jsonl"

# The frozen arg benchmark lives in the untracked ``logs/`` scratch tree; the
# byte-for-byte reconciliation below is a local-only guard, skipped when absent.
_GOLD_PRESENT = Path(GOLD_PATH).exists()


# ── Layer 0: the Formal-500 gold spans (44 cases) ────────────────────────────────
# (message, expected draft) — expected is ``{"term": …, ["domain": …]}`` or
# ``None`` (gold ``{}`` = fail-closed, the Agent owns the turn). Auto-verified
# against the frozen dataset in ``test_gold_matches_the_frozen_dataset`` below.

GOLD_CASES: list[tuple[str, dict[str, str] | None]] = [
    ('把"keystone"加入我的工程词汇库', {"term": "keystone", "domain": "工程词汇库"}),
    ("帮我把 arch 这个词收进架构生词本", {"term": "arch", "domain": "架构生词本"}),
    ('add "gradient" to my machine learning vocabulary',
     {"term": "gradient", "domain": "machine learning vocabulary"}),
    ('把"C++"加入编程词汇本', {"term": "C++", "domain": "编程词汇本"}),
    ('把"machine-learning.v2"收进模型词库',
     {"term": "machine-learning.v2", "domain": "模型词库"}),
    ('把"throughput"这个词加进来', {"term": "throughput"}),
    ('把 "  backpressure  " 加入性能词汇本',
     {"term": "backpressure", "domain": "性能词汇本"}),
    ('把 "throughput" 这个词加到 工程词汇库',
     {"term": "throughput", "domain": "工程词汇库"}),
    ('add the word "latency" to my performance vocabulary',
     {"term": "latency", "domain": "performance vocabulary"}),
    ("把 熵 加进 信息论 词库", {"term": "熵", "domain": "信息论"}),
    ('add "backpropagation" to the deep learning glossary',
     {"term": "backpropagation", "domain": "deep learning glossary"}),
    ('在 编程词汇本 里加入 "idempotent"',
     {"term": "idempotent", "domain": "编程词汇本"}),
    ('add the term "hallucination" to the LLM terms domain',
     {"term": "hallucination", "domain": "LLM terms"}),
    ('把 "缓存穿透" 添到 后端 词库', {"term": "缓存穿透", "domain": "后端"}),
    ('"vectorization" 加到 numpy vocabulary',
     {"term": "vectorization", "domain": "numpy vocabulary"}),
    ('把 "量化" 这个词收录到 金融 词汇库',
     {"term": "量化", "domain": "金融 词汇库"}),
    ('add "sharding" to the database glossary',
     {"term": "sharding", "domain": "database glossary"}),
    ('把 "正则化" 加到词库', {"term": "正则化"}),
    ("帮我加个词到词库里", {}),
    ("add this word to the glossary", {}),
    ('把 "pipeline" 加到 工程词汇库',
     {"term": "pipeline", "domain": "工程词汇库"}),
    ('在 算法 词库里加入 "贪心"', {"term": "贪心", "domain": "算法"}),
    ('把 "背压" 收进 流处理 词汇本',
     {"term": "背压", "domain": "流处理 词汇本"}),
    ('"residual connection" 加到 深度学习术语库',
     {"term": "residual connection", "domain": "深度学习术语库"}),
    ('把 "幂等性" 收录到 分布式系统 词库',
     {"term": "幂等性", "domain": "分布式系统"}),
    ('在 前端 词汇本里加 "hydration"',
     {"term": "hydration", "domain": "前端 词汇本"}),
    ('把 "套利" 加进 金融 词库', {"term": "套利", "domain": "金融"}),
    ('"spectrogram" 收录到 语音 术语表',
     {"term": "spectrogram", "domain": "语音 术语表"}),
    ('把 "缓存击穿" 添到 后端 词汇库',
     {"term": "缓存击穿", "domain": "后端 词汇库"}),
    ("帮我加个词到 机器学习 词库", {}),
    ('add "idempotency" to the distributed systems glossary',
     {"term": "idempotency", "domain": "distributed systems glossary"}),
    ('put "tokenizer" in my NLP vocabulary',
     {"term": "tokenizer", "domain": "NLP vocabulary"}),
    ('add the term "sharding key" to the database terms',
     {"term": "sharding key", "domain": "database terms"}),
    ('record "bias-variance" in the stats glossary',
     {"term": "bias-variance", "domain": "stats glossary"}),
    ('add "backpressure" to the streaming terms',
     {"term": "backpressure", "domain": "streaming terms"}),
    ('put "memoization" into the algorithms vocabulary',
     {"term": "memoization", "domain": "algorithms vocabulary"}),
    ('add "hedging" to my finance glossary',
     {"term": "hedging", "domain": "finance glossary"}),
    ('add "quantization" to the edge computing terms',
     {"term": "quantization", "domain": "edge computing terms"}),
    ('save "convolution" to the vision terms',
     {"term": "convolution", "domain": "vision terms"}),
    ('add "circuit breaker" to the microservices glossary',
     {"term": "circuit breaker", "domain": "microservices glossary"}),
    ('put "hyperparameter" in the ML vocabulary',
     {"term": "hyperparameter", "domain": "ML vocabulary"}),
    ('add "lexer" to my compiler terms',
     {"term": "lexer", "domain": "compiler terms"}),
    ('add "consistency" to the glossary', {"term": "consistency"}),
    ('add "eventual consistency" to the glossary',
     {"term": "eventual consistency"}),
]


@pytest.mark.parametrize("message,expected", GOLD_CASES)
async def test_formal500_gold_spans(message, expected):
    handler = AddTermHandler()
    assert await handler.acquire(query=message, facts=None) == expected


@pytest.mark.skipif(
    not _GOLD_PRESENT,
    reason="frozen arg benchmark logs/_argbench/dataset_formal500.jsonl not present",
)
def test_gold_matches_the_frozen_dataset():
    """The embedded gold mirrors ``dataset_formal500.jsonl`` byte for byte — a
    literal reconciliation against the benchmark rather than a paraphrase."""
    import json

    rows = [
        json.loads(line)
        for line in Path(GOLD_PATH).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    actual = {
        r.get("query"): (r.get("gold_arguments") or None)
        for r in rows
        if r.get("capability_id") == "cap-add-term"
    }
    # the DET layer's empty {} draft is the same no-args final outcome the frozen
    # file records as null/None (the model then fails to resolve it too -> Agent),
    # so normalize {} -> None for the byte-for-byte reconciliation.
    embedded = {message: (expected or None) for message, expected in GOLD_CASES}
    assert actual == embedded
    assert len(actual) == 44


# ── Layer 1a: term — a quoted span is copied verbatim, quotes dropped ─────────────


@pytest.mark.parametrize("message,expected", [
    ('把"alpha"加入编程词汇本', {"term": "alpha", "domain": "编程词汇本"}),
    ("把 'beta' 加入编程词汇本", {"term": "beta", "domain": "编程词汇本"}),
    ("把“项目”加入 架构 生词本", {"term": "项目", "domain": "架构 生词本"}),
    ("把「delta」加入 架构 生词本", {"term": "delta", "domain": "架构 生词本"}),
    # inner edge whitespace trimmed
    ('把 "  spaced  " 加入 编程词汇本', {"term": "spaced", "domain": "编程词汇本"}),
    # operator / punctuation chars kept
    ('把 "C++" 加入编程词汇本', {"term": "C++", "domain": "编程词汇本"}),
    ('把 "machine-learning.v2" 加入模型词库',
     {"term": "machine-learning.v2", "domain": "模型词库"}),
    ('把 "residual connection" 加入深度学习术语库',
     {"term": "residual connection", "domain": "深度学习术语库"}),
])
async def test_quoted_term_is_extracted_verbatim(message, expected):
    handler = AddTermHandler()
    assert await handler.acquire(query=message, facts=None) == expected


# ── Layer 1b: term — unquoted spans (Chinese 把/将, English insert verb) ───────────


@pytest.mark.parametrize("message,expected", [
    ("帮我把 arch 这个词收进架构生词本", "arch"),                  # ZH 把 … 这个词
    ("把 熵 加进 信息论 词库", "熵"),                              # ZH 把 … 加进
    ("帮我把 entropy 这个词加入 编程词汇本", "entropy"),
    ("add gradient to my machine learning vocabulary", "gradient"),       # EN unquoted
    ("put tokenizer into the NLP vocabulary", "tokenizer"),
])
async def test_unquoted_term(message, expected):
    handler = AddTermHandler()
    draft = await handler.acquire(query=message, facts=None)
    assert draft is not None and draft["term"] == expected


# ── Layer 1c: term — deictic / antecedent-less references yield the EMPTY draft ────
# a DET miss is NOT a bail: the empty ``{}`` draft still reaches the orchestrator,
# whose slot_plan hands term/domain to the model (rule 2). Only a blank query
# returns ``None``.


@pytest.mark.parametrize("message", [
    "add this word to the glossary",       # EN deictic
    "帮我加个词到词库里",                    # ZH: no named term
    "把这个词加进去",                        # ZH deictic
    "帮我加个词到 机器学习 词库",             # a named DOMAIN does not rescue a missing term
])
async def test_deictic_or_termless_yields_the_empty_draft(message):
    handler = AddTermHandler()
    assert await handler.acquire(query=message, facts=None) == {}


@pytest.mark.parametrize("blank", ["", "   ", "\n\t", None])
async def test_blank_query_fails_closed(blank):
    handler = AddTermHandler()
    assert await handler.acquire(query=blank, facts=None) is None


# ── Layer 2a: domain — verbatim, translation / slug / truncation are defects ──────


@pytest.mark.parametrize("message,expected", [
    ('把"keystone"加入我的工程词汇库', "工程词汇库"),        # determiner stripped, not slugified
    ("add \"gradient\" to my machine learning vocabulary", "machine learning vocabulary"),
    ('把"machine-learning.v2"收进模型词库', "模型词库"),       # attached 词库 kept whole
    ('"vectorization" 加到 numpy vocabulary', "numpy vocabulary"),
    ('add "backpropagation" to the deep learning glossary',
     "deep learning glossary"),
])
async def test_domain_copied_verbatim(message, expected):
    handler = AddTermHandler()
    draft = await handler.acquire(query=message, facts=None)
    assert draft is not None and draft["domain"] == expected


# ── Layer 2b: domain — a modifier keeps the carrier whole (spaces survive) ────────


@pytest.mark.parametrize("message,expected", [
    ('把 "量化" 这个词收录到 金融 词汇库', "金融 词汇库"),      # space preserved
    ('把 "背压" 收进 流处理 词汇本', "流处理 词汇本"),
    ('把 "缓存击穿" 添到 后端 词汇库', "后端 词汇库"),
    ('"spectrogram" 收录到 语音 术语表', "语音 术语表"),
    ('add "backpressure" to the streaming terms', "streaming terms"),
    ('add "quantization" to the edge computing terms', "edge computing terms"),
])
async def test_modifier_keeps_the_carrier(message, expected):
    handler = AddTermHandler()
    draft = await handler.acquire(query=message, facts=None)
    assert draft is not None and draft["domain"] == expected


# ── Layer 2c: domain — a spaced bare 词库 carries no name (the modifier remains) ──


@pytest.mark.parametrize("message,expected", [
    ("把 熵 加进 信息论 词库", "信息论"),        # space-separated carrier dropped
    ('把 "缓存穿透" 添到 后端 词库', "后端"),
    ('把 "套利" 加进 金融 词库', "金融"),
    ('在 算法 词库里加入 "贪心"', "算法"),
    ('把 "幂等性" 收录到 分布式系统 词库', "分布式系统"),
])
async def test_spaced_bare_word_library_is_dropped(message, expected):
    handler = AddTermHandler()
    draft = await handler.acquire(query=message, facts=None)
    assert draft is not None and draft["domain"] == expected


# ── Layer 2d: domain — a bare generic carrier / indicator-less region is absent ───


@pytest.mark.parametrize("message", [
    '把 "正则化" 加到词库',        # bare 词库
    'add "consistency" to the glossary',  # bare glossary
    'add "eventual consistency" to the glossary',
    '把 "throughput"这个词加进来',  # no vocabulary indicator at all
])
async def test_bare_generic_or_indicatorless_domain_is_absent(message):
    handler = AddTermHandler()
    draft = await handler.acquire(query=message, facts=None)
    assert draft is not None
    assert "domain" not in draft          # term survives; the domain is honestly absent


# ── Layer 3: the draft contract — partial draft, never definition ─────────────────


async def test_partial_draft_has_term_only():
    handler = AddTermHandler()
    assert await handler.acquire(query='把 "正则化" 加到词库', facts=None) == {"term": "正则化"}


async def test_only_the_two_slots_are_ever_emitted():
    handler = AddTermHandler()
    draft = await handler.acquire(
        query='把"keystone"加入我的工程词汇库', facts=types.SimpleNamespace())
    assert set(draft) == {"term", "domain"}   # never definition, never any other slot


def _facts_viewer(asset_id):
    return types.SimpleNamespace(
        has_viewer=True, viewer_asset_id=str(asset_id), viewer_current_page=1,
        viewer_page_from=None, viewer_page_to=None, has_viewer_selection=False,
        has_attachment=True, attachment_asset_id=str(asset_id),
        path_asset_id=str(asset_id), has_turn_context=True,
    )


async def test_asset_context_never_supplies_a_term_or_domain():
    # a termless sentence with an asset present must still carry NO term/domain:
    # facts are ignored on purpose, so no viewer/attachment id can leak in. The DET
    # layer yields the empty draft (the model, not the asset, owns the gap).
    handler = AddTermHandler()
    assert await handler.acquire(query="帮我加个词到词库里", facts=_facts_viewer("abc")) == {}


# ── Layer 3b: slot_plan — the model is authorized ONLY for the DET-unfilled slots ──


async def test_slot_plan_authorizes_both_when_det_resolves_neither():
    handler = AddTermHandler()
    draft = await handler.acquire(query="帮我加个词到词库里", facts=None)   # {}
    plan = handler.slot_plan(query="帮我加个词到词库里", facts=None, draft=draft)
    assert set(plan.model_slots) == {"term", "domain"}
    assert plan.default_slots == ()


async def test_slot_plan_authorizes_domain_when_det_got_only_the_term():
    handler = AddTermHandler()
    msg = '把 "正则化" 加到词库'                       # term DET-ok, domain bare -> None
    draft = await handler.acquire(query=msg, facts=None)
    assert draft == {"term": "正则化"}
    plan = handler.slot_plan(query=msg, facts=None, draft=draft)
    assert plan.model_slots == ("domain",)               # the term is ACQUIRED, not re-asked


async def test_slot_plan_authorizes_nothing_when_det_resolved_both():
    handler = AddTermHandler()
    msg = '把 "keystone" 加入我的工程词汇库'
    draft = await handler.acquire(query=msg, facts=None)
    assert set(draft) == {"term", "domain"}
    plan = handler.slot_plan(query=msg, facts=None, draft=draft)
    assert plan.model_slots == ()                        # DET-sufficient -> no model call


# ── Layer 4: the roster wiring ───────────────────────────────────────────────────


def test_roster_wires_cap_add_term():
    handler = handler_for("cap-add-term")
    assert isinstance(handler, AddTermHandler)
    assert handler.capability_id == "cap-add-term"
    assert HANDLERS["cap-add-term"] is handler


def test_cap_add_term_is_no_longer_on_the_generic_chain():
    assert handler_for("cap-add-term") is not None
    for cid in ("cap-open-pdf", ""):
        assert handler_for(cid) is None


# ── Layer 5: the funnel wiring (real cascade, faked Matcher HIT) ─────────────────


def _ctx(message: str):
    return types.SimpleNamespace(
        body=types.SimpleNamespace(message=message, attach=None, viewer=None),
        owned_asset_id=None, path_asset_id=None, research_turn=False,
        effective_handoff=None, session_id="s1", history=[],
    )


def _add_term_entry() -> CapabilityEntry:
    q = "把生词加入词库"
    return CapabilityEntry(
        capability_id="cap-add-term", tool_binding="add_term",
        description="Add a word to a vocabulary domain named by the user.",
        standard_queries=(QueryRecord(id="cap-add-term-q1", query=q,
                                      language=derive_language(q)),),
        parameters={
            "term": {"type": "string", "required": True, "max_len": 120,
                     "description": "the word to add, quotes removed"},
            "domain": {"type": "string", "required": True, "max_len": 60,
                       "description": "the vocabulary domain to add it into"},
        },
    )


class _Rec:
    def __init__(self) -> None:
        self.legacy: list = []


def _wire(monkeypatch, entry: CapabilityEntry):
    view = RegistryLiveView(fingerprint=content_fingerprint([entry]), entries=(entry,))
    rec = _Rec()

    async def fake_active(**kw):
        return view

    def fake_match(message, facts, v):
        return MatchResult(state=MATCH_HIT, capability_id=entry.capability_id,
                           matched_literal="把生词加入词库")

    async def fake_select_and_extract(query, candidates, *, entries_by_id,
                                      llm=None, facts=None):
        rec.legacy.append(query)
        raise AssertionError("legacy select_and_extract must never run on the new lane")

    monkeypatch.setattr(
        "core.application.chat.intent_funnel.registry.active_view", fake_active)
    monkeypatch.setattr(matcher_mod, "match", fake_match)
    monkeypatch.setattr(tool_intent_mod, "select_and_extract", fake_select_and_extract)
    monkeypatch.setattr(settings, "chat_cap_router_backend", "stub")
    return rec, view


def _deps(provider):
    return types.SimpleNamespace(
        session_factory=None, embedder=lambda: object(), llm=object(),
        acquisition_inputs=provider)


async def _route(ctx, deps, req):
    from core.application.chat.intent_funnel import funnel
    return await funnel.route(ctx, deps=deps, requirements=req)


async def test_complete_turn_certifies_handler_payload(monkeypatch):
    entry = _add_term_entry()
    rec, view = _wire(monkeypatch, entry)
    req = TurnRequirements()
    out = await _route(_ctx('把"keystone"加入我的工程词汇库'), _deps(None), req)

    assert out is not req                              # a CERTIFIED turn
    act = out.requested_action
    assert act["tool"] == "add_term"
    assert act["capability_id"] == "cap-add-term"
    assert act["args"] == {"term": "keystone", "domain": "工程词汇库"}  # verbatim
    assert act["funnel_stage"] == "tool_intent"        # the EXISTING handoff shape
    assert act["funnel_registry_version"] == view.fingerprint
    assert not rec.legacy                              # no legacy hop


async def test_partial_term_only_draft_is_blocked_by_the_binder(monkeypatch, caplog):
    """A partial ``{"term"}`` (domain absent) must NEVER certify or run the tool:
    the Binder's required-slot gate yields ``BIND_MISSING`` and the turn escalates
    to the Agent to clarify."""
    caplog.set_level(logging.INFO, logger=FUNNEL_LOGGER)
    entry = _add_term_entry()
    rec, _view = _wire(monkeypatch, entry)
    req = TurnRequirements()
    out = await _route(_ctx('把 "正则化" 加到词库'), _deps(None), req)

    assert out is req                                  # NOT certified — exit to Agent
    assert out.requested_action is None                # the tool is never handed off
    lines = [r.getMessage() for r in caplog.records
             if r.name == FUNNEL_LOGGER and "funnel_trace" in r.getMessage()]
    assert len(lines) == 1, lines
    assert re.search(r"fallback_reason=(\S+)", lines[0]).group(1) == REASON_BIND_MISSING
    assert not rec.legacy                              # no Qwen, no legacy extraction


async def test_termless_turn_without_extractor_exits_bind_missing(monkeypatch, caplog):
    """The DET miss is no longer an ``ACQUISITION_MISSING`` bail: the handler hands
    the empty ``{}`` draft + a plan authorizing term/domain to the model. With the
    extractor seam ABSENT (hermetic lane) nothing fills the required slots, so the
    SAME Binder gate exits ``BIND_MISSING`` — the Agent still owns the turn, but for
    the right reason (no legal value, not a pre-model short-circuit)."""
    caplog.set_level(logging.INFO, logger=FUNNEL_LOGGER)
    entry = _add_term_entry()
    rec, _view = _wire(monkeypatch, entry)
    req = TurnRequirements()
    out = await _route(_ctx("帮我加个词到词库里"), _deps(None), req)

    assert out is req
    lines = [r.getMessage() for r in caplog.records
             if r.name == FUNNEL_LOGGER and "funnel_trace" in r.getMessage()]
    assert len(lines) == 1, lines
    assert re.search(r"fallback_reason=(\S+)", lines[0]).group(1) == REASON_BIND_MISSING
    assert not rec.legacy                              # no legacy extraction


async def test_handler_bypasses_a_forged_generic_chain(monkeypatch):
    """A forged MODEL value never reaches the draft: the handler owns both slots."""
    entry = _add_term_entry()
    rec, _view = _wire(monkeypatch, entry)
    poisoned = {"cap-add-term": AcquisitionInputs(
        model_values={"term": "HALLUCINATED", "domain": "HALLUCINATED"})}
    req = TurnRequirements()
    out = await _route(_ctx('把 "keystone" 加入我的工程词汇库'),
                       _deps(lambda cid: poisoned.get(cid)), req)

    assert out is not req
    assert out.requested_action["args"] == {"term": "keystone", "domain": "工程词汇库"}
    assert not rec.legacy


async def test_handler_returning_none_is_acquisition_missing(monkeypatch):
    """The orchestrator branch: a handler with no legal draft (``None``) exits
    the turn to the Agent with ``ACQUISITION_MISSING`` — fail-closed."""
    from core.application.chat.intent_funnel import orchestrator as orch
    from core.application.chat.intent_funnel.cap_handler import roster

    class _NoneHandler:
        capability_id = "cap-add-term"

        async def acquire(self, *, query, facts):
            return None

    monkeypatch.setitem(roster.HANDLERS, "cap-add-term", _NoneHandler())
    entry = _add_term_entry()
    view = RegistryLiveView(fingerprint=content_fingerprint([entry]), entries=(entry,))
    trace: dict = {}
    out = await orch._acquisition_hop(
        TurnRequirements(), types.SimpleNamespace(), entry,
        query='把 "x" 加入工程词汇库', facts=None, view=view, trace=trace, capture=None)

    assert out is None
    assert trace["fallback"] == REASON_ACQUISITION_MISSING


# ── Layer 6: add_term declares WRITE explicitly ──────────────────────────────────


class _CapRuntime:
    def __init__(self) -> None:
        self.tools: dict = {}

    def register(self, defn) -> None:
        self.tools[defn.name] = defn


def test_add_term_declares_write_explicitly():
    runtime = _CapRuntime()
    ctx = types.SimpleNamespace(resolve=lambda key: None)
    add_term_tool.register(runtime, ctx, None)
    defn = runtime.tools["add_term"]

    # The declaration is EXPLICIT (not None): without it the auto-classifier
    # (term/domain/definition carry no write hint) would default this INSERT to READ.
    assert defn.permission is not None
    assert ToolPermission.WRITE in defn.permission
    assert classify_permissions(defn) == frozenset({ToolPermission.WRITE})
