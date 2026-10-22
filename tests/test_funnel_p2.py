"""P2 — node-level tests + the two-lane cascade through funnel.route.

Chain ruling: Matcher HIT -> Argument Acquisition -> Binder (no re-selection);
MISS/AMBIGUOUS -> Recall -> cap_router -> Argument Acquisition -> Binder; every
failure exits to the Agent. No recheck hop, no Decision node in the active path.

Discipline: EVERY node section below runs against fake contracts only — no node
needs another node to be alive to be tested. The selector and the argument
extractor are injected through ``ChatDeps``, so no test issues a real model call.

The cascade section pins the 8.10 exit classification: on every abstain/fault
the ORIGINAL requirements object is returned BY IDENTITY (the Agent's turn stays
byte-identical), and only a certified turn produces a new object.
"""
from __future__ import annotations

import asyncio
import logging
import types

import pytest
from core.application.chat.intent_funnel import cap_router, funnel, guardrails, matcher
from core.application.chat.intent_funnel.argument_acquisition.extractor import (
    ExtractionUnavailable,
)
from core.application.chat.intent_funnel.contract import (
    BIND_COMPLETE,
    BIND_INVALID,
    BIND_MISSING,
    MATCH_HIT,
    REASON_ACQUISITION_UNDECLARED,
    REASON_BIND_MISSING,
    REASON_NO_CANDIDATE,
    REASON_RECALL_TIMEOUT,
    REASON_RECALL_UNAVAILABLE,
    REASON_REGISTRY_UNAVAILABLE,
    REASON_VERSION_MISMATCH,
    Candidate,
    MatchResult,
    TurnFacts,
)
from core.application.chat.intent_funnel.registry import content_fingerprint
from core.application.chat.intent_funnel.registry import entry as T
from core.application.chat.understanding import (
    Complexity,
    Confidence,
    Signal,
    TurnRequirements,
)

# ════════════════════════ shared fakes (contract-shaped, minimal) ═══════════════


def _q(i, text, *, position=0, standard_query_id=None, kind="standard"):
    return T.QueryRecord(id=f"q{i}", query=text,
                         language=T.derive_language(text), position=position,
                         standard_query_id=standard_query_id)


def _entry(cid, *, tool="create_folder", corpus=(), patterns=(), aliases=(),
           arg_slots=None, enabled=True, status="active", examples=("做个事",),
           negatives=(), parameters=None):
    # ``corpus`` feeds the EXACT set: the live-table ruling hydrates it as
    # Standard (first row) + Similar query rows (ruling/26);
    # patterns/aliases stay storable but are inert as far as the Matcher goes.
    corpus = tuple(corpus)
    sims = tuple(_q(10 + n, s, position=n, standard_query_id="q1" if corpus else None)
                 for n, s in enumerate(corpus[1:]))
    return T.CapabilityEntry(
        capability_id=cid, tool_binding=tool, description=f"does {cid}",
        standard_queries=(_q(1, corpus[0]),) if corpus else (),
        similar_queries=sims,
        patterns=tuple(patterns), aliases=tuple(aliases),
        request_query_examples=tuple(examples),
        negatives=tuple(negatives), arg_slots=arg_slots if arg_slots is not None
        else {"name": {"source": "user_input"}},
        parameters=dict(parameters) if parameters is not None else {},
        enabled=enabled, status=status,
    )


# canonical-shaped schemas for the seeded tools (argument-acquisition targets)
_NAME_SCHEMA = {"name": {"type": "string", "description": "folder name",
                         "required": True, "max_len": 120}}
_TERM_SCHEMA = {
    "term": {"type": "string", "description": "the term", "required": True, "max_len": 120},
    "domain": {"type": "string", "description": "vocabulary domain",
               "required": True, "max_len": 60},
}


def _view(entries, version=1):
    # live-table ruling: the read model IS the live corpus, keyed
    # only by its content fingerprint (the ``version`` arg is call-site noise).
    return T.RegistryLiveView(
        fingerprint=content_fingerprint(list(entries)),
        entries=tuple(entries),
    )


class _Embedder:
    def __init__(self, vec, *, fail=False):
        self.vec, self.fail, self.calls = list(vec), fail, 0

    async def embed(self, texts):
        self.calls += 1
        if self.fail:
            raise RuntimeError("embedder down")
        return [list(self.vec) for _ in texts]


class _LLM:
    """Replies are popped in order; an Exception member raises (transport fault).
    ``calls`` records the per-call channel kwargs (model/base_url/api_key/timeout/
    temperature) so the online model's explicit forwarding is assertable."""

    def __init__(self, replies=()):
        self.replies = list(replies)
        self.prompts = []
        self.calls: list[dict] = []

    async def complete_json(self, prompt, *, system_prompt=None, **kw):
        self.prompts.append(prompt)
        self.calls.append(kw)
        if not self.replies:
            return {}
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _index(caps, *, version="corpus1-test"):
    """caps: [(capability_id, [sentences], [sentence vectors])]. Mirrors the
    LIVE corpus rows the real recall's in-process lane consumes (no
    session_factory on the fake -> no pgvector ANN in unit tests)."""
    corpus = []
    for cid, examples, vecs in caps:
        for i, (q, v) in enumerate(zip(examples, vecs)):
            corpus.append(types.SimpleNamespace(
                kind="standard", query_id=f"{cid}-{i}", capability_id=cid,
                query=q, language=T.derive_language(q),
                standard_query_id=None, vector=list(v),
            ))
    return types.SimpleNamespace(version=version, corpus=tuple(corpus))


def _ctx(msg, **kw):
    base = {
        "body": types.SimpleNamespace(message=msg, attach=None, viewer=None),
        "owned_asset_id": None, "research_turn": False, "effective_handoff": None,
        "session_id": "s-1",
    }
    base.update(kw)
    return types.SimpleNamespace(**base)


def _req(**kw):
    base = {"complexity": Complexity.LOW, "confidence": Confidence.LOW,
                "needs_web": Signal.LOW, "needs_memory": False}
    base.update(kw)
    return TurnRequirements(**base)


# ════════════════════════ TurnFacts ( contract ruling) ════════════════


def test_turn_facts_reads_structured_viewer_and_attach_only():
    viewer = types.SimpleNamespace(asset_id="a-7", page=12,
                                   selections=[types.SimpleNamespace(text="x")])
    ctx = _ctx("总结这一页", body=types.SimpleNamespace(
        message="总结这一页", attach={"asset_id": "b-1"}, viewer=viewer))
    f = TurnFacts.of(ctx)
    assert f == TurnFacts(
        has_viewer=True, viewer_asset_id="a-7", viewer_current_page=12,
        viewer_page_from=12, viewer_page_to=12,  # lone page == degenerate range
        has_viewer_selection=True, has_attachment=True,
        attachment_asset_id="b-1",  #: the attach's id is a settled fact
        has_turn_context=True,
    )


def test_turn_facts_asset_id_sources_and_precedence():
    # owned upload wins over the attach dict (legacy _asset_id precedence kept)
    ctx = _ctx("x", body=types.SimpleNamespace(
        message="x", attach={"asset_id": "b-1"}, viewer=None),
        owned_asset_id="o-9")
    f = TurnFacts.of(ctx)
    assert f.attachment_asset_id == "o-9"
    # object-shaped attach (not dict) still yields its id
    ctx = _ctx("x", body=types.SimpleNamespace(
        message="x", attach=types.SimpleNamespace(asset_id="o-2"), viewer=None))
    assert TurnFacts.of(ctx).attachment_asset_id == "o-2"
    # path_asset_id rides through from the upstream-resolved ctx field — no I/O here
    ctx = _ctx("x", body=types.SimpleNamespace(
        message="x", attach=None, viewer=None), path_asset_id="p-3")
    f = TurnFacts.of(ctx)
    assert f.path_asset_id == "p-3" and f.has_attachment is False \
        and f.attachment_asset_id == ""
    # plain turn: both asset facts empty, nothing guessed
    plain = TurnFacts.of(_ctx("hello", session_id=None))
    assert plain.attachment_asset_id == "" and plain.path_asset_id == ""


def test_turn_facts_plain_turn_is_all_empty():
    f = TurnFacts.of(_ctx("hello", session_id=None))
    assert f.has_viewer is False and f.has_attachment is False
    assert f.viewer_current_page is None and f.has_turn_context is False
    assert f.viewer_page_from is None and f.viewer_page_to is None


def test_matcher_contract_takes_facts_and_never_history():
    v = _view([_entry("cap-a", corpus=("新建文件夹",))])
    facts = TurnFacts(has_viewer=True, viewer_current_page=3)
    # facts is a REQUIRED contract slot; verdicts stay table-only (8.1)
    assert matcher.match("新建文件夹", facts, v).state == MATCH_HIT
    assert matcher.match("新建文件夹", TurnFacts(), v).state == MATCH_HIT


# ═══════════════════════════════ guardrails ═════════════════════════════════════


def test_entry_veto_is_gone():
    # L0 retirement: guardrails carries the negation predicate only. No rule may
    # re-route a turn before the Matcher sees it.
    assert not hasattr(guardrails, "turn_veto")


def test_negation_guard_is_a_pure_predicate():
    assert guardrails.negated("不要新建文件夹")
    assert not guardrails.negated("新建文件夹")


# ═══════════════════════════════ recall (quality gate only) ════════════════════


async def test_recall_keeps_every_hit_with_provenance_and_never_adjudicates():
    from core.application.chat.intent_funnel import recall as recall_node

    idx = _index([
        ("cap-a", ["做a1", "做a2"], [[1.0, 0.0], [1.0, 0.1]]),  # BOTH kept
        ("cap-b", ["加个词"], [[1.0, 0.5]]),
        ("cap-c", ["闲聊"], [[1.0, 2.0]]),                      # below min_score
    ])
    emb = _Embedder([1.0, 0.0])
    res = await recall_node.recall(idx, "新建文件夹", embedder=emb, min_score=0.82)
    ids = [c.capability_id for c in res.candidates]
    scores = [c.score for c in res.candidates]
    assert scores == sorted(scores, reverse=True)  # ranked, NOT margin-aborted
    assert ids.count("cap-a") == 2                 # no per-capability MAX merge:
    assert "cap-c" not in ids                      # every hit >= gate is its own candidate
    top = res.candidates[0]
    assert top.matched_example == "做a1" and top.origin == "recall"
    assert top.query_kind == "standard" and top.language == "zh" and top.query_id
    assert emb.calls == 1                          # ONE embedding for BOTH paths


async def test_recall_blank_query_and_embedder_fault():
    from core.application.chat.intent_funnel import recall as recall_node

    idx = _index([("cap-a", ["做a"], [[1.0, 0.0]])])
    assert (await recall_node.recall(idx, "  ", embedder=_Embedder([1.0]),
                                     min_score=0.5)).candidates == ()
    with pytest.raises(RuntimeError):
        await recall_node.recall(idx, "做a", embedder=_Embedder([1.0], fail=True),
                                 min_score=0.5)
    with pytest.raises(RuntimeError):
        await recall_node.recall(idx, "做a", embedder=_Embedder([]),
                                 min_score=0.5)


async def test_recall_gate_is_inclusive_near_ties_survive():
    from core.application.chat.intent_funnel import recall as recall_node

    # recall proposes, the cap_router disposes (design §3): the gate is a
    # >= filter, nothing more — no margin cut, no winner pick.
    idx = _index([("cap-a", ["a"], [[1.0, 0.0]]),
                  ("cap-b", ["b"], [[1.0, 0.02]])])
    res = await recall_node.recall(idx, "q", embedder=_Embedder([1.0, 0.0]),
                                   min_score=1.0)
    assert [c.capability_id for c in res.candidates] == ["cap-a"]  # exact 1.0 kept


def test_load_rows_sql_union_all_spacing():
    """Anti-regression pin ( shadow-live sim): the in-process
    degraded lane's loader is assembled from adjacent string literals — a
    missing space after "UNION ALL" rendered as ``ALLSELECT`` and blew up
    against real Postgres (every MISS lane => RECALL_UNAVAILABLE). Every
    funnel test monkeypatches ``load_index``, so only a string-level pin
    catches it. The same check covers the ANN statements' joins."""
    from core.application.chat.intent_funnel.recall import index as recall_index
    from core.application.chat.intent_funnel.recall import retriever as recall_retriever

    sql = " ".join(str(recall_index._LOAD_ROWS_SQL).split())
    assert " UNION ALL SELECT " in sql
    for stmt in (recall_retriever._STANDARD_SQL, recall_retriever._SIMILAR_SQL):
        s = " ".join(str(stmt).split())
        assert " ORDER BY " in s and " LIMIT :pool" in s


# ═══════════════════════════════ binder (four states, 8.7) ═════════════════════


def test_binder_complete_missing_invalid_and_c2():
    from core.application.chat import actions
    from core.application.chat.intent_funnel import binder

    entry = _entry("cap-a", tool="create_folder",
                   arg_slots={"name": {"source": "user_input"}})
    ok = binder.bind(entry, '新建文件夹"季度报告"', _ctx(""))
    assert ok.state == BIND_COMPLETE and ok.args == {"name": "季度报告"}

    miss = binder.bind(entry, "新建文件夹", _ctx(""))          # quoted name missing
    assert miss.state == BIND_MISSING

    wrong = _entry("cap-a", arg_slots={"folder": {"source": "user_input"}})
    bad = binder.bind(wrong, '新建文件夹"季度报告"', _ctx(""))  # whitelist mismatch
    assert bad.state == BIND_INVALID

    long_name = _entry("cap-a", arg_slots={})                   # defaults to schema
    bad = binder.bind(long_name, '新建文件夹"' + "名" * 121 + '"', _ctx(""))
    assert bad.state == BIND_INVALID                            # schema bound

    ghost = _entry("cap-g", tool="ghost_tool")
    with pytest.raises(actions.ActionIntegrityFailure):
        binder.bind(ghost, "新建文件夹", _ctx(""))              # C2 propagates


def test_binder_negation_is_missing_not_answer():
    from core.application.chat.intent_funnel import binder

    entry = _entry("cap-a")
    assert binder.bind(entry, '不要新建文件夹"x"', _ctx("")).state == BIND_MISSING


# ── validate-only gate over the acquisition draft (chain ruling) ──────────────


def test_binder_validate_normalizes_against_registry_schema():
    from core.application.chat.intent_funnel import binder

    entry = _entry("cap-a", parameters=_NAME_SCHEMA)
    ok = binder.validate(entry, {"name": "  季度报告  "})
    assert ok.state == BIND_COMPLETE and ok.args == {"name": "季度报告"}   # stripped
    assert binder.validate(entry, {"name": "x" * 121}).state == BIND_INVALID  # max_len
    assert binder.validate(entry, None).state == BIND_MISSING             # no draft
    assert binder.validate(entry, {"name": "  "}).state == BIND_MISSING   # blank required
    assert binder.validate(entry, {"bogus": "x"}).state == BIND_INVALID   # whitelist
    assert binder.validate(entry, {"name": 7}).args == {"name": "7"}      # coerced
    # non-string schema values are normalized, over-length still refused
    assert binder.validate(entry, {"name": "名" * 121}).state == BIND_INVALID
    # a schema-less capability needs no arguments: even None draft completes
    empty = _entry("cap-a", parameters={})
    done = binder.validate(empty, None)
    assert done.state == BIND_COMPLETE and done.args == {}
    # optional slot may be omitted; required ones may not
    opt = _entry("cap-a", parameters={
        "name": {"type": "string", "description": "n", "required": True},
        "note": {"type": "string", "description": "n", "required": False},
    })
    assert binder.validate(opt, {"name": "x"}).args == {"name": "x"}
    assert binder.validate(opt, {}).state == BIND_MISSING


# ── context-sourced slots: the FACTS, never the draft, answer asset identity ──
# (E2E-matrix ruling — asset_id "from turn facts, not the sentence")

_ASSET_SCHEMA = {"asset_id": {"type": "string", "max_len": 64, "required": True}}


def test_binder_asset_slot_overwrites_hallucinated_draft_with_the_fact():
    from core.application.chat.intent_funnel import binder

    entry = _entry("cap-a", parameters=_ASSET_SCHEMA)
    facts = TurnFacts(has_attachment=True, attachment_asset_id="real-uuid")
    bound = binder.validate(entry, {"asset_id": "hallucinated-uuid"}, facts)
    assert bound.state == BIND_COMPLETE and bound.args == {"asset_id": "real-uuid"}
    # model omitted it — the fact fills it, the draft is not needed
    assert binder.validate(entry, None, facts).args == {"asset_id": "real-uuid"}


def test_binder_asset_slot_stripped_when_facts_absent_never_guess():
    from core.application.chat.intent_funnel import binder

    entry = _entry("cap-a", parameters=_ASSET_SCHEMA)
    # A model-supplied asset id without any turn fact is REFUSED (missing →
    # Agent owns the clarification): facts are the only possible truth source.
    assert binder.validate(entry, {"asset_id": "made-up"}, TurnFacts()).state == BIND_MISSING
    assert binder.validate(entry, {"asset_id": "made-up"}, None).state == BIND_MISSING


def test_binder_asset_slot_fact_precedence_attach_path_viewer():
    from core.application.chat.intent_funnel import binder

    entry = _entry("cap-a", parameters=_ASSET_SCHEMA)
    f = TurnFacts(attachment_asset_id="A", path_asset_id="P", viewer_asset_id="V")
    assert binder.validate(entry, {}, f).args == {"asset_id": "A"}
    assert binder.validate(entry, {}, TurnFacts(path_asset_id="P",
                                                viewer_asset_id="V")).args == {"asset_id": "P"}
    assert binder.validate(entry, {}, TurnFacts(viewer_asset_id="V")).args == {"asset_id": "V"}


def test_binder_context_slots_touched_only_by_name_not_by_tool():
    from core.application.chat.intent_funnel import binder

    # a schema WITHOUT an asset_id slot is byte-identical behavior under facts
    entry = _entry("cap-a", parameters=_NAME_SCHEMA)
    facts = TurnFacts(attachment_asset_id="A")
    assert binder.validate(entry, {"name": "报告"}, facts).args == {"name": "报告"}
    # unknown extra draft keys still hit the whitelist gate, not the injector
    assert binder.validate(entry, {"name": "x", "asset_id": "A"}, facts).state == BIND_INVALID


# ═══════════════════════════════ cascade via route() ═══════════════════════════

def _open(monkeypatch, *, timeout=5.0):
    from core.config import settings

    monkeypatch.setattr(settings, "chat_tool_intent_timeout_seconds", timeout + 1)
    monkeypatch.setattr(settings, "chat_funnel_timeout_seconds", timeout)


class _Selector:
    """Test double for the ONE production selector (no model call).

    ``route`` overrides the answer; otherwise it selects ``cid`` when that
    capability is on the table.
    """

    def __init__(self, cid="cap-a", route=None):
        self.cid, self.route = cid, route
        self.calls, self.seen = 0, None

    async def select(self, query, candidates, *, entries_by_id, facts=None):
        self.calls += 1
        self.seen = list(candidates)
        if self.route is not None:
            return self.route
        if self.cid in [c.capability_id for c in candidates]:
            return cap_router.CapabilityRoute(cap_router.ROUTE_SELECTED, self.cid)
        return cap_router.CapabilityRoute(cap_router.ROUTE_NONE)


class _Extractor:
    """Test double for the MODEL-owned argument extractor (Qwen in production).

    Records every call so call conditions / inputs are assertable; ``values``
    may be a dict or a callable(entry, model_slots, bundle); ``fail`` raises
    ``ExtractionUnavailable`` (the fail-closed transport path).
    """

    def __init__(self, values=None, *, fail=False):
        self.values, self.fail = values, fail
        self.calls = []

    async def __call__(self, *, query, entry, model_slots, bundle, prompt=None):
        self.calls.append({"query": query, "entry": entry,
                           "model_slots": tuple(model_slots),
                           "bundle": bundle, "prompt": prompt})
        if self.fail:
            raise ExtractionUnavailable("extractor down")
        values = (self.values(entry, model_slots, bundle)
                  if callable(self.values) else dict(self.values or {}))
        return values, bundle.source


def _wire(monkeypatch, *, view, index, embedder, selector=None, extractor=None):
    calls = {"registry": 0}

    async def fake_active(**kw):
        calls["registry"] += 1
        return view

    async def fake_load(sf):
        return index

    monkeypatch.setattr(
        "core.application.chat.intent_funnel.registry.active_view", fake_active)
    monkeypatch.setattr(
        "core.application.chat.intent_funnel.recall.load_index", fake_load)
    deps = types.SimpleNamespace(
        session_factory=None, embedder=lambda: embedder, llm=object(),
        cap_router_selector=selector, argument_extractor=extractor)
    return calls, deps


MSG = '新建文件夹"季度报告"'
CAP = [_entry("cap-a", corpus=(MSG,), examples=("建个目录",), parameters=_NAME_SCHEMA)]


async def test_missing_deps_fails_open_to_agent(monkeypatch):
    """Single-path ruling supersedes the old dark-gate test: the
    cascade is always live; the remaining fail-open door is absent deps — the
    registry must never be read, the Agent keeps the turn byte-identical."""
    monkeypatch.setattr(
        "core.application.chat.intent_funnel.registry.active_view",
        lambda **kw: pytest.fail("registry must not be read"),
    )
    req = _req()
    out = await funnel.route(_ctx(MSG), deps=None, requirements=req)
    assert out is req


async def test_route_always_enters_the_cascade(monkeypatch):
    # L0 retirement: neither an L0-certified action nor a memory/demand flag
    # short-circuits the entry — every wired turn reaches the cascade, whose
    # own fail-open keeps the original object.
    _open(monkeypatch)
    calls = {"registry": 0}

    def fake_view(**kw):
        calls["registry"] += 1
        return None

    monkeypatch.setattr(
        "core.application.chat.intent_funnel.registry.active_view", fake_view,
    )
    req = _req(requested_action={"tool": "create_folder", "args": {}},
               needs_memory=True)
    deps = types.SimpleNamespace(session_factory=None)
    out = await funnel.route(_ctx(MSG), deps=deps, requirements=req)
    assert out is req and calls["registry"] == 1


async def test_registry_unavailable_fails_open(monkeypatch, caplog):
    _open(monkeypatch)
    calls, deps = _wire(monkeypatch, view=None, index=None,
                        embedder=_Embedder([1, 0]), selector=_Selector(),
                        extractor=_Extractor())
    req = _req()
    with caplog.at_level(logging.INFO, logger="core.application.chat.intent_funnel"):
        out = await funnel.route(_ctx(MSG), deps=deps, requirements=req)
    assert out is req and calls["registry"] == 1
    line = next(r.getMessage() for r in caplog.records if "funnel_trace" in r.getMessage())
    assert f"fallback_reason={REASON_REGISTRY_UNAVAILABLE}" in line
    assert "deepest_stage=registry" in line and "final_route=agent" in line


async def test_index_unavailable_is_fault_empty_set_is_business(monkeypatch, caplog):
    """Ruling: a missing index is a FAULT (RECALL_UNAVAILABLE);
    an EMPTY candidate set is a normal business result — it short-circuits to
    the Agent at deepest_stage=recall and the selector is NOT consulted.
    E2 (final semantics): only the MISS/AMBIGUOUS lane depends on the index,
    so both legs run on a non-corpus sentence."""
    _open(monkeypatch)
    req = _req()
    sel = _Selector()
    _, deps = _wire(monkeypatch, view=_view(CAP), index=None,
                    embedder=_Embedder([1, 0]), selector=sel,
                    extractor=_Extractor())
    with caplog.at_level(logging.INFO, logger="core.application.chat.intent_funnel"):
        assert await funnel.route(_ctx("随便聊聊"), deps=deps, requirements=req) is req
    assert f"fallback_reason={REASON_RECALL_UNAVAILABLE}" in caplog.records[-1].getMessage()
    assert sel.calls == 0

    sel2 = _Selector()
    _, deps = _wire(monkeypatch, view=_view(CAP), index=_index([]),
                    embedder=_Embedder([1, 0]), selector=sel2,
                    extractor=_Extractor())
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="core.application.chat.intent_funnel"):
        assert await funnel.route(_ctx("随便聊聊"), deps=deps, requirements=req) is req
    line = caplog.records[-1].getMessage()
    assert f"fallback_reason={REASON_NO_CANDIDATE}" in line
    assert "deepest_stage=recall" in line and "tool_intent=-" in line
    assert sel2.calls == 0                          # the selector was NOT consulted


async def test_recall_faults_report_unavailable_never_fake_empty(monkeypatch, caplog):
    """Ruling (point 2): a system fault in recall — embedder error
    or embedder/corpus DIM mismatch (profile config error) — must report
    RECALL_UNAVAILABLE, never the business-result NO_CANDIDATE; no selection
    is attempted."""
    _open(monkeypatch)
    req = _req()
    idx = _index([("cap-a", [MSG], [[1.0, 0.0]])])

    boom_emb = _Embedder([1.0, 0.0], fail=True)
    sel = _Selector()
    _, deps = _wire(monkeypatch, view=_view(CAP), index=idx,
                    embedder=boom_emb, selector=sel, extractor=_Extractor())
    with caplog.at_level(logging.INFO, logger="core.application.chat.intent_funnel"):
        assert await funnel.route(_ctx("随便聊聊"), deps=deps, requirements=req) is req
    line = next(r.getMessage() for r in caplog.records if "funnel_trace" in r.getMessage())
    assert f"fallback_reason={REASON_RECALL_UNAVAILABLE}" in line
    assert "NO_CANDIDATE" not in line
    assert sel.calls == 0

    # dim mismatch: 2-d query vector vs 3-d corpus row — a profile/dim config
    # fault, not a silent truncation that would fake its way to an empty set.
    wide_idx = _index([("cap-a", [MSG], [[1.0, 0.0, 0.0]])])
    sel2 = _Selector()
    _, deps = _wire(monkeypatch, view=_view(CAP), index=wide_idx,
                    embedder=_Embedder([1.0, 0.0]), selector=sel2,
                    extractor=_Extractor())
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="core.application.chat.intent_funnel"):
        assert await funnel.route(_ctx("随便聊聊"), deps=deps, requirements=req) is req
    line = next(r.getMessage() for r in caplog.records if "funnel_trace" in r.getMessage())
    assert f"fallback_reason={REASON_RECALL_UNAVAILABLE}" in line
    assert "dim mismatch" in "\n".join(
        r.getMessage() for r in caplog.records if "fail-open" in r.getMessage())
    assert sel2.calls == 0


async def test_matcher_hit_certifies_through_argument_acquisition(monkeypatch, caplog):
    """HIT: the Matcher already pinned the capability. The cascade must NOT
    consult the selector, must NOT recall, and hands the capability straight to
    the Argument Path Router; the injected extractor fills the MODEL slot and the
    Binder certifies the action."""
    _open(monkeypatch)
    sel, ext = _Selector(), _Extractor({"name": "季度报告"})
    view = _view(CAP)
    _, deps = _wire(monkeypatch, view=view, index=_index([]),
                    embedder=_Embedder([1.0, 0.0]), selector=sel, extractor=ext)
    req = _req()
    with caplog.at_level(logging.INFO, logger="core.application.chat.intent_funnel"):
        out = await funnel.route(_ctx(MSG), deps=deps, requirements=req)
    assert out is not req
    act = out.requested_action
    assert act["tool"] == "create_folder" and act["args"] == {"name": "季度报告"}
    assert act["capability_id"] == "cap-a"
    # live-table ruling: the ONLY Registry stamp is the content fingerprint
    # (the legacy index-version action stamp is gone from the action shape).
    assert "registry_version" not in act
    assert act["funnel_registry_version"] == view.fingerprint
    # ``funnel_stage`` is a frozen observable value (the ACTION handoff).
    assert act["funnel_stage"] == "tool_intent"
    assert out.needs_action is Signal.HIGH and out.complexity is Complexity.LOW
    assert sel.calls == 0                              # NEVER re-selected
    assert len(ext.calls) == 1
    assert ext.calls[0]["query"] == MSG
    assert ext.calls[0]["model_slots"] == ("name",)
    line = next(r.getMessage() for r in caplog.records if "funnel_trace" in r.getMessage())
    assert "final_route=action" in line and "fallback_reason=-" in line
    assert "matcher=HIT:cap-a" in line          # the Matcher made the decision
    assert "tool_intent=-" in line              # no selection hop was spent


async def test_matcher_hit_never_touches_the_recall_index(monkeypatch, caplog):
    """E2 (final semantics): an exact HIT certifies INDEPENDENTLY of Recall
    availability — the HIT lane never calls load_index, so an unembedded or
    faulting corpus can never veto a table-proven turn. The MISS lane keeps
    the RECALL_UNAVAILABLE semantics (pinned above)."""
    _open(monkeypatch)
    ext = _Extractor({"name": "季度报告"})
    _, deps = _wire(monkeypatch, view=_view(CAP), index=None,
                    embedder=_Embedder([1.0, 0.0]), selector=_Selector(),
                    extractor=ext)

    async def boom_load(sf):
        raise AssertionError("E2: the HIT lane must never load the Recall index")

    monkeypatch.setattr(
        "core.application.chat.intent_funnel.recall.load_index", boom_load)
    with caplog.at_level(logging.INFO, logger="core.application.chat.intent_funnel"):
        out = await funnel.route(_ctx(MSG), deps=deps, requirements=_req())
    assert out.requested_action["capability_id"] == "cap-a"
    assert len(ext.calls) == 1
    line = next(r.getMessage() for r in caplog.records if "funnel_trace" in r.getMessage())
    assert "index_version=-" in line and "final_route=action" in line


async def test_recall_lane_selects_then_extracts(monkeypatch):
    """MISS: Recall -> the selector picks the capability -> the extractor fills
    its MODEL slots -> the Binder certifies. The selector sees the recall
    candidates; the extractor sees ONLY the selected capability."""
    _open(monkeypatch)
    sel, ext = _Selector("cap-a"), _Extractor({"name": "季度报告"})
    idx = _index([("cap-a", [MSG], [[1.0, 0.0]])])
    # no table corpus: this is the MISS -> Recall path
    view = _view([_entry("cap-a", examples=("建个目录",), parameters=_NAME_SCHEMA)])
    _, deps = _wire(monkeypatch, view=view, index=idx,
                    embedder=_Embedder([1.0, 0.0]), selector=sel, extractor=ext)
    out = await funnel.route(_ctx(MSG), deps=deps, requirements=_req())
    assert [c.capability_id for c in sel.seen] == ["cap-a"]
    assert sel.seen[0].origin == "recall"
    assert ext.calls[0]["entry"].capability_id == "cap-a"
    assert ext.calls[0]["model_slots"] == ("name",)
    assert out.requested_action["funnel_stage"] == "tool_intent"
    assert out.requested_action["args"] == {"name": "季度报告"}


async def test_ambiguous_carries_all_candidates_into_the_selector(monkeypatch):
    _open(monkeypatch)
    v = _view([_entry("cap-a", corpus=("季度汇总",), parameters=_NAME_SCHEMA),
               _entry("cap-b", tool="add_term", corpus=("季度汇总",),
                      parameters=_TERM_SCHEMA)])
    sel = _Selector("cap-b")
    ext = _Extractor({"term": "季度汇总", "domain": "财务"})
    _, deps = _wire(monkeypatch, view=v, index=_index([]),
                    embedder=_Embedder([1.0, 0.0]), selector=sel, extractor=ext)
    out = await funnel.route(_ctx("季度汇总"), deps=deps, requirements=_req())
    assert sel.calls == 1                          # ONE selection decision
    assert sorted(c.capability_id for c in sel.seen) == ["cap-a", "cap-b"]
    assert out.requested_action["capability_id"] == "cap-b"
    assert out.requested_action["args"] == {"term": "季度汇总", "domain": "财务"}


async def test_empty_extraction_exits_bind_missing(monkeypatch, caplog):
    """The capability is decided but the extractor honestly returns no value:
    the Binder lands MISSING and the Agent owns the clarification."""
    _open(monkeypatch)
    sel, ext = _Selector("cap-a"), _Extractor({})
    view = _view([_entry("cap-a", examples=("建个目录",), parameters=_NAME_SCHEMA)])
    _, deps = _wire(monkeypatch, view=view,
                    index=_index([("cap-a", [MSG], [[1.0, 0.0]])]),
                    embedder=_Embedder([1.0, 0.0]), selector=sel, extractor=ext)
    req = _req()
    with caplog.at_level(logging.INFO, logger="core.application.chat.intent_funnel"):
        out = await funnel.route(_ctx(MSG), deps=deps, requirements=req)
    assert out is req
    line = next(r.getMessage() for r in caplog.records if "funnel_trace" in r.getMessage())
    assert f"fallback_reason={REASON_BIND_MISSING}" in line
    assert "deepest_stage=binder" in line
    assert sel.calls == 1 and len(ext.calls) == 1


async def test_bare_create_folder_demand_is_a_correct_clarification_exit(
    monkeypatch, caplog
):
    """Sim-A/B item 1 pin: "新建一个文件夹" IS a real capability
    demand but the sentence carries NO name. The extractor must not invent one
    and the Binder must not guess one: an empty draft -> MISSING -> Agent
    owns the clarification. This is the designed fallback (report class
    binder_missing_fallback), never a Funnel defect and never a takeover."""
    _open(monkeypatch)
    bare = "新建一个文件夹"
    sel, ext = _Selector("cap-a"), _Extractor({})   # honest: no name to extract
    idx = _index([("cap-a", [bare], [[1.0, 0.0]])])
    view = _view([_entry("cap-a", examples=(bare,), parameters=_NAME_SCHEMA)])
    _, deps = _wire(monkeypatch, view=view, index=idx,
                    embedder=_Embedder([1.0, 0.0]), selector=sel, extractor=ext)
    req = _req()
    with caplog.at_level(logging.INFO, logger="core.application.chat.intent_funnel"):
        out = await funnel.route(_ctx(bare), deps=deps, requirements=req)
    assert out is req                       # Agent keeps the turn, byte-identical
    line = next(r.getMessage() for r in caplog.records if "funnel_trace" in r.getMessage())
    assert f"fallback_reason={REASON_BIND_MISSING}" in line
    assert "deepest_stage=binder" in line and "final_route=agent" in line
    assert len(ext.calls) == 1              # ONE extraction attempt, no re-ask


async def test_extractor_unavailable_exits_agent_without_fabrication(monkeypatch, caplog):
    """The MODEL-owned acquisition path fails closed: an unreachable extractor
    never yields a value, the Agent keeps the turn, and no selector/model call
    is retried."""
    _open(monkeypatch)
    sel, ext = _Selector(), _Extractor(fail=True)
    _, deps = _wire(monkeypatch, view=_view(CAP), index=_index([]),
                    embedder=_Embedder([1.0, 0.0]), selector=sel, extractor=ext)
    req = _req()
    with caplog.at_level(logging.INFO, logger="core.application.chat.intent_funnel"):
        out = await funnel.route(_ctx(MSG), deps=deps, requirements=req)
    assert out is req
    line = next(r.getMessage() for r in caplog.records if "funnel_trace" in r.getMessage())
    assert "fallback_reason=ACQUISITION_MODEL_PENDING" in line
    assert sel.calls == 0 and len(ext.calls) == 1


async def test_matcher_hit_on_a_capability_absent_from_the_table_is_version_mismatch(
        monkeypatch, caplog):
    _open(monkeypatch)
    monkeypatch.setattr(
        matcher, "match",
        lambda *a, **k: MatchResult(state=MATCH_HIT, capability_id="cap-ghost"))
    _, deps = _wire(monkeypatch, view=_view(CAP), index=_index([]),
                    embedder=_Embedder([1.0, 0.0]), selector=_Selector(),
                    extractor=_Extractor())
    req = _req()
    with caplog.at_level(logging.INFO, logger="core.application.chat.intent_funnel"):
        out = await funnel.route(_ctx("x"), deps=deps, requirements=req)
    assert out is req
    line = next(r.getMessage() for r in caplog.records if "funnel_trace" in r.getMessage())
    assert f"fallback_reason={REASON_VERSION_MISMATCH}" in line


async def test_invalid_draft_exits_bind_invalid(monkeypatch, caplog):
    _open(monkeypatch)
    sel = _Selector("cap-a")
    ext = _Extractor({"name": "名" * 121})               # over max_len
    view = _view([_entry("cap-a", examples=("建个目录",), parameters=_NAME_SCHEMA)])
    _, deps = _wire(monkeypatch, view=view,
                    index=_index([("cap-a", [MSG], [[1.0, 0.0]])]),
                    embedder=_Embedder([1.0, 0.0]), selector=sel, extractor=ext)
    req = _req()
    with caplog.at_level(logging.INFO, logger="core.application.chat.intent_funnel"):
        out = await funnel.route(_ctx(MSG), deps=deps, requirements=req)
    assert out is req
    line = next(r.getMessage() for r in caplog.records if "funnel_trace" in r.getMessage())
    assert "fallback_reason=BIND_INVALID" in line


async def test_cascade_timeout_is_attributed_to_its_stage(monkeypatch, caplog):
    _open(monkeypatch, timeout=0.05)

    async def slow_load(sf):
        await asyncio.sleep(0.3)
        return _index([])

    monkeypatch.setattr(
        "core.application.chat.intent_funnel.registry.active_view",
        lambda **kw: asyncio.ensure_future(_view_ok()),
    )
    monkeypatch.setattr("core.application.chat.intent_funnel.recall.load_index", slow_load)
    deps = types.SimpleNamespace(
        session_factory=None, embedder=lambda: _Embedder([1.0, 0.0]), llm=object(),
        cap_router_selector=_Selector(), argument_extractor=_Extractor())
    req = _req()
    # E2: the HIT lane never loads the index, so the recall-stage timeout can
    # only be reached by a MISS — a HIT here would skip slow_load entirely.
    with caplog.at_level(logging.INFO, logger="core.application.chat.intent_funnel"):
        out = await funnel.route(_ctx("随便聊聊"), deps=deps, requirements=req)
    assert out is req
    line = next(r.getMessage() for r in caplog.records if "funnel_trace" in r.getMessage())
    assert f"fallback_reason={REASON_RECALL_TIMEOUT}" in line
    assert "deepest_stage=recall" in line


async def _view_ok():
    return _view(CAP)


async def test_negated_matcher_hit_is_forced_to_miss(monkeypatch, caplog):
    _open(monkeypatch)
    # the corpus itself contains the negated sentence: an exact HIT WOULD fire —
    # only the 8.1-a guard can veto it. With the HIT vetoed and recall empty,
    # the model-facing set is empty -> NO_CANDIDATE short-circuit (ruling
    # ); the selector is never consulted on a turn with no card.
    view = _view([_entry("cap-a", corpus=("不要新建文件夹",))])
    sel, ext = _Selector(), _Extractor()
    _, deps = _wire(monkeypatch, view=view, index=_index([]),
                    embedder=_Embedder([1.0, 0.0]), selector=sel, extractor=ext)
    req = _req()
    with caplog.at_level(logging.INFO, logger="core.application.chat.intent_funnel"):
        out = await funnel.route(_ctx("不要新建文件夹"), deps=deps, requirements=req)
    assert out is req
    line = next(r.getMessage() for r in caplog.records if "funnel_trace" in r.getMessage())
    assert "matcher=MISS" in line
    assert f"fallback_reason={REASON_NO_CANDIDATE}" in line
    assert sel.calls == 0 and ext.calls == []


# ═══════════ E1/E2 — final semantics landing ( audit) ═════════════════


def test_aggregate_by_capability_keeps_the_winning_candidate():
    """E1 unit: ONE capability-level candidate per capability_id — the highest-
    scoring hit rides WITH ITS OWN provenance; ties keep the earlier arrival
    (the matcher seed is seeded first, so table evidence wins a tie); distinct
    capabilities are all preserved (no top_k, no dropping)."""
    from core.application.chat.intent_funnel.candidate_aggregation import (
        aggregate_by_capability as _aggregate_by_capability,
    )

    low = Candidate("cap-a", 0.89, matched_example="低", query_id="q2")
    win = Candidate("cap-a", 0.94, matched_example="高", query_id="q1")
    other = Candidate("cap-b", 0.86, matched_example="图", query_id="q3")
    amb = Candidate("cap-a", 0.0, origin="matcher_ambiguous")
    out = _aggregate_by_capability([amb, low, win, other])
    assert len(out) == 2                       # summary x3 + mindmap -> summary + mindmap
    assert out[0] is win and out[1] is other   # max rides with its provenance
    t1 = Candidate("cap-a", 0.90, matched_example="先")
    t2 = Candidate("cap-a", 0.90, matched_example="后")
    assert _aggregate_by_capability([t1, t2])[0] is t1


async def test_capability_aggregation_collapses_duplicate_hits_into_one_card(monkeypatch):
    """E1 integration: Raw Recall keeps every >= threshold hit (pinned at the
    recall node above), but the model-facing list is capability-level — cap-a
    arrives twice (cos 1.000 + 0.981) and the selector sees ONE cap-a candidate
    at the WINNING score/provenance; cap-b is untouched."""
    _open(monkeypatch)
    sel = _Selector("cap-a")
    ext = _Extractor({"name": "季度报告"})
    idx = _index([("cap-a", ["近a", "远b"], [[1.0, 0.0], [1.0, 0.2]]),
                  ("cap-b", ["加个词"], [[1.0, 0.4]])])
    view = _view([_entry("cap-a", examples=("做个事",), parameters=_NAME_SCHEMA),
                  _entry("cap-b", tool="add_term", examples=("写个词",),
                         parameters=_TERM_SCHEMA)])
    _, deps = _wire(monkeypatch, view=view, index=idx,
                    embedder=_Embedder([1.0, 0.0]), selector=sel, extractor=ext)
    out = await funnel.route(_ctx("随便聊聊"), deps=deps, requirements=_req())
    assert sel.calls == 1
    assert [(c.capability_id, c.matched_example) for c in sel.seen] == [
        ("cap-a", "近a"), ("cap-b", "加个词")]
    assert out.requested_action["capability_id"] == "cap-a"
