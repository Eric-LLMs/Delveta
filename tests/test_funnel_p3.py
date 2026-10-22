"""P3 — intent_kind: the table widened, the gates did not.

Pinning the P3 ruling ("扩表不改接口,逐开关灰度"):

* the kind is Registry DATA: it rides the live capability row, the snapshot
  payload and the content fingerprint (changing a kind changes the fingerprint);
* pre-P3 payloads and rows default to ACTION — a historical snapshot can never
  restore a widened kind that the gates do not honor;
* a decided capability on a CLOSED kind exits with FUNNEL_KIND_DISABLED and the
  Agent turn stays byte-identical (8.10);
* the interface: nothing outside the Registry changed — the selection /
  acquisition seams are the same ``ChatDeps`` contracts every lane uses.
"""
from __future__ import annotations

import logging
import types

from core.application.chat.intent_funnel import cap_router, funnel
from core.application.chat.intent_funnel.contract import REASON_KIND_DISABLED
from core.application.chat.intent_funnel.registry import content_fingerprint
from core.application.chat.intent_funnel.registry import snapshot as pub
from core.application.chat.intent_funnel.registry.entry import (
    KIND_ACTION,
    KIND_PRIVATE,
    KIND_WEB,
    CapabilityEntry,
    QueryRecord,
    RegistryLiveView,
    derive_language,
)

MSG = '新建文件夹"季度报告"'


def _entry(cid, *, tool="create_folder", corpus=(), kind=KIND_ACTION, **kw):
    # corpus = the exact-set sentences (Standard + Similar live rows)
    corpus = tuple(corpus)
    std = (QueryRecord(id="q1", query=corpus[0], language=derive_language(corpus[0])),) \
        if corpus else ()
    sims = tuple(QueryRecord(id=f"q{10 + n}", query=s, language=derive_language(s),
                             position=n, standard_query_id="q1" if std else None)
                 for n, s in enumerate(corpus[1:]))
    return CapabilityEntry(
        capability_id=cid, tool_binding=tool, description=f"does {cid}",
        standard_queries=std, similar_queries=sims,
        parameters={"name": {"type": "string", "required": True,
                             "max_len": 120, "description": "folder name"}},
        arg_slots={"name": {"source": "user_input"}}, intent_kind=kind, **kw,
    )


def _view(entries, version=1):
    return RegistryLiveView(
        fingerprint=content_fingerprint(list(entries)),
        entries=tuple(entries),
    )


# ════════════════════════ data: the kind round-trips everywhere ═════════════════


def test_kind_round_trips_through_payload_and_defaults_to_action():
    e = _entry("cap-p", kind=KIND_PRIVATE)
    assert CapabilityEntry.from_payload(e.to_payload()).intent_kind == KIND_PRIVATE
    # a pre-P3 history payload has no kind field at all -> ACTION, the safe
    # historical default (an old snapshot can never restore a widened kind)
    legacy = {k: v for k, v in e.to_payload().items() if k != "intent_kind"}
    assert CapabilityEntry.from_payload(legacy).intent_kind == KIND_ACTION


def test_kind_is_content_so_it_moves_the_fingerprint():
    a = _view([_entry("cap-a", kind=KIND_ACTION)])
    p = _view([_entry("cap-a", kind=KIND_PRIVATE)])
    assert a.fingerprint != p.fingerprint


def test_validation_gate_rejects_unknown_kind():
    issues = pub.validate_entries([_entry("cap-x", kind="quantum")])
    assert any("intent_kind" in s for s in issues)
    # the three real kinds pass the kind rule (other rules may still apply)
    assert not [s for s in pub.validate_entries([_entry("cap-x", kind=KIND_WEB)])
                if "intent_kind" in s]


def test_kind_enabled_matrix():
    # Single-path ruling: kind_enabled is settings-free — only the
    # action family rides the live funnel; private/web kinds are fail-closed
    # forever (the dev switches guarded zero live Registry rows), unknown kinds
    # route nothing.
    assert funnel.kind_enabled(KIND_ACTION)
    assert funnel.kind_enabled("")                   # historical rows: action
    assert not funnel.kind_enabled("quantum")        # unknown kind routes nothing
    assert not funnel.kind_enabled(KIND_PRIVATE)
    assert not funnel.kind_enabled(KIND_WEB)


# ════════════════════════ cascade: in the table != ON (8.10 exit) ═══════════════


def _ctx(msg):
    return types.SimpleNamespace(
        body=types.SimpleNamespace(message=msg, attach=None, viewer=None),
        owned_asset_id=None, research_turn=False, effective_handoff=None,
        session_id="s-1",
    )


def _req():
    from core.application.chat.understanding import (
        Complexity,
        Confidence,
        Signal,
        TurnRequirements,
    )
    return TurnRequirements(complexity=Complexity.LOW, confidence=Confidence.LOW,
                            needs_web=Signal.LOW, needs_memory=False)


class _Embedder:
    async def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


def _open(monkeypatch):
    from core.config import settings

    # Single-path ruling: the rollout gates + matcher shadow mode were
    # deleted; ACTION is the shipped kind, every other kind fail-closes unconditionally.
    monkeypatch.setattr(settings, "chat_tool_intent_timeout_seconds", 5.0)
    monkeypatch.setattr(settings, "chat_funnel_timeout_seconds", 5.0)


class _Selector:
    """Deterministic selector double (no model call): picks the top candidate."""

    def __init__(self):
        self.calls = 0

    async def select(self, query, candidates, *, entries_by_id, facts=None):
        self.calls += 1
        return cap_router.CapabilityRoute(
            cap_router.ROUTE_SELECTED, candidates[0].capability_id)


class _Extractor:
    """Deterministic argument-extractor double: quote-strips the name."""

    def __init__(self):
        self.calls = 0

    async def __call__(self, *, query, entry, model_slots, bundle, prompt=None):
        self.calls += 1
        quoted = query.split('"')
        return ({"name": quoted[1]} if len(quoted) > 1 else {}), bundle.source


def _wire(monkeypatch, *, view):
    async def fake_active(**kw):
        return view

    async def fake_load(sf):
        return types.SimpleNamespace(version="corpus1-test", corpus=())

    monkeypatch.setattr(
        "core.application.chat.intent_funnel.registry.active_view", fake_active)
    monkeypatch.setattr(
        "core.application.chat.intent_funnel.recall.load_index", fake_load)
    return types.SimpleNamespace(
        session_factory=None, embedder=lambda: _Embedder(), llm=object(),
        cap_router_selector=_Selector(), argument_extractor=_Extractor())


async def test_closed_private_kind_exits_with_reason_byte_identical(monkeypatch, caplog):
    _open(monkeypatch)
    view = _view([_entry("cap-p", kind=KIND_PRIVATE, corpus=(MSG,))])
    deps = _wire(monkeypatch, view=view)
    req = _req()
    with caplog.at_level(logging.INFO, logger="core.application.chat.intent_funnel"):
        out = await funnel.route(_ctx(MSG), deps=deps, requirements=req)
    assert out is req                      # Agent keeps the turn (8.10)
    line = next(r.getMessage() for r in caplog.records if "funnel_trace" in r.getMessage())
    assert f"fallback_reason={REASON_KIND_DISABLED}" in line


async def test_private_kind_fail_closed_without_any_switch(monkeypatch, caplog):
    """Single-path ruling: the private rollout switch was deleted —
    non-ACTION kinds fail closed unconditionally until the lane re-ships."""
    _open(monkeypatch)
    view = _view([_entry("cap-p", kind=KIND_PRIVATE, corpus=(MSG,))])
    deps = _wire(monkeypatch, view=view)
    req = _req()
    with caplog.at_level(logging.INFO, logger="core.application.chat.intent_funnel"):
        out = await funnel.route(_ctx(MSG), deps=deps, requirements=req)
    assert out is req                      # Agent keeps the turn (8.10)
    line = next(r.getMessage() for r in caplog.records if "funnel_trace" in r.getMessage())
    assert f"fallback_reason={REASON_KIND_DISABLED}" in line


async def test_action_kind_needs_no_extra_switch(monkeypatch):
    _open(monkeypatch)
    view = _view([_entry("cap-a", kind=KIND_ACTION, corpus=(MSG,))])
    deps = _wire(monkeypatch, view=view)
    out = await funnel.route(_ctx(MSG), deps=deps, requirements=_req())
    assert out.requested_action["funnel_kind"] == KIND_ACTION
