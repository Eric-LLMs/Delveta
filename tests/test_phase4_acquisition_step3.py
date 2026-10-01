"""Phase 4 Step 3 (2026-10-01) — MIXED argument merge WIRING.

Drives the REAL cascade (``run_nodes``) on the new lane
(``chat_cap_router_backend != off``) and pins the merge contract:

* SYSTEM_BINDER slot + MODEL slot -> the ARP derives **MIXED**;
* ``merge`` combines the injected ``system_values`` with the injected
  ``model_values`` into ONE draft, **system wins on any collision**;
* provenance records the ACTUAL source of each side (system: ``system_sources``,
  model: ``model_source``) — never a merely-allowed source;
* an unprovenanced system value is NOT merged (never fabricated);
* the merged draft reaches the EXISTING ``binder.validate`` unchanged, and the
  certified action carries the merged args;
* ``asset_id`` (SYSTEM_BINDER) is NOT authoritative here: the Binder still
  resolves it from TurnFacts, so a divergent seam value loses to the fact truth
  (decision 2026-10-01, option (a));
* no Qwen, no legacy ``select_and_extract`` on the new lane; QUERY_TO_QWEN /
  QUERY_PLUS_5_USER_TURNS still exit ``ACQUISITION_MODEL_PENDING``;
* ``backend=off`` keeps the legacy lane byte-for-byte.

Test data is REAL where a real asset id matters: the id is a real upload blob
(``data/objects/uploads/<uuid>/chunk_0``, ``%PDF`` magic), READ-ONLY. The MODEL
side is per-test injected through the acquisition seam (no Qwen is wired).
"""
from __future__ import annotations

import types
from pathlib import Path

import pytest

from core.application.chat.intent_funnel import matcher as matcher_mod
from core.application.chat.intent_funnel import orchestrator as orch_mod
from core.application.chat.intent_funnel import tool_intent as tool_intent_mod
from core.application.chat.intent_funnel.argument_acquisition import (
    AcquisitionInputs,
    SlotDecl,
)
from core.application.chat.intent_funnel.argument_acquisition.contract import (
    OWNERSHIP_MODEL,
    OWNERSHIP_SYSTEM_BINDER,
    SOURCE_CONVERSATION_5_USER_TURNS,
    SOURCE_QUERY,
    SOURCE_UI_CONTEXT,
    STRATEGY_CONTEXT_DIRECT,
    STRATEGY_MIXED,
    STRATEGY_QUERY_PLUS_5_USER_TURNS,
    STRATEGY_QUERY_TO_QWEN,
    ArgumentProvenance,
)
from core.application.chat.intent_funnel.argument_acquisition.merge import merge
from core.application.chat.intent_funnel.contract import (
    MATCH_HIT,
    REASON_ACQUISITION_MODEL_PENDING,
    MatchResult,
)
from core.application.chat.intent_funnel.observability import new_trace
from core.application.chat.intent_funnel.registry import content_fingerprint
from core.application.chat.intent_funnel.registry.entry import (
    CapabilityEntry,
    QueryRecord,
    RegistryLiveView,
    derive_language,
)
from core.application.chat.understanding import TurnRequirements
from core.config import settings

UPLOADS = Path(__file__).resolve().parents[1] / "data" / "objects" / "uploads"


# ── the real-PDF corpus (read-only; same source as the Step 2 suite) ─────────────


def _real_pdf_assets() -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    if not UPLOADS.is_dir():
        return out
    for d in sorted(UPLOADS.iterdir()):
        blob = d / "chunk_0"
        if not (d.is_dir() and blob.is_file()):
            continue
        try:
            with blob.open("rb") as fh:
                if fh.read(4) != b"%PDF":
                    continue
        except OSError:
            continue
        out.append((d.name, blob.stat().st_size))
    return out


@pytest.fixture(scope="module")
def real_pdf() -> tuple[str, int]:
    assets = _real_pdf_assets()
    if not assets:
        pytest.skip("no real PDF upload blob under data/objects/uploads")
    return assets[0]


# ── A. merge() unit surface (pure; no cascade) ───────────────────────────────────

_DECL = {
    "asset_id": SlotDecl(OWNERSHIP_SYSTEM_BINDER, allowed_sources=(SOURCE_UI_CONTEXT,)),
    "pages": SlotDecl(OWNERSHIP_MODEL, allowed_sources=(SOURCE_QUERY,)),
}


def _ass() -> dict:
    """The declaration dict the router would consume."""
    return dict(_DECL)


def test_merge_combines_system_and_model_values():
    merged = merge({"asset_id": "A1"}, {"pages": "3"},
                   declaration=_ass(), system_sources={"asset_id": SOURCE_UI_CONTEXT},
                   model_source=SOURCE_QUERY)
    assert merged.args == {"asset_id": "A1", "pages": "3"}
    assert merged.unprovenanced == ()


def test_merge_system_value_wins_over_same_slot_model_value():
    # (a) a declared SYSTEM_BINDER slot: the model value is STRIPPED, not merged.
    declared = merge({"asset_id": "SYS"}, {"asset_id": "MODEL"},
                     declaration=_ass(),
                     system_sources={"asset_id": SOURCE_UI_CONTEXT},
                     model_source=SOURCE_QUERY)
    assert declared.args["asset_id"] == "SYS"
    # (b) an undeclared collision: the system loop still overwrites the model one.
    raw = merge({"mode": "SYS"}, {"mode": "MODEL"}, declaration={},
                system_sources={"mode": SOURCE_UI_CONTEXT},
                model_source=SOURCE_QUERY)
    assert raw.args["mode"] == "SYS"


def test_merge_records_system_provenance():
    merged = merge({"asset_id": "A1"}, {"pages": "3"},
                   declaration=_ass(), system_sources={"asset_id": SOURCE_UI_CONTEXT},
                   model_source=SOURCE_QUERY)
    prov = {p.slot: p for p in merged.provenance}
    assert prov["asset_id"] == ArgumentProvenance(
        slot="asset_id", source=SOURCE_UI_CONTEXT, produced_by=OWNERSHIP_SYSTEM_BINDER)


def test_merge_records_model_provenance():
    merged = merge({"asset_id": "A1"}, {"pages": "3"},
                   declaration=_ass(), system_sources={"asset_id": SOURCE_UI_CONTEXT},
                   model_source=SOURCE_QUERY)
    prov = {p.slot: p for p in merged.provenance}
    assert prov["pages"] == ArgumentProvenance(
        slot="pages", source=SOURCE_QUERY, produced_by=OWNERSHIP_MODEL)


def test_merge_unprovenanced_system_value_not_merged():
    # No ACTUAL source supplied for asset_id -> not merged, never fabricated.
    merged = merge({"asset_id": "A1"}, {"pages": "3"},
                   declaration=_ass(), system_sources={},
                   model_source=SOURCE_QUERY)
    assert "asset_id" not in merged.args
    assert merged.args == {"pages": "3"}
    assert merged.unprovenanced == ("asset_id",)


# ── B. cascade wiring (HIT -> ARP -> MIXED -> merge -> Binder -> certified) ───────


def _ctx(message: str, *, open_asset_id: str = ""):
    viewer = (types.SimpleNamespace(asset_id=open_asset_id, page=None, selections=[])
              if open_asset_id else None)
    return types.SimpleNamespace(
        body=types.SimpleNamespace(message=message, attach=None, viewer=viewer),
        owned_asset_id=None, path_asset_id=open_asset_id, research_turn=False,
        effective_handoff=None, session_id="s1", history=[],
    )


def _entry(cid: str, parameters: dict) -> CapabilityEntry:
    q = "打开这个文件"
    return CapabilityEntry(
        capability_id=cid, tool_binding="open_file",
        description="open a PDF from My Drive",
        standard_queries=(QueryRecord(id=f"{cid}-q1", query=q,
                                      language=derive_language(q)),),
        parameters=parameters,
    )


class _Rec:
    def __init__(self) -> None:
        self.legacy: list = []


def _wire(monkeypatch, entry: CapabilityEntry, *, backend: str,
          legacy_verdict=None) -> _Rec:
    view = RegistryLiveView(fingerprint=content_fingerprint([entry]), entries=(entry,))
    rec = _Rec()

    async def fake_active(**kw):
        return view

    def fake_match(message, facts, v):
        return MatchResult(state=MATCH_HIT, capability_id=entry.capability_id,
                           matched_literal="打开这个文件")

    async def fake_select_and_extract(query, candidates, *, entries_by_id,
                                      llm=None, facts=None):
        rec.legacy.append(query)
        if legacy_verdict is None:
            raise AssertionError("legacy select_and_extract must not run on the new lane")
        return legacy_verdict

    monkeypatch.setattr(
        "core.application.chat.intent_funnel.registry.active_view", fake_active)
    monkeypatch.setattr(matcher_mod, "match", fake_match)
    monkeypatch.setattr(tool_intent_mod, "select_and_extract", fake_select_and_extract)
    monkeypatch.setattr(settings, "chat_cap_router_backend", backend)
    return rec


def _deps(provider):
    return types.SimpleNamespace(
        session_factory=None, embedder=lambda: object(), llm=object(),
        acquisition_inputs=provider)


async def _run(monkeypatch, *, entry, ctx, inputs, backend="stub"):
    rec = _wire(monkeypatch, entry, backend=backend)
    provider = (lambda cid: inputs.get(cid)) if inputs is not None else None
    req = TurnRequirements()
    trace = new_trace()
    capture: dict = {}
    out = await orch_mod.run_nodes(ctx, _deps(provider), req, trace, capture=capture)
    return out, req, rec, trace, capture


_MIXED_PARAMS = {
    "asset_id": {"type": "string", "required": True, "description": "the PDF"},
    "pages": {"type": "string", "required": True, "description": "page range"},
}


def _mixed_inputs(asset_id: str, *, seam_asset: str | None = None) -> dict:
    return {"cap-open-pdf": AcquisitionInputs(
        declaration=_ass(),
        evidence={"pages": SOURCE_QUERY},
        system_values={"asset_id": asset_id if seam_asset is None else seam_asset},
        system_sources={"asset_id": SOURCE_UI_CONTEXT},
        model_values={"pages": "3"},
        model_source=SOURCE_QUERY,
    )}


async def test_mixed_merges_and_certifies(monkeypatch, real_pdf):
    cid, _ = real_pdf
    entry = _entry("cap-open-pdf", _MIXED_PARAMS)
    out, req, rec, trace, capture = await _run(
        monkeypatch, entry=entry,
        ctx=_ctx("打开这个文件第3页", open_asset_id=cid),
        inputs=_mixed_inputs(cid))

    assert not rec.legacy                                   # no Qwen / no legacy hop
    assert capture["acquisition"]["strategy"] == STRATEGY_MIXED
    assert out is not req                                   # a CERTIFIED turn
    act = out.requested_action
    assert act["capability_id"] == "cap-open-pdf"
    assert act["args"] == {"asset_id": cid, "pages": "3"}   # merged draft, Binder-approved
    assert act["funnel_stage"] == "tool_intent"             # the EXISTING handoff shape


async def test_mixed_provenance_is_recorded(monkeypatch, real_pdf):
    # telemetry only: capture["acquisition"]["provenance"], never the action schema.
    cid, _ = real_pdf
    entry = _entry("cap-open-pdf", _MIXED_PARAMS)
    out, req, rec, trace, capture = await _run(
        monkeypatch, entry=entry,
        ctx=_ctx("打开这个文件第3页", open_asset_id=cid),
        inputs=_mixed_inputs(cid))
    prov = {p["slot"]: p for p in capture["acquisition"]["provenance"]}
    assert prov["asset_id"]["source"] == SOURCE_UI_CONTEXT
    assert prov["asset_id"]["produced_by"] == OWNERSHIP_SYSTEM_BINDER
    assert prov["pages"]["source"] == SOURCE_QUERY
    assert prov["pages"]["produced_by"] == OWNERSHIP_MODEL
    # the certified action schema is UNCHANGED: no provenance key on the action.
    assert "provenance" not in out.requested_action


async def test_mixed_asset_id_final_authority_is_turn_facts(monkeypatch, real_pdf):
    # The seam supplies a DIVERGENT asset id; the Binder still resolves asset_id
    # from TurnFacts (decision 2026-10-01, option (a)) -> the fact truth wins.
    cid, _ = real_pdf
    entry = _entry("cap-open-pdf", _MIXED_PARAMS)
    out, req, rec, trace, capture = await _run(
        monkeypatch, entry=entry,
        ctx=_ctx("打开这个文件第3页", open_asset_id=cid),
        inputs=_mixed_inputs(cid, seam_asset="SEAM-DIVERGENT-ASSET"))
    assert out is not req
    assert out.requested_action["args"]["asset_id"] == cid      # NOT the seam value
    assert out.requested_action["args"]["pages"] == "3"
    # the merge still recorded the seam-sourced provenance it was GIVEN (telemetry).
    prov = {p["slot"]: p for p in capture["acquisition"]["provenance"]}
    assert prov["asset_id"]["source"] == SOURCE_UI_CONTEXT


async def test_query_to_qwen_still_model_pending(monkeypatch, real_pdf):
    cid, _ = real_pdf
    entry = _entry("cap-open-pdf", {
        "pages": {"type": "string", "required": True, "description": "page range"}})
    inputs = {"cap-open-pdf": AcquisitionInputs(
        declaration={"pages": SlotDecl(OWNERSHIP_MODEL, allowed_sources=(SOURCE_QUERY,))},
        evidence={"pages": SOURCE_QUERY},
        model_values={"pages": "3"}, model_source=SOURCE_QUERY)}
    out, req, rec, trace, capture = await _run(
        monkeypatch, entry=entry, ctx=_ctx("打开这个文件第3页", open_asset_id=cid),
        inputs=inputs)
    assert trace["acquisition"] == STRATEGY_QUERY_TO_QWEN
    assert out is None and trace["fallback"] == REASON_ACQUISITION_MODEL_PENDING
    assert not rec.legacy


async def test_query_plus_5_user_turns_still_model_pending(monkeypatch, real_pdf):
    cid, _ = real_pdf
    entry = _entry("cap-open-pdf", {
        "pages": {"type": "string", "required": True, "description": "page range"}})
    inputs = {"cap-open-pdf": AcquisitionInputs(
        declaration={"pages": SlotDecl(
            OWNERSHIP_MODEL,
            allowed_sources=(SOURCE_QUERY, SOURCE_CONVERSATION_5_USER_TURNS))},
        evidence={"pages": SOURCE_CONVERSATION_5_USER_TURNS},
        model_values={"pages": "3"}, model_source=SOURCE_CONVERSATION_5_USER_TURNS)}
    out, req, rec, trace, capture = await _run(
        monkeypatch, entry=entry, ctx=_ctx("打开这个文件第3页", open_asset_id=cid),
        inputs=inputs)
    assert trace["acquisition"] == STRATEGY_QUERY_PLUS_5_USER_TURNS
    assert out is None and trace["fallback"] == REASON_ACQUISITION_MODEL_PENDING
    assert not rec.legacy


async def test_context_direct_has_no_model_side(monkeypatch, real_pdf):
    # a system-only capability still takes CONTEXT_DIRECT — merge is never entered.
    cid, _ = real_pdf
    entry = _entry("cap-open-pdf", {
        "asset_id": {"type": "string", "required": True, "description": "the PDF"}})
    inputs = {"cap-open-pdf": AcquisitionInputs(
        declaration={"asset_id": SlotDecl(OWNERSHIP_SYSTEM_BINDER,
                                          allowed_sources=(SOURCE_UI_CONTEXT,))},
        system_values={"asset_id": cid},
        system_sources={"asset_id": SOURCE_UI_CONTEXT})}
    out, req, rec, trace, capture = await _run(
        monkeypatch, entry=entry, ctx=_ctx("打开这个文件", open_asset_id=cid),
        inputs=inputs)
    assert capture["acquisition"]["strategy"] == STRATEGY_CONTEXT_DIRECT
    assert out is not req
    assert out.requested_action["args"] == {"asset_id": cid}
    assert "provenance" not in capture["acquisition"]
    assert not rec.legacy


# ── C. backend=off: the legacy lane is untouched ─────────────────────────────────


async def test_backend_off_keeps_legacy_lane(monkeypatch, real_pdf):
    from core.application.chat.intent_funnel.contract import (
        TOOL_INTENT_CONFIDENT,
        ToolIntentVerdict,
    )

    cid, _ = real_pdf
    entry = _entry("cap-open-pdf", _MIXED_PARAMS)
    verdict = ToolIntentVerdict(TOOL_INTENT_CONFIDENT, "cap-open-pdf",
                                arguments={"asset_id": cid, "pages": "9"})
    rec = _wire(monkeypatch, entry, backend="off", legacy_verdict=verdict)
    req = TurnRequirements()
    trace = new_trace()
    capture: dict = {}
    out = await orch_mod.run_nodes(_ctx("打开这个文件第9页", open_asset_id=cid),
                                   _deps(None), req, trace, capture=capture)
    assert rec.legacy                                     # legacy hop WAS used
    assert out is not req
    assert out.requested_action["args"]["pages"] == "9"   # legacy extraction result
    assert "acquisition" not in capture                   # ARP never entered
