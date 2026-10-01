"""Phase 4 Step 2 (2026-10-01) — Argument Path Router WIRING.

Drives the REAL cascade (``funnel.route`` -> orchestrator) on the new lane
(``chat_cap_router_backend != off``) and pins the wiring contract:

* MATCH_HIT -> Argument Path Router DIRECTLY: no Recall, no Candidate
  Aggregation, no cap_router, no re-selection;
* CONTEXT_DIRECT reuses the EXISTING Binder/certified handoff — this step adds
  no merge/validation logic of its own — consuming ONLY the legal system values
  supplied through the injected ``acquisition_inputs`` seam;
* an acquisition-undeclared capability, and a MISSING (no legal source) one,
  each exit to the Agent;
* a MODEL acquisition need (QUERY_TO_QWEN / QUERY_PLUS_5_USER_TURNS / MIXED)
  requires Qwen, which this step does not wire -> it exits to the Agent and the
  legacy ``select_and_extract`` is never called.

Test data is REAL: the asset id is a real upload blob (``data/objects/uploads/
<uuid>/chunk_0``, ``%PDF`` magic). A fake-DB drive fixture establishes the
logical ``My Drive/<name>.pdf -> asset_id`` record and a simulated UI "open"
feeds that asset id into the turn facts — the blob itself is READ-ONLY, never
moved, copied or modified. No production slot->fact mapping is invented here;
the acquisition inputs are injected per capability by a fixture provider.
"""
from __future__ import annotations

import logging
import re
import types
from pathlib import Path

import pytest

from core.application.chat.intent_funnel import matcher as matcher_mod
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
)
from core.application.chat.intent_funnel.contract import (
    MATCH_HIT,
    REASON_ACQUISITION_MISSING,
    REASON_ACQUISITION_MODEL_PENDING,
    REASON_ACQUISITION_UNDECLARED,
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
UPLOADS = Path(__file__).resolve().parents[1] / "data" / "objects" / "uploads"


# ── the real-PDF corpus (read-only) ──────────────────────────────────────────────


def _real_pdf_assets() -> list[tuple[str, int]]:
    """(asset_id, size) for every real uploaded blob whose first bytes are %PDF.
    The blob DIRECTORY NAME is the asset id (chunked uploads: chunk_0 is the
    assembled object). Nothing is written — this only reads 4 magic bytes."""
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


# ── a fake-DB "My Drive" (logical record only; the blob is untouched) ─────────────


class _FakeDrive:
    """In-memory My Drive: the logical ``<name>.pdf -> asset_id`` mapping. It
    mirrors ``DriveService.resolve_personal_path`` semantics (My Drive/ prefix
    strip, exact folder + name) so the test exercises the SAME resolution shape
    without a DB — and never touches the real blob it points at."""

    def __init__(self) -> None:
        self._by_path: dict[tuple[str, str], str] = {}

    def register(self, name: str, asset_id: str, folder: str = "") -> None:
        self._by_path[(folder, name)] = asset_id

    def resolve_personal_path(self, user_id, path: str) -> dict | None:
        p = (path or "").strip().rstrip("/")
        for root in ("My Drive/", "我的云盘/"):
            if p.startswith(root):
                p = p[len(root):]
                break
        p = p.lstrip("/")
        folder, _, name = p.rpartition("/")
        asset_id = self._by_path.get((folder, name))
        if asset_id is None:
            return None
        return {"asset_id": asset_id, "name": name, "folder_path": folder}


# ── turn context (simulated UI "open the PDF") ───────────────────────────────────


def _ctx(message: str, *, open_asset_id: str = ""):
    """One turn's context. ``open_asset_id`` simulates the UI having the PDF on
    screen: it rides ``body.viewer.asset_id``, which ``TurnFacts.of`` lifts to
    ``viewer_asset_id`` — the fact the Binder resolves the ``asset_id`` slot
    from. It also lifts the entry guardrail's referenced-input veto."""
    viewer = (types.SimpleNamespace(asset_id=open_asset_id, page=None, selections=[])
              if open_asset_id else None)
    return types.SimpleNamespace(
        body=types.SimpleNamespace(message=message, attach=None, viewer=viewer),
        owned_asset_id=None, path_asset_id=open_asset_id, research_turn=False,
        effective_handoff=None, session_id="s1", history=[],
    )


# ── a capability whose schema/declaration are supplied per scenario ──────────────


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


def _wire(monkeypatch, entry: CapabilityEntry) -> _Rec:
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


def _fallback(caplog) -> str:
    lines = [r.getMessage() for r in caplog.records
             if r.name == FUNNEL_LOGGER and "funnel_trace" in r.getMessage()]
    assert len(lines) == 1, lines
    return re.search(r"fallback_reason=(\S+)", lines[0]).group(1)


async def _run(monkeypatch, caplog, *, entry, ctx, inputs):
    caplog.set_level(logging.INFO, logger=FUNNEL_LOGGER)
    rec, view = _wire(monkeypatch, entry)
    provider = (lambda cid: inputs.get(cid)) if inputs is not None else None
    req = TurnRequirements()
    out = await _route(ctx, _deps(provider), req)
    return out, req, rec, view


async def _route(ctx, deps, req):
    from core.application.chat.intent_funnel import funnel
    return await funnel.route(ctx, deps=deps, requirements=req)


# ── A/B. CONTEXT_DIRECT: existing Binder reused, no Qwen, optional MODEL inert ────


async def test_context_direct_reuses_binder_with_a_real_pdf_asset(
        monkeypatch, caplog, real_pdf):
    cid, size = real_pdf
    assert size > 4                      # a real, non-empty upload blob
    drive = _FakeDrive()
    drive.register("spec.pdf", cid)      # logical My Drive record -> the real asset
    resolved = drive.resolve_personal_path("u1", "My Drive/spec.pdf")
    assert resolved["asset_id"] == cid

    # pages is an OPTIONAL MODEL slot present in the schema: mere presence is NOT
    # evidence (§D Gate 2), so this stays CONTEXT_DIRECT — no Qwen.
    entry = _entry("cap-open-pdf", {
        "asset_id": {"type": "string", "required": True, "description": "the PDF"},
        "pages": {"type": "string", "required": False, "description": "page range"},
    })
    inputs = {"cap-open-pdf": AcquisitionInputs(
        declaration={"asset_id": SlotDecl(OWNERSHIP_SYSTEM_BINDER,
                                          allowed_sources=(SOURCE_UI_CONTEXT,))},
        system_values={"asset_id": cid},
        system_sources={"asset_id": SOURCE_UI_CONTEXT},
    )}
    out, req, rec, view = await _run(
        monkeypatch, caplog, entry=entry,
        ctx=_ctx("打开这个文件", open_asset_id=resolved["asset_id"]), inputs=inputs)

    assert not rec.legacy                            # no legacy hop on the new lane
    assert out is not req                            # a CERTIFIED turn
    act = out.requested_action
    assert act["capability_id"] == "cap-open-pdf"
    assert act["args"] == {"asset_id": cid}          # optional pages omitted
    assert act["funnel_registry_version"] == view.fingerprint
    assert act["funnel_stage"] == "tool_intent"      # the EXISTING handoff shape


# ── C. undeclared capability -> Agent (§G) ───────────────────────────────────────


async def test_undeclared_capability_exits_to_agent(monkeypatch, caplog, real_pdf):
    cid, _ = real_pdf
    entry = _entry("cap-open-pdf", {
        "asset_id": {"type": "string", "required": True, "description": "the PDF"},
    })
    # no declaration supplied -> the ARP answers undeclared
    inputs = {"cap-open-pdf": AcquisitionInputs(system_values={"asset_id": cid})}
    out, req, rec, _ = await _run(
        monkeypatch, caplog, entry=entry,
        ctx=_ctx("打开这个文件", open_asset_id=cid), inputs=inputs)
    assert out is req and _fallback(caplog) == REASON_ACQUISITION_UNDECLARED
    assert not rec.legacy


async def test_no_provider_at_all_is_undeclared(monkeypatch, caplog, real_pdf):
    # production default: acquisition_inputs absent -> undeclared -> Agent
    cid, _ = real_pdf
    entry = _entry("cap-open-pdf", {
        "asset_id": {"type": "string", "required": True, "description": "the PDF"},
    })
    out, req, rec, _ = await _run(
        monkeypatch, caplog, entry=entry,
        ctx=_ctx("打开这个文件", open_asset_id=cid), inputs=None)
    assert out is req and _fallback(caplog) == REASON_ACQUISITION_UNDECLARED
    assert not rec.legacy


# ── D. MISSING: required MODEL slot with no legal source -> Agent (§D) ───────────


async def test_required_model_slot_without_evidence_is_missing(monkeypatch, caplog,
                                                              real_pdf):
    cid, _ = real_pdf
    entry = _entry("cap-open-pdf", {
        "pages": {"type": "string", "required": True, "description": "page range"},
    })
    inputs = {"cap-open-pdf": AcquisitionInputs(
        declaration={"pages": SlotDecl(OWNERSHIP_MODEL,
                                       allowed_sources=(SOURCE_QUERY,))},
    )}
    out, req, rec, _ = await _run(
        monkeypatch, caplog, entry=entry,
        ctx=_ctx("打开这个文件", open_asset_id=cid), inputs=inputs)
    assert out is req and _fallback(caplog) == REASON_ACQUISITION_MISSING
    assert not rec.legacy


# ── E. MODEL acquisition need -> Agent (Qwen not wired this step) ────────────────


async def test_query_evidence_model_need_is_not_yet_executable(monkeypatch, caplog,
                                                               real_pdf):
    cid, _ = real_pdf
    entry = _entry("cap-open-pdf", {
        "pages": {"type": "string", "required": True, "description": "page range"},
    })
    # query evidence present -> QUERY_TO_QWEN (a legal MODEL strategy) — but the
    # Qwen extractor is not wired in Step 2, so the turn exits to the Agent.
    inputs = {"cap-open-pdf": AcquisitionInputs(
        declaration={"pages": SlotDecl(
            OWNERSHIP_MODEL,
            allowed_sources=(SOURCE_QUERY, SOURCE_CONVERSATION_5_USER_TURNS))},
        evidence={"pages": SOURCE_QUERY},
    )}
    out, req, rec, _ = await _run(
        monkeypatch, caplog, entry=entry,
        ctx=_ctx("打开这个文件", open_asset_id=cid), inputs=inputs)
    assert out is req and _fallback(caplog) == REASON_ACQUISITION_MODEL_PENDING
    assert not rec.legacy            # no Qwen, no legacy extraction


async def test_history_evidence_model_need_is_not_yet_executable(monkeypatch, caplog,
                                                                 real_pdf):
    cid, _ = real_pdf
    entry = _entry("cap-open-pdf", {
        "pages": {"type": "string", "required": True, "description": "page range"},
    })
    inputs = {"cap-open-pdf": AcquisitionInputs(
        declaration={"pages": SlotDecl(
            OWNERSHIP_MODEL,
            allowed_sources=(SOURCE_QUERY, SOURCE_CONVERSATION_5_USER_TURNS))},
        evidence={"pages": SOURCE_CONVERSATION_5_USER_TURNS},
    )}
    out, req, rec, _ = await _run(
        monkeypatch, caplog, entry=entry,
        ctx=_ctx("打开这个文件", open_asset_id=cid), inputs=inputs)
    assert out is req and _fallback(caplog) == REASON_ACQUISITION_MODEL_PENDING
    assert not rec.legacy


# ── HIT invariant: HIT -> ARP directly, never re-selected (§A.10 / §I) ───────────


async def test_hit_reaches_arp_without_reselection(monkeypatch, caplog, real_pdf):
    cid, _ = real_pdf
    entry = _entry("cap-open-pdf", {
        "pages": {"type": "string", "required": True, "description": "page range"},
    })
    inputs = {"cap-open-pdf": AcquisitionInputs(
        declaration={"pages": SlotDecl(
            OWNERSHIP_MODEL, allowed_sources=(SOURCE_QUERY,))},
        evidence={"pages": SOURCE_QUERY},
    )}
    out, req, rec, _ = await _run(
        monkeypatch, caplog, entry=entry,
        ctx=_ctx("打开这个文件", open_asset_id=cid), inputs=inputs)
    # reaching an ACQUISITION reason proves the ARP ran; the legacy hop never did.
    assert _fallback(caplog) == REASON_ACQUISITION_MODEL_PENDING
    assert not rec.legacy
    assert out is req
