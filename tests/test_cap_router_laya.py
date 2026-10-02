"""cap_router laya backend — LayaSelector + the B_noprov card renderer.

LayaChoice is served OUT OF PROCESS by the ``deploy/laya`` sidecar. This module
pins the client side of that seam:

* the card renderer reproduces the FROZEN ``B_noprov`` bytes byte-for-byte (the
  golden anchor is the shipped test bundle), and carries NO query examples,
  NO provenance/``evidence:`` line, NO ``matched_example:`` line;
* a valid choice becomes ``ROUTE_SELECTED`` with the model's probability;
* EVERY other outcome — no endpoint, timeout, transport error, HTTP status,
  malformed body, missing answer, non-choice answer, off-candidate label —
  raises ``CapabilityRouterUnavailable`` (never a silent NONE, never Qwen).
"""
from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import httpx
import pytest
from core.application.chat.intent_funnel.cap_router import (
    BACKEND_LAYA,
    ROUTE_SELECTED,
    CapabilityRouterUnavailable,
)
from core.application.chat.intent_funnel.cap_router.backends.laya import (
    QID,
    LayaSelector,
)
from core.application.chat.intent_funnel.cap_router.card_renderer import (
    INSTRUCTIONS,
    render_card,
    render_choice_question,
)
from core.application.chat.intent_funnel.contract import Candidate
from core.application.chat.intent_funnel.registry.entry import CapabilityEntry
from core.config import settings

URL = "http://localhost:18092"
FROZEN = (Path(__file__).resolve().parents[1]
          / "scripts" / "laya_finetune" / "V1" / "data"
          / "layachoice_v1_test_b_noprov.jsonl.gz")


def _entry(cid="cap-a", *, tool="create_folder", desc="does a",
           negatives=(), params=None):
    return CapabilityEntry(capability_id=cid, tool_binding=tool, description=desc,
                           negatives=tuple(negatives),
                           parameters=dict(params or {}))


def _c(cid, score=0.9):
    return Candidate(cid, score, origin="recall")


# ── httpx doubles in the test_funnel_p2 shape ────────────────────────────────────


class _Resp:
    def __init__(self, status, body=None, *, json_raises=False):
        self.status_code = status
        self._body = body
        self._json_raises = json_raises

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "boom", request=httpx.Request("POST", URL),
                response=httpx.Response(self.status_code))

    def json(self):
        if self._json_raises:
            raise ValueError("not json")
        return self._body


class _Client:
    def __init__(self, resp=None, *, exc=None):
        self._resp = resp
        self._exc = exc
        self.seen: dict = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None):
        self.seen["url"] = url
        self.seen["payload"] = json
        if self._exc is not None:
            raise self._exc
        return self._resp


def _ok_body(choice, probs=None):
    return {"model": "laya-rl-agent",
            "answers": {QID: {"type": "choice", "choice": choice,
                              "probabilities": probs if probs is not None
                              else {choice: 0.81}}},
            "usage": {}}


async def _select(monkeypatch, client=None, *, candidates=None, entries=None,
                  url=URL):
    from core.application.chat.intent_funnel.cap_router.backends import laya as laya_mod
    monkeypatch.setattr(settings, "chat_cap_router_laya_url", url)
    if client is not None:
        monkeypatch.setattr(laya_mod.httpx, "AsyncClient", lambda **kw: client)
    return await LayaSelector().select(
        "q", candidates if candidates is not None else [_c("cap-a")],
        entries_by_id=entries if entries is not None else {})


# ── the factory resolves the real backend ────────────────────────────────────────


def test_factory_resolves_laya():
    from core.application.chat.intent_funnel.cap_router.backends import selector_for
    assert isinstance(selector_for(BACKEND_LAYA), LayaSelector)


# ── failure contract: everything unusable -> CapabilityRouterUnavailable ─────────


async def test_no_endpoint_is_unavailable(monkeypatch):
    with pytest.raises(CapabilityRouterUnavailable):
        await _select(monkeypatch, url="")


async def test_timeout_is_unavailable(monkeypatch):
    client = _Client(exc=httpx.TimeoutException("slow"))
    with pytest.raises(CapabilityRouterUnavailable):
        await _select(monkeypatch, client)


async def test_transport_error_is_unavailable(monkeypatch):
    client = _Client(exc=httpx.ConnectError("refused"))
    with pytest.raises(CapabilityRouterUnavailable):
        await _select(monkeypatch, client)


async def test_http_status_is_unavailable(monkeypatch):
    with pytest.raises(CapabilityRouterUnavailable):
        await _select(monkeypatch, _Client(_Resp(500, {})))


async def test_non_json_body_is_unavailable(monkeypatch):
    with pytest.raises(CapabilityRouterUnavailable):
        await _select(monkeypatch, _Client(_Resp(200, json_raises=True)))


async def test_missing_answers_is_unavailable(monkeypatch):
    with pytest.raises(CapabilityRouterUnavailable):
        await _select(monkeypatch, _Client(_Resp(200, {"model": "x"})))


async def test_missing_answer_key_is_unavailable(monkeypatch):
    with pytest.raises(CapabilityRouterUnavailable):
        await _select(monkeypatch, _Client(_Resp(200, {"answers": {}})))


async def test_non_choice_answer_is_unavailable(monkeypatch):
    body = {"answers": {QID: {"type": "text", "text": "cap-a"}}}
    with pytest.raises(CapabilityRouterUnavailable):
        await _select(monkeypatch, _Client(_Resp(200, body)))


async def test_off_candidate_choice_is_unavailable(monkeypatch):
    body = _ok_body("cap-zzz")
    with pytest.raises(CapabilityRouterUnavailable):
        await _select(monkeypatch, _Client(_Resp(200, body)),
                      candidates=[_c("cap-a")])


async def test_non_string_choice_is_unavailable(monkeypatch):
    body = {"answers": {QID: {"type": "choice", "choice": 7}}}
    with pytest.raises(CapabilityRouterUnavailable):
        await _select(monkeypatch, _Client(_Resp(200, body)))


# ── the happy path ───────────────────────────────────────────────────────────────


async def test_valid_choice_selects_with_the_model_probability(monkeypatch):
    entry = _entry("cap-a")
    client = _Client(_Resp(200, _ok_body("cap-a", {"cap-a": 0.77, "cap-b": 0.23})))
    route = await _select(monkeypatch, client,
                          candidates=[_c("cap-a"), _c("cap-b")],
                          entries={"cap-a": entry})
    assert route.decision == ROUTE_SELECTED and route.selected
    assert route.capability_id == "cap-a"
    assert route.confidence == 0.77
    # wire shape: state carries the query, one choice question under QID
    assert client.seen["url"] == URL + "/v1/systemone"
    q = client.seen["payload"]["questions"][QID]
    assert q["type"] == "choice" and q["instructions"] == INSTRUCTIONS
    assert set(q["criteria"]) == {"cap-a"}      # only the candidate with a card
    assert client.seen["payload"]["state"] == "q"


async def test_trailing_slash_on_the_base_url_is_tolerated(monkeypatch):
    client = _Client(_Resp(200, _ok_body("cap-a")))
    await _select(monkeypatch, client, url=URL + "/")
    assert client.seen["url"] == URL + "/v1/systemone"


async def test_missing_probabilities_yields_no_fabricated_confidence(monkeypatch):
    body = {"answers": {QID: {"type": "choice", "choice": "cap-a"}}}
    route = await _select(monkeypatch, _Client(_Resp(200, body)))
    assert route.selected and route.capability_id == "cap-a"
    assert route.confidence is None


# ── the card renderer: golden vs the frozen B_noprov bundle ───────────────────────


def _entry_from_option(opt: str) -> CapabilityEntry:
    """Reverse-parse one frozen bundle option back into the Registry entry it was
    rendered from, so ``render_card`` can be checked byte-for-byte."""
    head, rest = opt.split("\nnegative examples:\n", 1)
    neg_block, params_block = rest.split("\nparams:", 1)
    lines = head.split("\n")
    cid = lines[0][len("### "):]
    tool = lines[1][len("tool: "):]
    desc = "\n".join(lines[2:])[len("does: "):]
    negs = ([] if neg_block.strip() == "- (none curated)"
            else [ln[2:] for ln in neg_block.split("\n") if ln.startswith("- ")])
    params: dict = {}
    if not params_block.startswith(" none"):
        for ln in params_block.split("\n"):
            ln = ln.strip()
            if not ln.startswith("- "):
                continue
            m = re.match(r"- (\S+) \(([^)]*)\)(?:: (.*))?$", ln)
            bits = [b.strip() for b in m.group(2).split(",")]
            spec = {"type": bits[0], "required": "required" in bits}
            for b in bits:
                if b.startswith("max_len="):
                    spec["max_len"] = int(b.split("=", 1)[1])
            if m.group(3):
                spec["description"] = m.group(3)
            params[m.group(1)] = spec
    return CapabilityEntry(capability_id=cid, tool_binding=tool, description=desc,
                           negatives=tuple(negs), parameters=params)


@pytest.mark.skipif(not FROZEN.exists(), reason="frozen bundle not present")
def test_render_card_is_byte_identical_to_the_frozen_bundle():
    with gzip.open(FROZEN, "rt", encoding="utf-8") as fh:
        row = json.loads(fh.readline())
    assert row["options"], "frozen row carries no options"
    for opt in row["options"]:
        assert render_card(_entry_from_option(opt)) == opt


def test_card_carries_no_query_examples_provenance_or_matched_example():
    card = render_card(_entry("cap-a", desc="does a", negatives=["not b"],
                              params={"name": {"type": "string",
                                               "required": True,
                                               "max_len": 120,
                                               "description": "the name"}}))
    assert "query examples:" not in card
    assert "evidence:" not in card
    assert "matched_example:" not in card
    assert "recall:" not in card
    assert card.startswith("### cap-a\ntool: create_folder\ndoes: does a\n")
    assert "- not b" in card
    assert "- name (string, required, max_len=120): the name" in card


def test_card_with_no_negatives_or_params_uses_the_none_lines():
    card = render_card(_entry("cap-a"))
    assert "negative examples:\n- (none curated)" in card
    assert card.endswith("params: none (arguments must be {})")


def test_choice_question_skips_candidates_without_a_live_entry():
    entries = {"cap-a": _entry("cap-a")}
    q = render_choice_question([_c("cap-a"), _c("cap-gone")], entries)
    assert set(q["criteria"]) == {"cap-a"}
    assert q["criteria"]["cap-a"] == render_card(entries["cap-a"])
