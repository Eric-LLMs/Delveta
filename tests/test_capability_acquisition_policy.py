"""Per-capability acquisition policy — the explicit strategy dispatch.

These tests pin the FIVE business states the orchestrator must keep apart:

* ``NO_ARGS`` — the capability genuinely takes no arguments and certifies ``{}``;
* ``RULE_CONTEXT`` — a handler resolves every slot from trusted context / a
  deterministic rule; the shared extractor is never called (the file
  capabilities may use their own page-scope resolver, never identity);
* ``MODEL`` — a handler authorizes the extractor for its MODEL-owned slots only,
  and every other draft value stays protected;
* ``MIXED`` — deterministic slots are protected while the model fills only the
  authorized remainder; a ``required_model_slots`` slot that does not come back
  exits to the Agent instead of being silently dropped;
* ``DEFERRED_AGENT`` — admitted, but the real tool contract cannot be satisfied
  from any legal producer on this path; the turn exits with its OWN reason.

They also pin the failure taxonomy: a missing policy, a missing handler, a
NO_ARGS declaration that contradicts the schema, and a MODEL declaration whose
handler cannot plan are FOUR distinct exits — never one generic "undeclared".
"""
from __future__ import annotations

import types

import pytest
from core.application.chat.intent_funnel import orchestrator as orch_mod
from core.application.chat.intent_funnel.argument_acquisition.extractor import (
    ExtractionUnavailable,
)
from core.application.chat.intent_funnel.argument_acquisition.page_scope_resolver import (
    STATUS_AMBIGUOUS,
    TARGET_UNKNOWN,
    PageScope,
)
from core.application.chat.intent_funnel.argument_acquisition.slot_refine import (
    accept_model_values,
)
from core.application.chat.intent_funnel.cap_handler import (
    CATEGORY_EXTRACTION_FAILED,
    POLICIES,
    STAGE_ARGUMENT_EXTRACTION,
    STRATEGY_DEFERRED_AGENT,
    STRATEGY_MIXED,
    STRATEGY_MODEL,
    STRATEGY_NO_ARGS,
    STRATEGY_RULE_CONTEXT,
    AcquisitionPolicy,
    SlotPlan,
    policy_for,
    roster,
)
from core.application.chat.intent_funnel.contract import (
    REASON_ACQUISITION_DEFERRED,
    REASON_ACQUISITION_HANDLER_MISSING,
    REASON_ACQUISITION_MISSING,
    REASON_ACQUISITION_MODEL_PENDING,
    REASON_ACQUISITION_POLICY_CONFLICT,
    REASON_ACQUISITION_UNDECLARED,
    REASON_BIND_INVALID,
    REASON_BIND_MISSING,
    TurnFacts,
)
from core.application.chat.intent_funnel.registry import content_fingerprint
from core.application.chat.intent_funnel.registry.entry import (
    CapabilityEntry,
    QueryRecord,
    RegistryLiveView,
    derive_language,
)
from core.application.chat.understanding import TurnRequirements

# The formally admitted set (the four hidden config capabilities are excluded).
ADMITTED = frozenset({
    "cap-add-term",
    "cap-create-folder",
    "cap-rag-search",
    "cap-web-search",
    "cap-social-search",
    "cap-vision",
    "cap-read-document",
    "cap-read-file",
    "cap-pdf-extract-text",
    "cap-pdf-table-to-text",
    "cap-translate",
    "cap-mindmap",
    "cap-slides",
    "cap-summary",
})


def _entry(cid: str, tool: str, params: dict) -> CapabilityEntry:
    q = "做点事"
    return CapabilityEntry(
        capability_id=cid, tool_binding=tool, description="d",
        standard_queries=(QueryRecord(id=f"{cid}-q", query=q,
                                      language=derive_language(q)),),
        parameters=params,
    )


def _view(entry: CapabilityEntry) -> RegistryLiveView:
    return RegistryLiveView(fingerprint=content_fingerprint([entry]),
                            entries=(entry,))


async def _hop(entry: CapabilityEntry, *, query: str, extractor=None,
               facts=None, history=(), resolver=None):
    trace: dict = {}
    capture: dict = {}
    deps = types.SimpleNamespace(argument_extractor=extractor,
                                 page_scope_resolver=resolver)
    out = await orch_mod._acquisition_hop(
        TurnRequirements(), deps, entry, query=query, facts=facts,
        view=_view(entry), trace=trace, capture=capture, history=history)
    return out, trace, capture


def _extractor(returned, *, seen=None):
    async def _extract(*, query, entry, model_slots, bundle, prompt):
        if seen is not None:
            seen.update(query=query, slots=tuple(model_slots))
        return dict(returned), bundle.source
    return _extract


def _boom(**kw):
    raise AssertionError("the model must not be called on this path")


# ── the declaration ─────────────────────────────────────────────────────────────


def test_every_admitted_capability_declares_exactly_one_policy():
    assert set(POLICIES) == ADMITTED
    assert all(p.strategy for p in POLICIES.values())


def test_hidden_capabilities_have_no_acquisition_policy():
    # These are excluded from the funnel by the capability-admission predicate;
    # a policy here would make them look executable on this path.
    for cid in ("cap-artifact", "cap-bash", "cap-edit-file", "cap-research"):
        assert policy_for(cid) is None


def test_toolkit_pair_is_deferred_and_documents_the_real_contract_gap():
    # ``cap-summary`` is NOT here: its tool now publishes the string slots the
    # funnel can produce, so a handler owns it (see the MIXED tests).
    for cid in ("cap-mindmap", "cap-slides"):
        policy = POLICIES[cid]
        assert policy.strategy == STRATEGY_DEFERRED_AGENT
        assert "paths" in policy.detail
        assert "string slots only" in policy.detail


# ── NO_ARGS: genuine no-args is certified, never inferred from ``{}`` ───────────


async def test_declared_no_args_certifies_an_empty_argument_set(monkeypatch):
    monkeypatch.setitem(POLICIES, "cap-no-args",
                        AcquisitionPolicy(STRATEGY_NO_ARGS))
    entry = _entry("cap-no-args", "noop", {})
    out, trace, capture = await _hop(entry, query="do it", extractor=_boom)

    assert trace["acquisition_policy"] == STRATEGY_NO_ARGS
    assert trace["acquisition"] == "NO_ARGS"
    assert capture["acquisition"] == {
        "declared": True, "strategy": STRATEGY_NO_ARGS, "args": {}}
    assert out.requested_action["args"] == {}


async def test_no_args_cannot_bypass_a_published_parameter_schema(monkeypatch):
    monkeypatch.setitem(POLICIES, "cap-no-args-bad",
                        AcquisitionPolicy(STRATEGY_NO_ARGS))
    entry = _entry("cap-no-args-bad", "noop", {
        "required_input": {"type": "string", "required": True,
                           "description": "must be acquired"},
    })
    out, trace, _ = await _hop(entry, query="do it", extractor=_boom)

    assert out is None
    assert trace["fallback"] == REASON_ACQUISITION_POLICY_CONFLICT


# ── RULE_CONTEXT: deterministic acquisition never calls the model ───────────────


async def test_rule_context_capability_never_calls_the_extractor():
    entry = _entry("cap-read-file", "read_file", {
        "path": {"type": "string", "required": True, "description": "path"},
    })
    out, trace, capture = await _hop(
        entry, query="读取 ./data/train.csv", extractor=_boom, facts=None)

    assert trace["acquisition_policy"] == STRATEGY_RULE_CONTEXT
    assert out.requested_action["args"] == {"path": "./data/train.csv"}
    assert "extractor" not in capture["acquisition"]


async def test_missing_context_is_never_fabricated():
    # No viewer / attachment fact exists, so the handler abstains and the turn
    # exits MISSING; no asset id is invented.
    entry = _entry("cap-read-document", "read_document", {
        "asset_id": {"type": "string", "required": True,
                     "description": "from turn facts"},
    })
    out, trace, _ = await _hop(entry, query="总结这个文档", extractor=_boom,
                              facts=None)

    assert out is None
    assert trace["fallback"] == REASON_ACQUISITION_MISSING


# ── MODEL: only authorized slots; existing deterministic values are protected ───


async def test_model_may_replace_a_declared_fallback_but_not_a_protected_value(
        monkeypatch):
    class _Handler:
        capability_id = "cap-web-search"

        async def acquire(self, *, query, facts):
            return {"query": "raw sentence", "platform": "reddit"}

        def slot_plan(self, *, query, facts, draft):
            return SlotPlan(model_slots=("query", "platform"),
                            fallback_slots=("query",))

    monkeypatch.setitem(roster.HANDLERS, "cap-web-search", _Handler())
    entry = _entry("cap-web-search", "web_search", {
        "query": {"type": "string", "required": True, "description": "q"},
        "platform": {"type": "string", "required": False, "description": "p"},
    })
    out, trace, capture = await _hop(
        entry, query="raw sentence",
        extractor=_extractor({"query": "clean topic", "platform": "x"}))

    assert trace["acquisition_policy"] == STRATEGY_MODEL
    # query is a declared fallback -> replaced; platform is deterministic -> kept.
    assert out.requested_action["args"] == {"query": "clean topic",
                                            "platform": "reddit"}
    assert capture["acquisition"]["slot_plan"] == {
        "model_slots": ["query", "platform"],
        "fallback_slots": ["query"],
        "default_slots": [],
    }


def test_protected_draft_values_survive_a_model_value_without_a_fallback_decl():
    out = accept_model_values(
        {"query": "raw", "asset_id": "asset-1"},
        {"query": "clean", "asset_id": "hallucinated"},
        model_slots=("query", "asset_id"),
        parameters={"query": {"type": "string", "required": True},
                    "asset_id": {"type": "string", "required": True}},
    )
    assert out == {"query": "raw", "asset_id": "asset-1"}


def test_a_blank_draft_value_is_not_a_protected_value():
    out = accept_model_values(
        {"query": "   "},
        {"query": "clean"},
        model_slots=("query",),
        parameters={"query": {"type": "string", "required": True}},
    )
    assert out == {"query": "clean"}


def test_model_values_outside_the_authorized_slots_are_never_written():
    out = accept_model_values(
        {"query": "raw"},
        {"query": "clean", "top_k": "99"},
        model_slots=("query",),
        parameters={"query": {"type": "string", "required": True},
                    "top_k": {"type": "integer", "required": False}},
        fallback_slots=("query",),
    )
    assert out == {"query": "clean"}


async def test_model_policy_result_still_passes_the_same_binder():
    entry = _entry("cap-web-search", "web_search", {
        "query": {"type": "string", "required": True, "max_len": 10,
                  "description": "q"},
    })
    out, trace, _ = await _hop(
        entry, query="raw sentence",
        extractor=_extractor({"query": "x" * 11}))

    assert out is None
    assert trace["fallback"] == REASON_BIND_INVALID


async def test_model_policy_that_produces_nothing_falls_to_the_binder_gate():
    entry = _entry("cap-add-term", "add_term", {
        "term": {"type": "string", "required": True, "description": "word"},
        "domain": {"type": "string", "required": True, "description": "domain"},
    })
    out, trace, _ = await _hop(entry, query="加个词", extractor=_extractor({}))

    assert out is None
    assert trace["fallback"] == REASON_BIND_MISSING


# ── DEFERRED_AGENT: a known contract gap has its own exit ──────────────────────


# ── MIXED (cap-summary): deterministic source + a REQUIRED model slot ──────────

_SUMMARY_PARAMS = {
    "paths": {"type": "array", "required": False, "description": "files"},
    "path": {"type": "string", "required": False, "description": "one file"},
    "asset_id": {"type": "string", "required": False, "description": "doc"},
    "pages": {"type": "string", "required": False, "description": "pages"},
    "context": {"type": "string", "required": False, "description": "chat"},
    "output_dir": {"type": "string", "required": False, "description": "dir"},
}


def _summary_entry() -> CapabilityEntry:
    return _entry("cap-summary", "summary_gen", dict(_SUMMARY_PARAMS))


async def test_summary_mixed_acquires_a_literal_workspace_path_without_a_model():
    # cap-summary is handler-owned: a literal path in the sentence is the source.
    # With NO save cue the plan authorizes nothing, so the extractor is never
    # called — and therefore no output_dir can be invented.
    entry = _summary_entry()
    out, trace, capture = await _hop(
        entry, query="总结 ./reports/q3.md", extractor=_boom, facts=None)

    assert trace["acquisition_policy"] == STRATEGY_MIXED
    assert out.requested_action["args"] == {"path": "./reports/q3.md"}
    assert "extractor" not in capture["acquisition"]
    assert "slot_plan" not in capture["acquisition"]


async def test_summary_save_cue_folds_the_extracted_output_dir():
    # The user explicitly asked to save; the shared extractor supplies the
    # directory and the deterministic content source is kept.
    entry = _summary_entry()
    seen: dict = {}
    out, trace, capture = await _hop(
        entry, query="总结 ./reports/q3.md 并保存到 outputs", facts=None,
        extractor=_extractor({"output_dir": "outputs"}, seen=seen))

    assert trace["acquisition_policy"] == STRATEGY_MIXED
    assert out.requested_action["args"] == {"path": "./reports/q3.md",
                                            "output_dir": "outputs"}
    assert seen["slots"] == ("output_dir",)
    assert capture["acquisition"]["slot_plan"]["required_model_slots"] == [
        "output_dir"]


async def test_summary_extractor_unavailable_hands_the_turn_to_the_agent():
    # The ONLINE extraction failed. The requested output mode cannot be honoured,
    # so the turn exits with the structured record instead of silently returning
    # the summary in chat.
    entry = _summary_entry()

    async def _down(**kw):
        raise ExtractionUnavailable("endpoint timed out")

    out, trace, capture = await _hop(
        entry, query="总结 ./reports/q3.md 并保存到 outputs", facts=None,
        extractor=_down)

    assert out is None
    assert trace["fallback"] == REASON_ACQUISITION_MODEL_PENDING
    record = capture["acquisition"]["failure"]
    assert record["category"] == CATEGORY_EXTRACTION_FAILED
    assert record["stage"] == STAGE_ARGUMENT_EXTRACTION
    assert "endpoint timed out" in record["detail"]
    assert "output_dir" in record["detail"]


async def test_summary_extraction_without_the_slot_hands_the_turn_to_the_agent():
    # The call succeeded but returned no usable directory: same rule, no
    # chat-only downgrade.
    entry = _summary_entry()
    out, trace, capture = await _hop(
        entry, query="总结 ./reports/q3.md 并保存到 outputs", facts=None,
        extractor=_extractor({"output_dir": "   "}))

    assert out is None
    assert trace["fallback"] == REASON_ACQUISITION_MODEL_PENDING
    assert capture["acquisition"]["failure"]["category"] == CATEGORY_EXTRACTION_FAILED


async def test_summary_required_slot_without_an_extractor_hands_the_turn_to_the_agent():
    entry = _summary_entry()
    out, trace, capture = await _hop(
        entry, query="总结 ./reports/q3.md 并保存到 outputs", facts=None,
        extractor=None)

    assert out is None
    assert trace["fallback"] == REASON_ACQUISITION_MODEL_PENDING
    record = capture["acquisition"]["failure"]
    assert record["category"] == CATEGORY_EXTRACTION_FAILED
    assert "no shared argument extractor is wired" in record["detail"]


async def test_summary_ambiguous_intent_with_a_viewer_page_uses_that_page():
    # Priority 2: the LLM answered normally but did not settle the intent. The
    # viewer carries an asset id AND a current page, so exactly that page of
    # that document is summarized — not the whole document, not a previous reply.
    entry = _summary_entry()

    async def _ambiguous(**kw):
        return PageScope(target=TARGET_UNKNOWN, page_scope="unknown",
                         status=STATUS_AMBIGUOUS, reason="target not determined")

    out, trace, capture = await _hop(
        entry, query="总结一下", extractor=_boom,
        facts=TurnFacts(viewer_asset_id="asset-9", viewer_current_page=4),
        resolver=_ambiguous,
        history=[{"role": "assistant", "content": "之前说过 RRF"}])

    assert out.requested_action["args"] == {"asset_id": "asset-9", "pages": "4"}
    assert trace["acquisition_policy"] == STRATEGY_MIXED
    assert capture["page_scope"]["status"] == STATUS_AMBIGUOUS


async def test_summary_ambiguous_intent_without_a_page_uses_the_last_reply():
    entry = _entry("cap-summary", "summary_gen", {
        "context": {"type": "string", "required": False, "description": "chat"},
    })

    async def _ambiguous(**kw):
        return PageScope(target=TARGET_UNKNOWN, page_scope="unknown",
                         status=STATUS_AMBIGUOUS)

    out, _trace, _capture = await _hop(
        entry, query="总结一下", extractor=_boom,
        facts=TurnFacts(viewer_asset_id="asset-9"),
        resolver=_ambiguous,
        history=[{"role": "assistant", "content": "之前说过 RRF"}])

    assert out.requested_action["args"] == {"context": "之前说过 RRF"}


async def test_summary_without_a_resolver_still_receives_the_chat_history():
    # The history is not part of the page-scope seam: cap-summary's last-reply
    # fallback must work even when the host wired no scope model.
    entry = _entry("cap-summary", "summary_gen", {
        "context": {"type": "string", "required": False, "description": "chat"},
    })
    out, _trace, _capture = await _hop(
        entry, query="总结一下", extractor=_boom,
        history=[{"role": "assistant", "content": "之前说过 RRF"}])

    assert out.requested_action["args"] == {"context": "之前说过 RRF"}


@pytest.mark.parametrize("cid,tool", [
    ("cap-mindmap", "mindmap_gen"),
    ("cap-slides", "slides_gen"),
])
async def test_deferred_capability_exits_with_its_own_reason(cid, tool):
    entry = _entry(cid, tool, {})       # the live Registry really declares {}
    out, trace, capture = await _hop(entry, query="总结这个文件",
                                     extractor=_boom)

    assert out is None
    assert trace["fallback"] == REASON_ACQUISITION_DEFERRED
    assert capture["acquisition_policy"]["strategy"] == STRATEGY_DEFERRED_AGENT
    assert "paths" in capture["acquisition_policy"]["detail"]


# ── the failure taxonomy: four different wiring states, four different exits ────


async def test_policy_without_a_handler_is_handler_missing(monkeypatch):
    monkeypatch.setitem(POLICIES, "cap-orphan-model",
                        AcquisitionPolicy(STRATEGY_MODEL))
    entry = _entry("cap-orphan-model", "orphan", {
        "x": {"type": "string", "required": True, "description": "x"},
    })
    out, trace, _ = await _hop(entry, query="do it", extractor=_boom)

    assert out is None
    assert trace["fallback"] == REASON_ACQUISITION_HANDLER_MISSING


async def test_a_handler_without_a_policy_is_a_conflict(monkeypatch):
    class _Handler:
        capability_id = "cap-unlisted-handler"

        async def acquire(self, *, query, facts):
            return {}

    monkeypatch.setitem(roster.HANDLERS, "cap-unlisted-handler", _Handler())
    entry = _entry("cap-unlisted-handler", "unlisted", {})
    out, trace, _ = await _hop(entry, query="do it", extractor=_boom)

    assert out is None
    assert trace["fallback"] == REASON_ACQUISITION_POLICY_CONFLICT


async def test_capability_with_neither_policy_nor_declaration_is_undeclared():
    # A capability the policy table does not cover and whose per-turn provider
    # declares nothing is NOT implicitly "no args": it is policy-undeclared.
    entry = _entry("cap-unlisted", "unlisted", {})
    out, trace, _ = await _hop(entry, query="do it", extractor=None)

    assert out is None
    assert trace["fallback"] == REASON_ACQUISITION_UNDECLARED
