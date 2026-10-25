"""B1 — online extractor ↔ shared SFT contract (native function calling).

The extractor now speaks NATIVE function calling: the request sends ONE tool
(``name == capability_id``, ``parameters`` = ONLY the authorized model-owned slots
in slot-plan order) with ``tool_choice="required"``, and the reply is read from
``message.tool_calls``. This suite is the protocol gate — it does NOT run a model
(no endpoint), it pins the CONTRACT:

* the six capabilities' function schema, slot ORDER and required-set are the SAME
  contract the training builder applies (``pe_contract``), and that order is tied
  to the production handler's ``slot_plan()`` (one source of truth);
* a native call whose name differs from the pinned capability, or whose
  ``arguments`` is not a JSON object, is a hard reject — and a valid content reply
  can NEVER mask such a native error;
* the content fallback is used ONLY when there is no native call;
* native values still pass the SAME authorization gate (``accept_model_values``):
  an unauthorized slot never reaches the draft, and a missing required slot is not
  fabricated (the Binder remains the final gate).
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from core.application.chat.intent_funnel.argument_acquisition import context_bundle
from core.application.chat.intent_funnel.argument_acquisition.extractor import (
    ExtractionUnavailable,
    _function_schema,
    _parse_tool_call,
    _payload,
    _tools_for,
)
from core.application.chat.intent_funnel.argument_acquisition.slot_refine import (
    accept_model_values,
)
from core.application.chat.intent_funnel.cap_handler import handler_for

# ``pe_contract`` is the stdlib-only shared contract under fine_tuning/ (no
# production import, so the standalone builder/verifier and this test read the
# SAME snapshot).
_SCRIPTS = (Path(__file__).resolve().parents[1] / "fine_tuning"
            / "qwen_parameter_extraction_finetune")
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import pe_contract as C  # noqa: E402

CAPS = ["cap-web-search", "cap-rag-search", "cap-social-search",
        "cap-translate", "cap-add-term", "cap-create-folder"]

# The capabilities the ONLINE funnel still authorizes the shared extractor for.
# ``cap-translate`` stays in the frozen training contract (the Formal-500 rows are
# historical data) but its production lane is RULE_CONTEXT: the sentence rides
# verbatim and the translate tool's own call decides the payload.
LIVE_EXTRACTOR_CAPS = [c for c in CAPS if c != "cap-translate"]


def _entry(cid: str):
    """A Registry-shaped entry built from the shared contract's param table."""
    return types.SimpleNamespace(
        capability_id=cid, tool_binding=C.tool_name(cid),
        description=C.function_description(cid),
        parameters=C.SCHEMA[cid]["params"],
    )


# ── one source of truth: pe_contract order == the handler's slot_plan order ───────


@pytest.mark.parametrize("cid", LIVE_EXTRACTOR_CAPS)
def test_contract_slot_order_matches_the_production_handler(cid):
    # The online order comes from the handler; the training contract mirrors it.
    h = handler_for(cid)
    assert h is not None, f"no handler registered for {cid}"
    plan = h.slot_plan(query="x", facts=None, draft={"query": "x"})
    assert C.model_slots(cid) == list(plan.model_slots)


# ── the function face == the SFT contract (schema + order + required set) ─────────


@pytest.mark.parametrize("cid", CAPS)
def test_function_schema_is_the_shared_contract(cid):
    model_slots = C.model_slots(cid)
    schema = _function_schema(_entry(cid), model_slots)
    assert schema == C.model_schema(cid)                 # identical face
    assert list(schema["properties"]) == model_slots     # order preserved
    # MODEL-vs-EXECUTION separation: the MODEL face never forces `required` (the
    # model may omit a required slot it cannot honestly extract); the EXECUTION
    # side keeps the registry's required=True and the Binder clarifies. The
    # prompt's slot block still flags required/optional as guidance.
    assert schema["required"] == []
    assert C.required_model_slots(cid)                   # execution side unchanged


@pytest.mark.parametrize("cid", CAPS)
def test_tools_for_pins_one_function_named_for_the_capability(cid):
    tools = _tools_for(_entry(cid), C.model_slots(cid))
    assert len(tools) == 1
    fn = tools[0]["function"]
    assert fn["name"] == cid                             # == capability_id
    assert fn["parameters"] == C.model_schema(cid)


@pytest.mark.parametrize("cid", CAPS)
def test_payload_sends_tools_and_requires_a_call(cid):
    bundle = context_bundle.build("q")
    payload = _payload(_entry(cid), C.model_slots(cid), bundle)
    assert payload["tool_choice"] == "required"
    assert len(payload["tools"]) == 1
    assert payload["tools"][0]["function"]["name"] == cid


# ── native tool_call parsing: strict, fail-closed ────────────────────────────────


def _call(name, arguments):
    return {"tool_calls": [{"function": {"name": name, "arguments": arguments}}]}


def test_native_call_parses_to_a_flat_slot_map():
    msg = _call("cap-web-search", '{"query": "clean topic", "top_k": 3}')
    assert _parse_tool_call(msg, _entry("cap-web-search"), ["query", "top_k"]) == {
        "query": "clean topic", "top_k": 3}


def test_native_call_object_arguments_also_accepted():
    msg = _call("cap-web-search", {"query": "clean"})
    assert _parse_tool_call(msg, _entry("cap-web-search"), ["query"]) == {"query": "clean"}


def test_native_call_empty_arguments_is_empty_extraction():
    msg = _call("cap-create-folder", "")
    assert _parse_tool_call(msg, _entry("cap-create-folder"), ["name"]) == {}


def test_native_call_name_mismatch_is_rejected():
    # A name that is a DIFFERENT capability would silently re-select — never allowed.
    msg = _call("cap-rag-search", '{"query": "x"}')
    with pytest.raises(ExtractionUnavailable, match="!= selected capability"):
        _parse_tool_call(msg, _entry("cap-web-search"), ["query"])


def test_native_call_missing_name_is_rejected():
    msg = {"tool_calls": [{"function": {"arguments": "{}"}}]}
    with pytest.raises(ExtractionUnavailable):
        _parse_tool_call(msg, _entry("cap-web-search"), ["query"])


def test_native_call_bad_json_arguments_is_rejected():
    msg = _call("cap-web-search", "{not valid json")
    with pytest.raises(ExtractionUnavailable):
        _parse_tool_call(msg, _entry("cap-web-search"), ["query"])


def test_native_call_non_object_arguments_is_rejected():
    msg = _call("cap-web-search", json.dumps([1, 2, 3]))
    with pytest.raises(ExtractionUnavailable):
        _parse_tool_call(msg, _entry("cap-web-search"), ["query"])


# ── the content fallback (old wire) can never mask a native-call error ────────────


def test_valid_content_cannot_mask_a_native_name_mismatch():
    # The reply carries a WRONG native call AND a perfectly valid content JSON:
    # the native error must win — no lenient content fallback.
    msg = {"content": '{"arguments": {"query": "safe"}}',
           "tool_calls": [{"function": {"name": "cap-rag-search",
                                        "arguments": '{"query": "x"}'}}]}
    with pytest.raises(ExtractionUnavailable):
        _parse_tool_call(msg, _entry("cap-web-search"), ["query"])


def test_valid_content_cannot_mask_malformed_native_arguments():
    msg = {"content": '{"arguments": {"query": "safe"}}',
           "tool_calls": [{"function": {"name": "cap-web-search",
                                        "arguments": "{oops"}}]}
    with pytest.raises(ExtractionUnavailable):
        _parse_tool_call(msg, _entry("cap-web-search"), ["query"])


def test_content_fallback_is_used_only_without_a_native_call():
    msg = {"content": '{"arguments": {"query": "clean"}}'}
    assert _parse_tool_call(msg, _entry("cap-web-search"), ["query"]) == {"query": "clean"}


# ── authorization + Binder gates still apply to native values ────────────────────


def test_native_unauthorized_slot_never_reaches_the_draft():
    msg = _call("cap-web-search", '{"query": "clean", "platform": "x"}')
    values = _parse_tool_call(msg, _entry("cap-web-search"), ["query"])
    # the parser preserves shape; the AUTHORIZATION gate drops the stray slot.
    draft = accept_model_values({"query": "raw"}, values, model_slots=("query",),
                                parameters=C.SCHEMA["cap-web-search"]["params"],
                                fallback_slots=("query",))
    assert draft == {"query": "clean"}
    assert "platform" not in draft


def test_social_platform_is_optional_in_the_shared_contract():
    # Product contract: omitting platform searches EVERY platform, so it must stay
    # omittable (never required) in both the handler plan and the function schema.
    params = C.SCHEMA["cap-social-search"]["params"]
    assert params["platform"]["required"] is False
    assert "platform" in C.model_slots("cap-social-search")
    h = handler_for("cap-social-search")
    assert "platform" in h.slot_plan(query="x", facts=None, draft={}).model_slots
