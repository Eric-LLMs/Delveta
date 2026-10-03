"""Contract-alignment regression pins from the Registry /
Capability Contract Audit (P0 + P1 only), with the 0927 lane correction.

Plane 1 (P0, lane-corrected): ``cap-research`` is an ACTIVE research-lane
capability — it stays visible in the table with its curated corpus (the
research lane executes via Plugin mount + handoff, never via this row), while
``intent_kind="research"`` removes it from every chat/files funnel plane:
Matcher exact index, Recall corpus, funnel entries_by_id (all via the single
``chat_plane_candidate`` predicate) and the executor/funnel kind gate as the
certification backstop. Its binding names the research Plugin, a legal mount
unit the write gate cross-checks against the plugin roster, not the chat
ToolRuntime roster.

Plane 2 (P1): for the seven backfilled capabilities the LIVE Registry
``parameters`` agree with the real tool schema — identical required set,
declared slots exist, full runtime slot set covered, and every stated
``max_len`` matches the runtime bound.

Plane 3 (no-regression): the healthy three (add-term, create-folder,
pdf-extract-text) keep their frozen parameters, still pass the Binder gate
with a complete draft, and pass the executor's final schema gate; an empty
draft is MISSING (never a silent COMPLETE {} — the F-1 failure mode).

The Registry tables are the runtime truth (ruling 0014), so these tests read
the LIVE dev Registry; they skip honestly when the DB is unreachable (same
posture as the p5 prod lane).
"""
from __future__ import annotations

import pytest
from sqlalchemy import exc as sa_exc

from core.application.chat.intent_funnel import binder, recall
from core.application.chat.intent_funnel.binder.binder import _CONTEXT_SLOT_SOURCES
from core.application.chat.intent_funnel.matcher import build_index
from core.application.chat.intent_funnel.registry import active_view, list_capabilities
from core.application.chat.intent_funnel.registry import entry as entry_mod
from core.infrastructure.db import SessionLocal

# ── the live roster, assembled from the REAL tool definitions ─────────────────────

class _RT:
    def __init__(self):
        self.defs = {}

    def register(self, d):
        self.defs[d.name] = d


class _Ctx:
    def resolve(self, name):  # register-time bodies never resolve
        raise AssertionError(f"ctx.resolve({name!r}) at register time")


def _real_roster() -> dict[str, dict[str, dict]]:
    """tool -> {slot: {"max_len", "required"}} from the same source of truth
    the executor projects (apps/api tool modules + the social plugin)."""
    from api.tools import (
        add_term_tool, create_folder_tool, pdf_tools, rag_search_tool,
        read_document_tool, translate_tool, vision_tool, web_search_tool,
    )
    from plugins.social_search.plugin import PLUGIN as SOCIAL_PLUGIN

    rt = _RT()
    ctx, llm = _Ctx(), None
    for mod in (add_term_tool, create_folder_tool, pdf_tools, rag_search_tool,
                read_document_tool, translate_tool, vision_tool, web_search_tool):
        mod.register(rt, ctx, llm)
    for d in SOCIAL_PLUGIN.tools:
        rt.defs[d.name] = d
    out: dict[str, dict[str, dict]] = {}
    for name, d in rt.defs.items():
        params = d.parameters or {}
        props = params.get("properties") or {}
        req = params.get("required")
        out[name] = {
            str(k): {"max_len": int((v or {}).get("maxLength") or 0),
                     "required": bool(req is None or k in (req or []))}
            for k, v in props.items()
        }
    return out


ROSTER = _real_roster()

# capability -> (tool, params declared in the P1 backfill / frozen healthy rows)
P1_CAPS = {
    "cap-rag-search": "rag_search",
    "cap-web-search": "web_search",
    "cap-social-search": "search_social",
    "cap-translate": "translate",
    "cap-vision": "vision",
    "cap-read-document": "read_document",
    "cap-pdf-table-to-text": "pdf_table_to_text",
}
HEALTHY_CAPS = {
    "cap-add-term": "add_term",
    "cap-create-folder": "create_folder",
    "cap-pdf-extract-text": "pdf_extract_text",
}
# The exact P1/healthy parameters written — pinned verbatim so a
# later edit must be a deliberate, reviewed change (dual-source drift is the
# original audit finding).
FROZEN_PARAMETERS: dict[str, dict] = {
    "cap-add-term": {
        "term": {"type": "string", "max_len": 120, "required": True,
                 "description": "the word/term to add, quotes removed"},
        "domain": {"type": "string", "max_len": 60, "required": True,
                   "description": "the vocabulary domain to add it into"},
    },
    "cap-create-folder": {
        "name": {"type": "string", "max_len": 120, "required": True,
                 "description": "the folder name, quotes removed"},
    },
    "cap-pdf-extract-text": {
        "asset_id": {"type": "string", "max_len": 64, "required": True,
                     "description": "asset id of the attached/opened document "
                                    "(from turn facts, not the sentence)"},
    },
    "cap-rag-search": {
        "query": {"type": "string", "required": True,
                  "description": "the search query, exactly as the user words it"},
        "top_k": {"type": "integer", "required": False,
                  "description": "optional result count; only when the sentence states one"},
        "domain": {"type": "string", "required": False,
                   "description": "optional domain id scope; only when the sentence names one"},
    },
    "cap-web-search": {
        "query": {"type": "string", "required": True,
                  "description": "the search query, exactly as the user words it"},
        "top_k": {"type": "integer", "required": False,
                  "description": "optional number of results; only when the sentence states one"},
    },
    "cap-social-search": {
        "query": {"type": "string", "required": True,
                  "description": "the search query, exactly as the user words it"},
        "platform": {"type": "string", "required": False,
                     "description": "optional platform (reddit / x / zhihu / auto); "
                                    "only when the sentence names one"},
        "subreddit": {"type": "string", "required": False,
                      "description": "optional reddit-only scope; only when the sentence names one"},
        "limit": {"type": "integer", "required": False,
                  "description": "optional max results per platform; only when the sentence states one"},
    },
    "cap-translate": {
        "text": {"type": "string", "required": True,
                 "description": "the text to translate, copied from the sentence"},
    },
    "cap-vision": {
        "asset_id": {"type": "string", "max_len": 64, "required": True,
                     "description": "asset id of the attached/opened image "
                                    "(from turn facts, not the sentence)"},
        "question": {"type": "string", "required": False,
                     "description": "optional question about the image, copied from the sentence"},
    },
    "cap-read-document": {
        "asset_id": {"type": "string", "max_len": 64, "required": True,
                     "description": "asset id of the attached/opened document "
                                    "(from turn facts, not the sentence)"},
        "pages": {"type": "string", "required": False,
                  "description": "optional page/slide spec like \"2\" or \"1-3\"; "
                                 "only when the sentence states one"},
    },
    "cap-pdf-table-to-text": {
        "asset_id": {"type": "string", "max_len": 64, "required": True,
                     "description": "asset id of the attached/opened document "
                                    "(from turn facts, not the sentence)"},
    },
}

# minimal complete draft per capability (required slots, plausible strings)
MINIMAL_ARGS = {
    "cap-rag-search": {"query": "注意力机制"},
    "cap-web-search": {"query": "2026 AI news"},
    "cap-social-search": {"query": "transformer papers discussion"},
    "cap-translate": {"text": "The weather is nice."},
    "cap-vision": {"asset_id": "0ea50a94-f4f7-45eb-bbaa-43966d6af575"},
    "cap-read-document": {"asset_id": "0ea50a94-f4f7-45eb-bbaa-43966d6af575"},
    "cap-pdf-table-to-text": {"asset_id": "0ea50a94-f4f7-45eb-bbaa-43966d6af575"},
    "cap-add-term": {"term": "photosynthesis", "domain": "English"},
    "cap-create-folder": {"name": "notes"},
    "cap-pdf-extract-text": {"asset_id": "0ea50a94-f4f7-45eb-bbaa-43966d6af575"},
}


async def _entries_by_id() -> dict:
    try:
        entries = await list_capabilities(session_factory=SessionLocal)
    except (sa_exc.SQLAlchemyError, OSError) as exc:  # DB down — honest skip
        pytest.skip(f"live Registry unreachable: {exc!r}")
    return {e.capability_id: e for e in entries}


@pytest.fixture(autouse=True)
async def _fresh_pool_each_test():
    """pytest-asyncio gives every test its own loop; the shared asyncpg pool
    must not carry connections across loops (teardown would bind to a closed
    one)."""
    yield
    from core.infrastructure.db import engine
    await engine.dispose()


# ── Plane 1: cap-research ─────────────────────────────────────────────────────────

def test_research_is_a_plugin_name_not_a_roster_tool():
    """"research" must NOT exist as an executable tool: the C2 terminal the
    audit flagged (route promises a capability the system cannot run and the
    doctrine forbids the Agent to recover from it)."""
    assert "research" not in ROSTER
    from plugins.research.plugin import build_research_plugin

    names = {t.name for t in build_research_plugin(None).tools}
    assert "research" not in names and names, "research plugin tools unexpectedly changed"


async def test_cap_research_lane_scoped_active_and_invisible_in_chat_planes():
    """Lane ruling ( correction): cap-research is NOT globally
    disabled — it stays active for the RESEARCH lane (catalog + curated
    corpus), while ``intent_kind="research"`` removes it from every chat/files
    funnel consumption point."""
    entries = await _entries_by_id()
    e = entries["cap-research"]
    assert e.enabled and e.status == "active", \
        "cap-research must stay active: the research lane owns this row"
    assert e.intent_kind == "research"
    # not a chat-plane candidate at all (the single shared predicate)
    assert not entry_mod.chat_plane_candidate(e)

    view = await active_view(session_factory=SessionLocal)
    assert view is not None
    # Matcher exact index
    indexed = {cid for caps in build_index(view).get("exact", {}).values()
               for cid in caps}
    assert "cap-research" not in indexed
    # Recall corpus (SQL now carries c.intent_kind <> 'research')
    index = await recall.load_index(SessionLocal)
    if index is not None:
        assert "cap-research" not in {c.capability_id for c in index.corpus}
    # funnel entries_by_id uses the same predicate; and the certification/
    # TOCTOU backstop: the funnel kind gate never opens for research
    from core.application.chat.intent_funnel.funnel import kind_enabled
    assert kind_enabled("research") is False


# ── Plane 2: P1 parameters == real tool schema ────────────────────────────────────

async def test_p1_parameters_agree_with_real_tool_schema():
    entries = await _entries_by_id()
    for cid, tool in P1_CAPS.items():
        e = entries[cid]
        declared = {k: v for k, v in (e.parameters or {}).items()}
        runtime = ROSTER[tool]
        assert declared, f"{cid}: parameters still empty — the F-1 failure mode is back"
        # full-slot agreement (the 0014 write-gate rule)
        assert set(declared) == set(runtime), \
            f"{cid}: declared slots {sorted(declared)} != runtime slots {sorted(runtime)}"
        # required flags mirror the tool schema
        assert {k for k, v in declared.items() if v.get("required")} \
            == {k for k, v in runtime.items() if v["required"]}, \
            f"{cid}: required set drifted"
        # stated max_len must equal a runtime bound that states one
        for k, spec in declared.items():
            ml = spec.get("max_len")
            if ml is not None and runtime[k]["max_len"]:
                assert int(ml) == runtime[k]["max_len"], f"{cid}: max_len[{k}] drift"
        # Card-minimal shape on every slot
        for k, spec in declared.items():
            assert isinstance(spec, dict) and spec.get("type") and spec.get("description")
            assert isinstance(spec.get("required"), bool), f"{cid}: required[{k}] not bool"
        # the exact written values are pinned — any change must be deliberate
        assert declared == FROZEN_PARAMETERS[cid], f"{cid}: parameters drifted off the 0927 fix"


async def test_healthy_three_keep_frozen_parameters():
    """No-regression pin: the three pre-existing rows are untouched by the
    fix (their optional-slot full-schema drift is a REPORTED gate finding, not
    silently rewritten here)."""
    entries = await _entries_by_id()
    for cid, tool in HEALTHY_CAPS.items():
        e = entries[cid]
        assert dict(e.parameters or {}) == FROZEN_PARAMETERS[cid], \
            f"{cid}: healthy row changed — audit scope violation"
        runtime = ROSTER[tool]
        for k, spec in e.parameters.items():
            assert k in runtime, f"{cid}: slot {k!r} unknown to {tool}"
            if spec.get("max_len") is not None and runtime[k]["max_len"]:
                assert int(spec["max_len"]) == runtime[k]["max_len"]


# ── Plane 3: Binder + executor schema gates ───────────────────────────────────────

# (E2E-matrix ruling): asset_id is a CONTEXT slot — the Binder
# sources it from TurnFacts, never from the model draft. The contract tests
# therefore supply the settled fact; the draft's copy is inert either way.
_FACTS_ASSET_ID = "0ea50a94-f4f7-45eb-bbaa-43966d6af575"


def _facts_for(cid: str):
    """TurnFacts carrying the asset the turn points at — harmless for caps
    without an asset slot (the injector only touches slots in the schema)."""
    from core.application.chat.intent_funnel.contract import TurnFacts
    return TurnFacts(attachment_asset_id=_FACTS_ASSET_ID)


# caps whose minimal draft carries asset_id = context-sourced slots
_ASSET_SOURCED = {cid for cid, args in MINIMAL_ARGS.items() if "asset_id" in args}


async def test_binder_requires_the_real_slots_and_accepts_a_complete_draft():
    from core.application.chat.intent_funnel.contract import TurnFacts
    entries = await _entries_by_id()
    for cid in {**P1_CAPS, **HEALTHY_CAPS}:
        e = entries[cid]
        draft_slots = {k for k, v in e.parameters.items()
                       if v.get("required") and k not in _CONTEXT_SLOT_SOURCES}
        if draft_slots:
            # F-1 regression: a cap with sentence-owned slots must NOT certify
            # on an empty draft (the old parameters={} short-circuit)
            assert not binder.validate(e, {}, _facts_for(cid)).is_complete, \
                f"{cid}: empty draft still COMPLETE — F-1 regression"
        else:
            # contract: an asset-only cap is FULLY determined by the
            # turn fact — empty draft + real attachment certifies,
            # empty draft + no fact stays MISSING (facts, not the model, decide)
            assert binder.validate(e, {}, _facts_for(cid)).args == \
                {"asset_id": _FACTS_ASSET_ID}, f"{cid}: fact did not fill asset slot"
            assert not binder.validate(e, {}, TurnFacts()).is_complete, \
                f"{cid}: asset certified with no turn fact"
        # a complete minimal draft normalizes to COMPLETE
        bound = binder.validate(e, dict(MINIMAL_ARGS[cid]), _facts_for(cid))
        assert bound.is_complete, f"{cid}: minimal draft rejected: {bound.state}"
        assert set(bound.args) == {
            k for k, v in e.parameters.items() if v.get("required")}
        # the context-slot contract: an asset_id ONLY from the
        # draft, with no turn fact, is never certified (no hallucinated bind)
        if cid in _ASSET_SOURCED:
            naked = binder.validate(e, dict(MINIMAL_ARGS[cid]), TurnFacts())
            assert not naked.is_complete, f"{cid}: draft-only asset_id certified"


async def test_executor_schema_gate_passes_the_certified_args():
    """The audit's certify->escalate defect: args that clear the Binder must
    ALSO clear validate_action against the live roster (no ActionSchemaError
    escalation for a correctly extracted turn)."""
    entries = await _entries_by_id()
    for cid, tool in {**P1_CAPS, **HEALTHY_CAPS}.items():
        e = entries[cid]
        bound = binder.validate(e, dict(MINIMAL_ARGS[cid]), _facts_for(cid))
        out = binder.validate_action(tool, bound.args, tool_schemas=ROSTER)
        assert out["tool"] == tool
        for k in {s for s, v in ROSTER[tool].items() if v["required"]}:
            assert k in out["args"], f"{cid}: required slot {k} lost at the final gate"


def test_research_binding_would_have_terminated_c2():
    """Documents the P0 rationale at the executor's own gate: a certified
    turn binding to "research" fails the live-roster existence check."""
    with pytest.raises(Exception) as exc_info:   # ActionSchemaError (pre-execution)
        binder.validate_action("research", {}, tool_schemas=ROSTER)
    assert "roster" in str(exc_info.value)
