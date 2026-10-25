# Parameter Acquisition — Three Classes and One Contract

Status: **current design.** This page owns the three acquisition mechanisms, their
responsibility boundaries, the routing relations between them, and their failure handling.

| Layer | Authoritative document |
|---|---|
| Overall funnel, call chain, budget, LLM channel lifecycle | [architecture.md](architecture.md) §25.16 / §28 |
| Parameter-processing diagrams | [architecture-diagrams.md](architecture-diagrams.md#parameter-processing-flow) |
| Environment variables | [configuration.md](configuration.md) |

This page does not restate those layers. Where an older paragraph elsewhere names a
removed mechanism, the removed mechanism is recorded as history in §9.

---

## 0. Position in the turn

```
Matcher HIT ─────────────────────────────────────────────┐
                                                         ├─→ capability decided
Matcher MISS / AMBIGUOUS → Recall → candidate aggregation → cap_router ─┘
                                                         │
                                                         ▼
                                             ARGUMENT ACQUISITION
                                              (this page: 3 classes)
                                                         │
                                                         ▼
                                      Binder → certified ACTION → executor
```

- **`MATCH_HIT` invariant.** A Matcher HIT means the capability is already determined:
  no Recall, no candidate aggregation, no `cap_router`, **no re-selection**. The turn goes
  straight to argument acquisition.
- **MISS / AMBIGUOUS.** Recall → candidate aggregation → `cap_router` selects the
  capability. Acquisition begins only after that decision.
- **Acquisition never selects a capability**, and the extractor never sees the candidate
  list. A `REJECT` / `NONE` decision exits to the Agent before this stage.
- **The capability's arguments and the capability's selection are separate concerns.**
  Nothing in this page may be used to re-open a decided capability.

---

## 1. The three acquisition classes

Every argument a capability receives is produced by exactly one of three mechanisms.

| # | Class | Producer | Deterministic | Produces | Never produces |
|---|---|---|---|---|---|
| 1 | **Rules / handler / trusted context** | capability handlers (`cap_handler/`), slot plan, `declaration.py` / `evidence.py` / `context_values.py` | yes | the handler draft: verbatim query text, DET values, and trusted identifiers from `TurnFacts` / UI context / callback context | a model-regenerated identifier; a capability choice |
| 2 | **Dedicated page-scope resolver** | `argument_acquisition/page_scope_resolver.py` (online LLM, narrow contract) over the handler's `handler_page_default.py` | no (model) / yes (handler page) | only the page scope of a document request: `all` / `single` / `range`, merged with the viewer's current page of the same document (§3.2) | `asset_id`; any other slot; tool selection or execution |
| 3 | **Shared Qwen extractor** | `argument_acquisition/extractor.py` (native function-call contract) | no (model) | only the MODEL-owned, caller-authorized slots of the already-selected tool | a capability choice; SYSTEM_BINDER / TOOL_DEFAULT / UNAVAILABLE slots; document identity |

Class 2 is **capability-local**: it is invoked inside a file handler's `acquire()`, not as a
global parameter rule. Class 3 is **shared**: the handler lane and the generic Argument Path
Router both call the same extractor.

---

## 2. Class 1 — rules, handler and trusted context

### 2.1 What this class owns

This class turns the turn's already-known facts into a **draft**:

- deterministic values the handler computes from the sentence and the turn facts;
- trusted identifiers carried by `TurnFacts` (viewer asset, viewer kind, current page,
  selected text, callback context);
- tool defaults, which belong to the tool and are resolved at the boundary.

It never invokes a model to re-derive something the turn already knows, and it never
decides which capability to run.

### 2.2 Handler precedence

A capability with a registered parameter handler (§25.16) and a declared policy is owned by
that handler; the generic Argument Path Router is **not** entered. A capability without a
handler uses the generic router. Both paths share the same Binder / certified-handoff tail.

Wiring faults fail closed:

- a handler mounted **without** a declared policy → Agent (a handler must not choose its own
  strategy implicitly);
- a declared handler-backed policy **without** a mounted handler → Agent;
- `NO_ARGS` on a capability that publishes parameters or mounts a handler → Agent.

### 2.3 Declared policies

The per-capability strategy table is explicit (`cap_handler/acquisition_policy.py`). It is
never inferred from `parameters == {}`, from a missing handler, or from “the slot is not in
the draft” — those are different business states.

| Policy | Meaning | Extractor |
|---|---|---|
| `NO_ARGS` | the capability genuinely takes no arguments (empty Registry schema and a tool that requires none) | never |
| `RULE_CONTEXT` | the handler resolves **every** slot from trusted context, the viewer, an attachment or an explicit rule in the user's sentence | never by the shared extractor; the handler may invoke class 2 |
| `MODEL` | the handler owns the capability; the shared extractor fills only the MODEL-owned slots the handler authorizes | at most once |
| `MIXED` | the handler has authoritative deterministic slots **and** MODEL-owned slots; deterministic values are never overwritten | at most once |
| `DEFERRED_AGENT` | the capability is admitted, but its real tool needs an input with no legal producer on this path | never — exit to the Agent |

An undeclared capability is not a policy. On the generic path it is
`ACQUISITION_UNDECLARED` → Agent; it never falls back to a fused select-and-extract path.

### 2.4 Trusted context first

If the turn already carries a trusted value, acquisition injects it verbatim:

- a viewer asset id, viewer kind, current page or selection;
- a `TurnFacts` field, a callback context value, or an attachment identity;
- an explicit filename or path the user wrote in the sentence.

**A trusted identifier is never regenerated by a model.** `asset_id` in particular is
**never produced by Qwen**, in either direction: it does not appear in the extractor's input
schema, and a model-copied or hallucinated identity field is rejected before the Binder.

### 2.5 Per-capability semantics and defaults stay with the capability

This page defines no tool-specific parameter rules. Each handler owns the meaning and the
default of its own slots:

- **File capabilities** — the page default belongs to the capability. `all` means “the
  request restricts no pages”: the tool call omits `pages` and the tool applies its own
  documented default. The handler's same-document current page participates only under the
  merge order in §3.2. It does not mean “page 1”, and it is never the result of a failure.
- **Summary** — `output_dir` is requested only when the sentence explicitly asks to save.
  With no save cue it is never requested, and the tool returns the summary to the chat
  window instead of writing a file.
- **Vision** — a video viewer always uses the client capture, never a cloud asset id,
  because the vision tool cannot read a video file.
- **Mind map / slides** — the real tool contract needs a selected input with no legal
  producer on this path, so the capability is `DEFERRED_AGENT`; no argument is invented to
  bridge the gap.

### 2.6 Slot plan, not inference

A handler's `slot_plan()` states, per slot, one of `ACQUIRED` / `PENDING_MODEL` /
`DEFAULTED` / `MISSING`. The caller never infers “needs a model” from “the slot is absent
from the draft”.

- An empty plan → no extractor call.
- A slot in `required_model_slots` must be filled **this turn**; if the extractor is
  unavailable or returns no valid value for it, the turn exits to the Agent rather than
  certifying a draft that silently dropped a requested argument.
- Any other authorized MODEL slot that fails extraction leaves the handler draft intact, so
  the original request is never lost.

### 2.7 Deterministic inputs of the generic Argument Path Router

The production provider (`argument_acquisition/provider.py::for_turn`) composes three
non-LLM producers per turn:

- **declaration** — the capability's `{slot: SlotDecl}`. Ownership is a property of the slot
  name via a frozen table, reconciled with the legacy `arg_slots` source when present; an
  explicit declaration always wins over the name heuristic.
- **evidence** — the deterministic `{slot: source}` signal. Evidence is **deterministic**,
  schema-aware and non-LLM; deciding whether to call the extractor is never delegated to an LLM.
  Evidence is a **router input** produced outside the router.
- **context values** — the settled SYSTEM_BINDER values and their actual sources from the
  turn facts.

`allowed_sources` gates which sources are acceptable; it never creates evidence. A MODEL
slot's source is **evidence-driven** — it is **never inferred from `allowed_sources`**: a
slot that merely allows `QUERY` does not thereby have query evidence.

Provenance records the **actual** source and value. A declaration's `allowed_sources[0]` is
never a legal provenance fallback, and a system slot with no actual value is not merged.

---

## 3. Class 2 — the dedicated page-scope resolver

### 3.1 Responsibility

One narrow job: turn the user's sentence plus the document context the turn already carries
into the reading scope a file tool accepts.

It emits exactly one verdict:

| Verdict | Meaning | Acquisition result |
|---|---|---|
| `all` | the request does not restrict pages — an explicit whole-document request, or no page named | the tool call omits `pages` (whole document); this overrides the handler value |
| `single` | exactly one page, including deictic “this page” resolved to the viewer's current page | canonical page spec |
| `range` | several pages, in the shared strict parser's grammar (`"3-5"`, `"1,3"`), bounded by `MAX_REQUESTED_PAGES` | canonical page spec |

### 3.2 Merge with the handler value

The handler supplies at most one deterministic candidate: the viewer's current page,
and only when `viewer_asset_id` equals the target `asset_id`. The producer is a value
reader on `TurnFacts`, not a sentence parser; a page belonging to another document, a
missing page and a non-positive / non-numeric page all yield no value.

| Model verdict | Handler value | Merged `pages` |
|---|---|---|
| `all` | any | omitted — the whole document; the model's explicit value wins |
| `single` / `range` | any | the model's canonical spec |
| semantic `unknown` / `ambiguous` | present | the handler value (`viewer` source) |
| semantic `unknown` / `ambiguous` | absent | missing — the capability follows its own failure path |
| `unavailable` / `invalid` | any | no merge — the capability follows its own failure path |

A valid model verdict always overrides the handler value. `asset_id` is system-owned:
neither source may be produced by the model, and a reply carrying one is a contract
violation.

### 3.3 Hard boundaries

- It produces **only** the page scope. No tool selection, no tool execution, no other
  business parameter.
- It never produces or rewrites `asset_id`. Document identity stays in `TurnFacts` and the
  existing authorization path; an identity field in the reply is a contract violation.
- It emits only the agreed JSON. An unknown field, an invalid page spec, an out-of-range
  page or an ambiguous target is rejected.
- It never guesses a page, never widens a half-understood range to the whole document, and
  never turns a failure into `all`.
- It runs only for handlers that declare `accepts_page_scope`, and only when the host wired
  the seam. It is a capability-local resolver, not a global extraction rule.

### 3.4 Non-paginated viewers — a structural skip

When the viewer kind is known to be non-paginated (for example an image), the model is not
consulted: there is nothing for it to settle. This is a **structural** check on the viewer
kind, not a set of deterministic keyword rules replacing model page semantics.

For every paginated document — and for every case where the viewer kind does not settle the
question — the model remains the sole semantic page judge. No sentence keyword rule may
override it. The handler's current-page value participates only under the merge order in
§3.2. The skip is recorded as “not asked”; it is never converted into an `all` verdict.

### 3.5 Failure taxonomy

The three outcomes stay distinct and never collapse:

- `ambiguous` — the model could not decide the target or the scope. This is a **semantic**
  “not extracted”, and the only outcome that keeps the handler's page value;
- `unavailable` — the call failed, timed out, or returned non-JSON. This is a **technical**
  failure: no reliable result exists;
- `invalid` — the reply violated this contract (bad enum, bad pages, identity field,
  unknown field, out-of-range page). This is also a **technical** failure.

`unavailable` and `invalid` never fall back to the handler value and are never reported as
a successful read. With no handler value, all three take the caller's acquisition-failure
path. None is retried into a different parameter, and none becomes a default beyond the
handler value named in §3.2.

---

## 4. Class 3 — the shared Qwen extractor

### 4.1 One extractor, two callers

The same extractor serves:

1. the **handler lane**, for `MODEL` / `MIXED` policies and the handler's authorized slot
   plan;
2. the **generic Argument Path Router**, for the `QUERY_TO_EXTRACTOR`,
   `QUERY_PLUS_5TURNS_TO_EXTRACTOR` and `MIXED` strategies.

There is no second extraction model and no fused select-and-extract path.

### 4.2 Fixed contract

- **Input:** only the MODEL-owned slots the caller authorizes, with their schema from
  `Registry.parameters`, plus the sanctioned context bundle.
- **Sanctioned bundle:** the current query alone, or the current query plus the last **5**
  `role == "user"` messages. Never the full transcript; never assistant / system / Agent
  history; never the candidate list.
- **Excluded slots:** `SYSTEM_BINDER`, `TOOL_DEFAULT` and `UNAVAILABLE` slots are not sent
  and are stripped if they appear in the reply. `asset_id` is never produced by Qwen.
- **Output is a draft, not authority.** The Binder validates. In a `MIXED` merge the
  deterministic value wins any collision, and per-slot provenance is preserved.

### 4.3 Call discipline

- **At most one call by default** on every strategy.
- Escalation from query-only to query + last-5 user turns is allowed **only when query-only
  acquisition is proven insufficient** — never “for safety”, never pre-emptively.
- The extractor is optional. A `NO_ARGS` or `RULE_CONTEXT` turn makes zero extraction calls;
  a capability must not be forced through Qwen merely because the model exists.

### 4.4 Output handling

- Valid authorized values are folded into the handler draft.
- An empty, invalid or unavailable reply leaves the handler draft intact, **unless** a slot
  was declared as required for this turn — then the turn exits to the Agent.
- A value is never fabricated to fill a slot, and a failed extraction never silently
  widens the operation (for example, a failed page extraction never becomes “whole
  document”).

---

## 5. Routing relations

Routing is by declared policy and deterministic inputs — never by inference from an empty
schema or a missing slot.

| Situation | Acquisition path | Model call |
|---|---|---|
| Policy `NO_ARGS` | empty arguments | none |
| Policy `RULE_CONTEXT` (handler) | handler draft only | none, unless the handler invokes class 2 |
| Policy `MODEL` (handler) | handler draft + shared extractor over authorized slots | Qwen, at most one |
| Policy `MIXED` (handler) | handler's deterministic slots + extractor for the remaining authorized slots | Qwen, at most one |
| Policy `DEFERRED_AGENT` | no value is produced | none — Agent |
| No policy, handler mounted | wiring fault | none — Agent |
| No policy, no handler | generic Argument Path Router | per strategy below |
| ARP `CONTEXT_DIRECT` | every required slot is already settled | none |
| ARP `QUERY_TO_EXTRACTOR` | extractor over the current query | Qwen, at most one |
| ARP `QUERY_PLUS_5TURNS_TO_EXTRACTOR` | extractor over the query plus the last 5 user turns | Qwen, at most one |
| ARP `MIXED` | system values + extractor values, merged with provenance | Qwen, at most one |
| ARP `MISSING` | no legal value for a required slot | none — Agent / follow-up |

The ARP's strategy vocabulary is model-agnostic: the names say where a MODEL-owned value
comes from, never which model extracts it. Swapping the extractor does not rename a
strategy.

---

## 6. Readiness and evidence — the double gate

Strategies are determined by two independent dimensions, evaluated in order. There is
**no linear precedence** between strategies and no global ladder such as
`MISSING > CONTEXT_DIRECT > …`.

**Gate 1 — Required Readiness.** For every `parameters[slot].required == True`:

- `SYSTEM_BINDER` → must have a real deterministic value now, or `UNSAT`;
- `TOOL_DEFAULT` → must resolve to the tool default, or `UNSAT`;
- `MODEL` → `PENDING_MODEL`, deferred to Gate 2;
- `UNAVAILABLE` (and required) → `UNSAT`.

**Gate 2 — Acquisition need.** Over MODEL-owned slots:

- a **required** MODEL slot always needs acquisition;
- an **optional** MODEL slot needs acquisition **iff** a real evidence signal exists. The
  mere presence of an optional MODEL slot in the schema is not evidence and must not
  trigger the extractor.

A required MODEL slot's source is decided by the actual evidence (query vs. last-5 user
turns), never by `allowed_sources`. If no legal source exists, the result is `MISSING`,
never a guess.

**A required parameter with no declaration entry** has **no legal acquisition** source:
ownership is undeclared, so nothing may fill it. It is treated as `UNAVAILABLE` →
`MISSING` → Agent / follow-up. It is never resolved by a legacy fallback, and no separate
“incomplete declaration” status is introduced — the outcome is the ordinary `MISSING` path.

---

## 7. Failure handling

### 7.1 Exit reasons

| Reason | Situation |
|---|---|
| `ACQUISITION_UNDECLARED` | no declared policy and no declaration on the generic path — the approved acquisition-undeclared semantics |
| `ACQUISITION_POLICY_CONFLICT` | `NO_ARGS` with a schema/handler; handler without a policy; a strategy with no dispatcher |
| `ACQUISITION_HANDLER_MISSING` | a handler-backed policy with no mounted handler |
| `ACQUISITION_DEFERRED` | `DEFERRED_AGENT`: the real tool contract cannot be satisfied on this path |
| `ACQUISITION_MISSING` | no legal value for a required slot |
| `ACQUISITION_MODEL_PENDING` | the strategy needs a model value that was not produced |
| `ACQUISITION_TIMEOUT` | the acquisition stage exceeded its wall-clock budget |
| `ACQUISITION_TERMINAL` | safety guard: an attempt to reach the executor without a usable argument record |

A handler that cannot produce a draft returns an `AcquisitionFailure` record instead of
raising. The record is labelled `CAP_HANDLER_ABORTED` in the acquisition capture and keeps
the handler's own category: `scope_contract_violation`, `argument_extraction_failed`,
`scope_unsettled`, `no_content`, `invalid_destination`. Its `terminal` flag marks a request
whose parameters could not be settled by any trusted producer; that turn ends with the
handler's honest answer rather than being papered over by the Agent. Every other handler
failure keeps its Agent-facing routing-context behaviour.

### 7.2 Common rules

- **Never fabricate an argument** to keep a turn certifiable.
- **Never silently widen the operation** — a failed page decision is not `all`; a dropped
  required slot is not “it was optional”.
- **Fail closed.** An acquisition failure exits to the Agent / follow-up with a structured
  record. The record is routing context for the Agent; it never authorizes a tool call.
- **One Binder tail.** Whatever produced the draft, it goes through the same
  Binder → certified ACTION → executor path. Class boundaries never create a second
  execution path.
- **Stage attribution.** An acquisition timeout is attributed to acquisition, never to the
  capability-selection hop that already returned.

---

## 8. Boundaries and design constraints

- Acquisition never selects or re-selects a capability.
- The extractor never sees candidate lists, and the page resolver never supplies document
  identity.
- **Qwen is optional.** Only MODEL-owned slots that require extraction invoke it; rule-only
  and no-argument turns make no model call.
- **Trusted context first.** Trusted identifiers are injected verbatim and are never
  regenerated by a model.
- **Per-capability semantics stay per capability.** This page is not a global
  parameter-rule engine; defaults and slot meanings are owned by the handler and the tool.
- **A technical model failure is never a default.** `unavailable` / `invalid` exit; neither
  degrades into a wider operation, a guessed value, or the handler's page. A semantic
  `ambiguous` reply is not a failure: it keeps the handler's page value when one exists
  (§3.2).
- **Future design goal, not current behavior.** Complex or unreliable `MODEL` / `MIXED`
  slots may later be planned to escalate to the online LLM through an additional narrow,
  per-parameter resolver in the style of class 2. That is a design goal only. Today the
  online LLM resolves **page scope alone**, and all other MODEL-owned slots go through the
  shared Qwen extractor.
- **No HTTP, routing or permission responsibility** belongs to this stage. Model access is
  governed by the dispatch gateway and its channel resolution
  ([architecture.md](architecture.md) §12.6); connection reuse is described in §12.8.

---

## 9. Historical decisions retained

The rules below were approved in the Step-0 contract and remain binding. The surrounding
framing of that document — “specification only, nothing implemented” — is historical: the
contract shipped, and `architecture.md` §28 plus this page describe the implemented scheme.

- **Declaration is separate from schema.** `Registry.parameters` remains the ONLY schema
  truth. The acquisition declaration carries `ownership` / `allowed_sources` / `escalation`
  only. It **must not reuse** the legacy `arg_slots` string semantics
  (`{slot: "user_input" | "plugin:<name>" | …}`) as a `SlotDecl` contract; the legacy shape
  is bridged only to select an ownership and disappears once the Registry serves an
  acquisition entry directly.
- **Four ownership values:** `MODEL`, `SYSTEM_BINDER`, `TOOL_DEFAULT`, `UNAVAILABLE`.
  `SYSTEM_BINDER` draws from `UI_CONTEXT` / `RESOLVER` / `CALLBACK_CONTEXT`; `MODEL` draws
  from `QUERY` / `CONVERSATION_5_USER_TURNS`; the two sets are disjoint.
- **The `backend=off | stub | laya` switch was removed** by the single-path ruling, together
  with the fused `ToolIntentModel.select_and_extract` lane. `backend=off` is not a current
  configuration option and must not be reintroduced as a rollback path; the funnel is the
  only formal lane. `ToolIntentModel.select_and_extract` is never a fallback.
- **`REASON_CAP_ROUTER_HIT_DEFERRED`** (`CAP_ROUTER_HIT_DEFERRED`) was transitional and has
  been removed. A `MATCH_HIT` goes directly to acquisition with **no re-selection**.
- **No “incomplete declaration” state.** A required parameter with no declaration entry is
  the ordinary `MISSING` path (see §6); no extra status or reason is defined.
- **Evidence is deterministic, schema-aware and non-LLM**, and was originally deferred to
  Step 2 as a router input. It is now produced by `evidence.py` and injected.
- **Strategy determination is two-dimensional** (Required Readiness, then Acquisition need),
  with no linear precedence.
- **Provenance records the actual source**, and `allowed_sources[0]` is never used as a
  fallback.
