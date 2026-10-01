# Phase 4 — Argument Acquisition Contract (Step 0)

Status: **Step 0 (2026-10-01). Specification only. Nothing implemented, nothing wired.**

Follow-up review (2026-10-01): rulings #1/#3/#4/#5 are folded in as explicit contract rules —
§B (evidence-driven MODEL slot source; actual-source merge provenance), §C (a required
parameter with no declaration entry → `MISSING`), §D (required MODEL slot source),
§E (evidence as a router input; the detector is deferred to Step 2). The status line above
still describes Step 0: this document is a specification; only the pure ARP modules of
Step 1 implement it, and nothing here is wired.

Precedence: this document is binding per `CLAUDE.md` → *Business Logic First*. Where current
code conflicts, the current code is **legacy / current implementation**, never the spec.
No production code, no `orchestrator` change, no `path_router`, no Qwen, no Laya, no
`selection_transition` removal, no Binder/Executor change are performed in Step 0.

Scope boundary (this file): the acquisition contract that sits **between** a decided
capability and the existing Binder. Abstract nodes only — the concrete modules
(`path_router.py`, `context_bundle.py`, `merge.py`, the Qwen extractor) are **Step 1+**.

Only business logic the user has explicitly approved is stated as contract. The four
previously open items (strategy precedence, declaration completeness, reason constants,
evidence constraints) are now **RESOLVED** — see the rulings inline in §B, §C, §E, §G.

---

## A. Approved Phase 4 business contract

1. **`Registry.parameters` is the ONLY schema truth.** `required` / `type` / `enum` /
   `description` / `max_len` come from `parameters` alone.
2. **The acquisition declaration is a NEW, independent entry** (working name
   `acquisition`). It MUST NOT reuse the legacy `arg_slots` string semantics
   (`{slot: "user_input" | "plugin:<name>" | ...}`) and MUST NOT be interpreted by a
   `SlotDecl`. The acquisition declaration describes **ownership / allowed_sources /
   escalation only** — never required/type/enum/description.
3. **Opt-in is explicit.** On the new lane (`backend=stub|laya`):
   - a capability **with** an `acquisition` declaration MAY enter the Argument Path Router;
   - a capability **without** one MUST NOT fall back to legacy
     `ToolIntentModel.select_and_extract`; the safe result is **Agent** with an
     acquisition-undeclared reason (see §G).
4. **`backend=off` is the legacy lane, unaffected.** It keeps the full legacy chain
   (HIT and MISS/AMBIGUOUS alike) byte-identically. The new contract neither reads nor
   changes it.
5. **Ownership is exactly four values** — `MODEL`, `SYSTEM_BINDER`, `TOOL_DEFAULT`,
   `UNAVAILABLE` — with the derivations in §C.
6. **`SYSTEM_BINDER` slots NEVER enter a Qwen schema. In particular, `asset_id` is never produced by Qwen**, in either direction (a model-copied or hallucinated asset id dies before the Binder).
7. **Readiness is a DOUBLE GATE** (§D): *Required Readiness* and *Acquisition Need* are
   distinct and MUST NOT be conflated.
8. **An optional MODEL slot existing in the schema is NOT evidence.** It triggers
   acquisition only when a deterministic, schema-aware *evidence* signal is present (§E);
   deciding “call Qwen or not” is never delegated to an LLM.
9. **cap_router and Qwen never overlap.** cap_router = capability selection only
   (`ONE | NONE`), never args, never Binder, never a tool call. Qwen = MODEL-owned
   argument acquisition only, and only AFTER the capability is decided; it never selects
   a capability.
10. **MATCH_HIT invariant.** When the Matcher returns HIT with a known capability: **no
    Recall, no Candidate Aggregation, no cap_router, no re-selection** — the turn goes
    **straight to the Argument Path Router**. Phase 3's `CAP_ROUTER_HIT_DEFERRED` is
    transitional compatibility only and is **replaced by the ARP in Phase 4**.

---

## B. Strategy determination — two dimensions (no global precedence)

**RESOLVED:** a strategy is **not** chosen by a fixed global precedence. It is decided by
two independent dimensions, evaluated in order:

- **A. Required Readiness** (§D, Gate 1) — are all `required` slots deterministically ready?
- **B. Acquisition Need** (§D, Gate 2) — is there a MODEL acquisition need?

Determination:

| Required readiness | Acquisition need (dimension B) | Strategy |
|---|---|---|
| all `required` deterministically ready (SYSTEM_BINDER/TOOL_DEFAULT) | none | **CONTEXT_DIRECT** |
| — | MODEL need, and the current query alone provides the needed MODEL evidence | **QUERY_TO_QWEN** |
| — | MODEL need, and the needed info depends on the last-5 user turns | **QUERY_PLUS_5_USER_TURNS** |
| — | system-sourced params AND a MODEL acquisition both present | **MIXED** |
| `required` args cannot be legally obtained from the allowed sources | — | **MISSING / INVALID** |

No linear precedence such as `MISSING > CONTEXT_DIRECT > MIXED > …` is defined. Ordering
between QUERY_TO_QWEN and QUERY_PLUS_5_USER_TURNS is not a precedence ladder — it is the
`escalation`/evidence determination (§E, §I).

**Model-slot source is evidence-driven — never inferred from `allowed_sources`.**
For every MODEL-owned slot that must be acquired, WHERE its value comes from is decided by
the presence of the **actual** evidence signal, never by what the declaration merely
*allows*:

- query evidence present (source `QUERY`, allowed) → **QUERY_TO_QWEN**;
- history evidence present (source `CONVERSATION_5_USER_TURNS`, allowed) →
  **QUERY_PLUS_5_USER_TURNS**;
- neither evidence present → the slot's source cannot be legally determined →
  **MISSING / INVALID** (§D); never a guess, never a legacy fallback.

`allowed_sources` gates whether a given evidence source is ACCEPTABLE; it never CREATES
evidence. A slot whose `allowed_sources` contains `QUERY` does **not** thereby have query
evidence.

**Merge provenance is the ACTUAL source — never inferred from `allowed_sources`.**
A system-owned slot (`SYSTEM_BINDER` / `TOOL_DEFAULT`) entering a `MIXED` merge records the
**actual** `source` and value it was resolved from, supplied to the merge as inputs
(`system_values` + `system_sources`). A declaration's `allowed_sources[0]` is **not** a
legal provenance source and MUST NOT be used as a fallback. A system slot with no actual
source/value produces **no** provenance record and is **not** merged — never fabricated
(§D). MODEL-owned provenance likewise records the **actual** acquisition source (query or
last-5 user turns), never a merely-allowed one.

Every strategy is derived deterministically from `(parameters, acquisition, turn facts,
query, recent-5-user-turns)`.

### Per-strategy detail (input / output / forbidden)

| Strategy | Input (precondition) | Output | Forbidden |
|---|---|---|---|
| **CONTEXT_DIRECT** | All `required` slots are deterministically satisfied (SYSTEM_BINDER/TOOL_DEFAULT) **and** no MODEL acquisition need | A fully-determined args draft → existing `certified()` / execution handoff | **Any Qwen call (Qwen call = 0)**; any Recall/aggregation/selection |
| **QUERY_TO_QWEN** | MODEL acquisition need satisfiable from the **current query alone**; no system-sourced slots to merge | Qwen returns MODEL-owned args → **Binder validates** → certified | Sending non-MODEL slots to Qwen; sending history; >1 Qwen call by default |
| **QUERY_PLUS_5_USER_TURNS** | MODEL acquisition need requires recent conversation context | Qwen input = **exactly** current query + last **5** `role=="user"` messages → Binder → certified | Sending assistant/system/Agent history; sending the full transcript; >1 Qwen call by default |
| **MIXED** | MODEL acquisition need **and** ≥1 non-MODEL slot filled by a system source (SYSTEM_BINDER/TOOL_DEFAULT) | **merge** system values + Qwen values → Binder/validate → certified, **provenance preserved per slot** | Binder before merge; dropping provenance; letting Qwen author a system-sourced slot |
| **MISSING / INVALID** | No legal value obtainable from query / last-5 user turns / UI-context / resolver / callback / default | **Agent / follow-up** (exit) | **Guessing / fabrication**; a partial certified action |

Inside `MIXED`, the Qwen **bundle** still obeys the query-vs-query+5 rule above.

---

## C. Acquisition declaration — data structure

A NEW Registry entry (working name `acquisition`), shape `{slot_name: SlotDecl}`:

```jsonc
{
  "asset_id": { "ownership": "SYSTEM_BINDER",
                "allowed_sources": ["UI_CONTEXT", "RESOLVER"] },
  "pages":    { "ownership": "MODEL",
                "allowed_sources": ["QUERY", "CONVERSATION_5_USER_TURNS"],
                "escalation":    ["QUERY", "CONVERSATION_5_USER_TURNS"] }
}
```

`SlotDecl` fields: `ownership` (required, no default) / `allowed_sources` / `escalation`.
It carries **no** `required`/`type`/`enum`/`description`/`max_len` — those live in
`Registry.parameters` alone (no duplication).

**Existing invariant (already true in code):**

- **I4** `allowed_sources` never crosses ownership boundaries — one table, no drift.
  (This invariant already exists in `argument_acquisition/contract.py`; it is restated
  here, not newly added.)

**RESOLVED — approved scope of the declaration (Step 0):**

- the `acquisition` entry is the **opt-in acquisition contract**;
- a capability **with** a declaration MAY enter the Argument Path Router;
- a capability **without** one MUST NOT fall back to legacy `select_and_extract`, and exits
  to the approved acquisition-undeclared → Agent semantics (§G);
- whether the declaration must cover every `required` parameter is **not** an additional
  state in Step 0 — **no** “incomplete declaration” status or reason is defined.

**RESOLVED — a `required` parameter with NO declaration entry has no legal acquisition
source.** Because ownership (and therefore any allowed source) is undeclared, nothing can
legally fill the slot: it is treated as `UNAVAILABLE` → Gate 1 `UNSAT` → **MISSING /
INVALID** → Agent / follow-up (§D). It is **never** resolved by falling back to the legacy
`select_and_extract`, and never guessed. This is **not** an additional “incomplete
declaration” reason: no separate status is introduced — the outcome is the ordinary
`MISSING` path.

### Ownership derivations (approved)

- **MODEL** → Qwen MAY obtain it; sources limited to `QUERY` / `CONVERSATION_5_USER_TURNS`.
- **SYSTEM_BINDER** → obtainable ONLY from deterministic system sources —
  `UI_CONTEXT` / `RESOLVER` / `CALLBACK_CONTEXT`. **Never Qwen.**
- **TOOL_DEFAULT** → obtainable ONLY from the tool's default (`DEFAULT`).
- **UNAVAILABLE** → obtainable from nothing; **never fabricated**.

---

## D. Readiness — the double gate

`Readiness` result fields: `ready` / `needs_acquisition` / `unsatisfiable`.

**Gate 1 — Required Readiness.** Iterate slots where `parameters[slot].required == True`:
- `SYSTEM_BINDER` → **must** have a real deterministic value now (from turn facts);
  value present → satisfied; absent → **UNSAT**.
- `TOOL_DEFAULT` → must resolve to the tool default; else **UNSAT**.
- `MODEL` → **PENDING_MODEL** (deferred to Gate 2 / acquisition).
- `UNAVAILABLE` (and required) → **UNSAT**.
- `ready == True` ⇔ no required slot is `PENDING_MODEL` and none is `UNSAT`.

**Gate 2 — Acquisition Need.** Over MODEL-owned slots:
- **required** MODEL slot → `needs_acquisition = True` (MUST);
- **optional** MODEL slot → `needs_acquisition = True` **iff** an evidence signal exists
  (§E); with no evidence it contributes **nothing**.
- An optional MODEL slot that merely *exists in the schema* MUST NOT trigger Qwen.

**Required MODEL slot — its source is evidence-driven (§B).** A required MODEL slot always
sets `needs_acquisition = True`, but WHERE its value comes from (`QUERY` vs
`CONVERSATION_5_USER_TURNS`) is decided by the **actual** evidence signal, never by
`allowed_sources`. If neither query nor history evidence is present — or the present
evidence's source is not in the slot's `allowed_sources` — the slot's source cannot be
legally determined → **MISSING / INVALID** (§C), never a guess, never a legacy fallback.

`unsatisfiable` = the required slots nothing can fill (Gate 1 `UNSAT`) → `MISSING` → Agent/follow-up.

### Worked examples

- **“打开这个文件”** — `asset_id` is provided by system context; `pages` is an **optional
  MODEL** slot with **no** query evidence. Gate 1: required satisfied. Gate 2: no need.
  → **CONTEXT_DIRECT** → **no Qwen call.**
- **“打开这个文件第 3 页”** — `pages` now has explicit query evidence (`第 3 页`).
  → **acquisition needed** → **must NOT be CONTEXT_DIRECT**; goes through Qwen.

---

## E. MODEL evidence — a router input (the detector is NOT implemented in Step 1)

**RESOLVED — Step 0 locks only these properties:**

- Evidence MUST be **deterministic**;
- Evidence MUST be **schema-aware**;
- Evidence MUST be **non-LLM** — deciding “call Qwen or not”, and “from where”, is never
  delegated to an LLM;
- Evidence is a **router input**: it is produced OUTSIDE the router by a deterministic,
  schema-aware, non-LLM producer and injected as `{slot: source}`. The router never computes
  evidence itself and never reads a DB or a parser.
- Evidence is used for **two** purposes:
  1. to decide whether an **optional** MODEL slot has a **real acquisition need** (Gate 2, §D);
  2. to decide the **source / path** of a required MODEL slot — query vs query + last-5 user
     turns (§B) — so a required MODEL slot with no determinable source becomes `MISSING`.
- **Default is conservative:** no evidence → no Qwen need for an optional slot (the slot is
  simply omitted, never fabricated); for a required slot → the source cannot be determined →
  **MISSING / INVALID**.
- **Step 1 does NOT implement the detector.** Evidence is an injected external input; the
  real producer and the orchestrator wiring land in **Step 2**. Step 0 / Step 1 specify **no**
  further implementation detail — no DB access, no latency budget, no matcher / regex /
  parser shape.

---

## F. Qwen input / output schema boundary (approved)

- Qwen receives a schema containing **only MODEL-owned slots** (their `parameters`
  entries). `SYSTEM_BINDER`, `TOOL_DEFAULT`, `UNAVAILABLE` slots are **excluded**.
- **`asset_id` (SYSTEM_BINDER) never appears in the Qwen schema or output.**
- Qwen output is a **draft**; the Binder remains the gate. Any non-MODEL slot appearing in
  the draft is ignored/stripped — the system value is truth, in both directions.
- Qwen input bundle: current query only, **or** current query + last 5 `role=="user"`
  messages (never assistant/system/Agent history, never the full transcript).

---

## G. Undeclared capability on the new lane

- `backend ∈ {stub, laya}` and the selected capability has **no** `acquisition`
  declaration → **Agent** with an acquisition-undeclared reason.
- **Never** fall back to legacy `ToolIntentModel.select_and_extract` on the new lane.
- `backend=off` is exempt (it IS the legacy lane).

**RESOLVED — semantics only (constant naming is implementation detail):**

- The approved business semantic is: **undeclared acquisition → Agent**, surfaced as an
  “acquisition-undeclared” reason.
- The full set of reason constants and their exact names are **implementation detail**, not
  part of this business contract (illustrative only, not locked: `ACQUISITION_UNDECLARED`).

Phase 4 **removes** `REASON_CAP_ROUTER_HIT_DEFERRED` (transitional; the ARP replaces it).

---

## H. `backend=off` vs `backend=stub|laya` (approved boundary)

| | `backend=off` (legacy / rollback) | `backend=stub|laya` (new lane) |
|---|---|---|
| HIT | legacy `select_and_extract` | **ARP** (no Recall/aggregation/cap_router/re-selection) |
| MISS/AMBIGUOUS | legacy `select_and_extract` | Recall → Aggregation → cap_router → **ARP** |
| Acquisition | n/a (legacy extraction inside select_and_extract) | ARP per §B |
| Undeclared cap | n/a | Agent (acquisition-undeclared) |
| `tool_intent.select_and_extract` | **reachable (rollback)** | **never called** |

The two lanes MUST stay physically distinguishable (CLAUDE.md rule 8).

---

## I. Escalation discipline & HIT invariant (approved)

- **No double Qwen by default.** Query-only scenarios: **at most 1** Qwen call. Turns
  already known to need history: query + last-5 user turns, **at most 1** Qwen call.
- **Escalation** to query + last-5 user turns is allowed **only when the query-only
  acquisition is proven insufficient** — never “for safety.”
- **Never send the full history** by default.
- **HIT invariant** (§A.10): HIT → ARP directly; no Recall, no Aggregation, no cap_router,
  no re-selection.

---

## J. Deliverables, deferred work, and resolved decisions

**Delivered (Step 0):** this contract + `tests/test_phase4_acquisition_contract.py`
(green vocabulary-invariant tests + doc markers).

**Deferred to Step 1+ (production):** the `acquisition` declaration entry; the Qwen extractor;
the orchestrator wiring; `selection_transition.py` removal;
`REASON_CAP_ROUTER_HIT_DEFERRED` removal. The deterministic ARP modules
(`path_router.py` / `context_bundle.py` / `merge.py`) are delivered as pure functions in
**Step 1**; the **evidence producer** and the **system value / source resolution** that feed
them are external inputs wired in **Step 2** — Step 1 implements neither.

**Resolved decisions (formerly open):**
1. **Strategy determination** — two dimensions (Required Readiness, then Acquisition Need),
   **no** global linear precedence (§B).
2. **Declaration completeness** — Step 0 defines only the opt-in contract and the
   undeclared → Agent semantics; **no** “incomplete declaration” state or reason (§C).
3. **Reason codes** — only the acquisition-undeclared **semantics** is locked; constant
   names are implementation detail (§G).
4. **Evidence** — locked only as deterministic / schema-aware / non-LLM, and as a **router
   input** with **two** approved uses: a **real** optional-MODEL acquisition need AND the
   **source / path** of a required MODEL slot. No DB / latency / matcher / regex / parser
   detail; the detector (producer) and the wiring are deferred to Step 2 (§E).
5. **Required parameter without a declaration entry** — no legal source → treated as
   `UNAVAILABLE` → `MISSING / INVALID`; never a legacy fallback and never a guess (§C).
6. **Provenance source** — the **actual** source/value supplied as inputs; a declaration's
   `allowed_sources[0]` is **never** a fallback, and an unprovenanced system slot is **not**
   merged (§B).

**Future test checklist (Step 1+) — to be written only when the modules exist.** Recorded
here as a list on purpose; **not** implemented as `skip`ped pytest tests in Step 0:
- Argument Path Router selection per strategy (CONTEXT_DIRECT / QUERY_TO_QWEN /
  QUERY_PLUS_5_USER_TURNS / MIXED / MISSING);
- the Readiness double gate (Gate 1 required readiness; Gate 2 acquisition need);
- optional-MODEL evidence vs. mere schema presence;
- the Qwen schema boundary (only MODEL-owned slots; `asset_id` excluded);
- the undeclared-capability rule on the new lane (`backend ∈ {stub, laya}`);
- `backend=off` still takes the legacy chain (byte-identical).

**Naming note:** `tests/test_funnel_p4.py` belongs to the OLD P0–P4 funnel numbering —
it is **not** this Phase 4.
