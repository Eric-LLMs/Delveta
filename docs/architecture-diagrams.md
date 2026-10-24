# Architecture Diagrams

Per-module architecture diagrams and their mermaid source (for editing / regeneration via mermaid.ink).

![Architecture overview — frontends, backend, data, and model services](./images/architecture-overview.png)

<details>
<summary>Mermaid source (for editing — regenerate via mermaid.ink)</summary>

```mermaid
flowchart LR
    subgraph Frontends
        web[React Web UI]
        desk[Electron Workbench]
        lt[Local Tools]
    end
    subgraph Backend
        api["FastAPI gateway · AgentKernel<br/>HTTP/SSE entry · usecases · job enqueue"]
        wk[arq worker<br/>TTS / images / explanations / finalize]
        rs[Retrieval service<br/>RAG pipeline · gRPC]
    end
    subgraph Data
        pg[(PostgreSQL<br/>pgvector + tsvector)]
        rd[(Redis<br/>cache / queue)]
    end
    subgraph Model services
        emb[TEI embedding · BGE-M3]
        tts[Kokoro TTS]
        llm[LiteLLM gateway<br/>→ upstream LLM]
    end

    web --> api
    desk --> api
    api --> rd
    rd --> wk
    api --> pg
    wk --> pg
    api --> rs
    rs --> pg
    api --> emb
    api --> tts
    api --> llm
    wk --> tts
    rs --> emb
```

</details>

**Agent kernel** — the core, composed in and running inside the FastAPI gateway process:

![Agent kernel — the ReactLoopAgent composition root](./images/agent-kernel.png)

<details>
<summary>Mermaid source (for editing — regenerate via mermaid.ink)</summary>

```mermaid
flowchart TB
    kernel[AgentKernel] --> loop[ReactLoopAgent step loop]
    loop --> prompt[CacheBoundaryAssembler<br/>cache-boundary prompt]
    loop --> tools[ToolGateway<br/>deferred tool loading]
    loop --> skills[SkillCatalog · SKILL.md]
    loop --> memory[MemoryService<br/>PG tsvector + pgvector RRF]
    tools --> sandbox[Sandbox<br/>READ / WRITE / NETWORK]
```

</details>

> Agent kernel — design: [architecture.md §5 Agent Module](architecture.md#5-agent-module).

**Answering flow** — each turn runs the `ReactLoopAgent` step loop, which decides what to retrieve and
composes the reply. It calls the **`rag_search` tool** (runs the RAG node pipeline and returns `top_k`
chunks — the retrieval layer described in the RAG section) and, only when local material is
insufficient, the **`web_search` tool** (duckduckgo / tavily / bing / google) for up-to-date external
results; it then composes the answer from both.

**Memory — two independent tracks, orchestrated by `MemoryService`** (writes go to the local
memdir, recall reads the database; the tracks never cross):

![Memory architecture — two independent tracks](./images/memory-architecture.png)

<details>
<summary>Mermaid source (for editing — regenerate via mermaid.ink)</summary>

```mermaid
flowchart TB
    %% Invariants: thinking never persisted · messages never deleted by compaction/retention ·
    %% memory_save writes the memdir only (never PG) · vector-channel failure → tsvector-only (never silent empty) ·
    %% superseding marks a memory, never deletes it (audit file kept) · cosmetic failures never block the main flow

    P[prompt memory section]

    subgraph trackA["Track A · local file memory — agent-writable, never the DB"]
        FILES[MEMORY.md index<br/>one frontmatter .md per memory<br/>type: user · feedback · project · reference<br/>superseded entries unindexed · files kept as audit]
        WRITE[memory_save · guardrailed write<br/>kebab-case · ≤4000 chars · type whitelist · write-then-readback<br/>importance 1-10 · supersede-in-place via status]
        SEARCH[memory_search · keyword × importance<br/>name ×3 · desc ×2 · body ×1<br/>superseded entries excluded]
        BRIEF[begin_session → MEMORY.md head<br/>first 200 lines as session brief · Lane-1 always on]
        WRITE --> FILES
        SEARCH --> FILES
        BRIEF --> P
    end

    subgraph trackB["Track B · session memory — PostgreSQL, system-written · agent read-only"]
        MSGS[(messages<br/>text · pgvector · tsvector · created_at)]
        SESS[(sessions<br/>title · summary · closed_at · type · compaction JSONB<br/>0=chat 1=research, hidden from sidebar)]
        EVTS[(session_events<br/>audit log · JSONB payload<br/>compaction summaries persist here)]
        KW[tsvector keyword recall<br/>to_tsvector english<br/>fts_config swappable for CJK]
        VEC[pgvector semantic recall]
        RRF[RRF fusion + recency decay<br/>30-day half-life · 1.0× recent<br/>0.68× at 30d · 0.55× at 90d]
        COMPACT[apply_compaction · dual persistence barrier<br/>over 40 msgs → one fold of raw SQL rows<br/>5-section summary · inclusive watermark CAS<br/>failure defers · never trims]
        FINAL[worker finalize<br/>backfill embeddings · first-time summary/title only<br/>never re-summarizes · cosmetic failures never block]
        SWEEP[retention cron · scheduled daily<br/>purge session_events > 30d<br/>audit log only · messages + checkpoints kept]
        MSGS --> KW
        MSGS --> VEC
        KW --> RRF
        VEC --> RRF
        MSGS --> COMPACT
        MSGS --> FINAL
        COMPACT --> P
        COMPACT --> EVTS
        EVTS --> SWEEP
        FINAL --> SESS
    end

    USER[user message]
    GATE[trigger gate · should_recall<br/>memory-seeking phrases or short elliptical<br/>else only the Lane-1 brief]
    USER --> GATE
    GATE -- seeking --> RRF
    GATE -- neutral --> P
    RRF --> P
    P --> AGENT[ReactLoopAgent step loop]
```

</details>

> **Memory invariants** — thinking is never persisted; `messages` are never deleted by compaction
> or retention; `memory_save` writes the memdir only (never the DB); a vector-channel failure
> degrades to tsvector-only (never a silent empty); superseding marks a memory `superseded` and
> keeps the file as an audit trail — it is never deleted; the recall gate skips only the deep
> recall on non-memory-seeking turns (the Lane-1 brief always injects); cosmetic failures
> (summary / title) never block the main flow; the session context rides the client's Live State,
> so a normal turn reads no SQL and a deferrable compaction never trims.
> Design: [architecture.md §5.4 — Memory, skills, sessions](architecture.md#54-memory-skills-sessions)
> and [§22 — Chat Session Memory v2](architecture.md#22-chat-session-memory-v2--client-live-state-authority--zero-read-turns).

**Prompt — cache-boundary assembly, compression, and deferred tool stubs** (architecture, design
features, and per-step process logic):

![Prompt — cache-boundary assembly, compression, and deferred tool stubs](./images/prompt-architecture.png)

<details>
<summary>Mermaid source (for editing — regenerate via mermaid.ink)</summary>

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 500, "nodeSpacing": 40, "rankSpacing": 50}}}%%
flowchart TB
    %% Prompt module: cache-boundary assembly + compression + deferred tool stubs.
    %% Invariants: the CACHE_BOUNDARY separator is internal-only and never rendered; the stable
    %% head (STATIC_PREFIX + PROJECT_CONTEXT) is byte-identical so the provider prefix cache is
    %% reused; only the DYNAMIC_SUFFIX re-renders per step and is skipped when unchanged; the
    %% per-message snip trims the request snapshot only, keeping the persistence copy raw; mounted
    %% tools appear as stable defer_loading stubs, with the full schema riding in the tool_search
    %% result.

    subgraph input["Input"]
        HIST["chat history + new user message"]
        AUTO["apply_compaction · /chat boundary<br/>over 40 msgs or char budget → fold raw rows<br/>inject one 5-section leading summary<br/>dual barrier · failure defers, never trims"]
        HIST --> AUTO
    end

    subgraph zones["Design · cache-boundary zones"]
        SP["STATIC_PREFIX · byte-identical across requests<br/>SOUL.md identity + full tool catalog<br/>+ complete skill catalog · never truncated<br/>(overflow → startup refused, 16 KB ceiling)"]
        PC["PROJECT_CONTEXT · stable per project<br/>DELVETA.md via read_project_context<br/>capped at 8k chars · empty → zone dropped"]
        DS["DYNAMIC_SUFFIX · re-rendered per step<br/>memory brief · Lane-1 always on<br/>recalled memory · gated by should_recall<br/>+ agent.inject content"]
        SP --> HEAD["stable head"]
        PC --> HEAD
    end

    subgraph render["render_prompt"]
        HEAD --> RENDER
        DS --> RENDER
        RENDER["zones joined by blank lines<br/>CACHE_BOUNDARY is internal-only · never sent"]
        RENDER --> KEY["snapshot_key · sha256 of static + project<br/>16 hex · cache identity measurable"]
    end

    subgraph perstep["Process · per step"]
        SNIP["per-message snip · prompt_message_max_chars<br/>request snapshot trimmed · persistence keeps full text"]
        AUTO --> SNIP
        SNIP --> REQ["LLM request · system + snipped messages"]
        REQ --> TOOLS["visible tools · core full schemas<br/>+ defer_loading stubs · name + description · empty params<br/>full schema via the tool_search result"]
        REQ --> EXEC["tool execution · dispatch via runtime<br/>concurrency-safe → parallel gather · else serial<br/>Sandbox permission-guarded"]
        EXEC --> COMMIT["tool result committed to messages<br/>role:tool + deferred contexts<br/>messages grows · next step reuses"]
        COMMIT --> REFRESH["refresh_dynamic · recompute only DYNAMIC_SUFFIX<br/>unchanged → system not re-sent · head reused"]
        REFRESH --> REQ
        RENDER --> REQ
    end
```

</details>

> **Prompt invariants** — the `CACHE_BOUNDARY` separator is internal-only and never rendered to the
> model; the stable head (STATIC_PREFIX + PROJECT_CONTEXT) is byte-identical across requests so the
> provider prefix cache is reused; only the DYNAMIC_SUFFIX re-renders per step and is skipped when
> unchanged; the per-message snip trims the request snapshot only, keeping the persistence copy raw;
> mounted tools appear as stable defer_loading stubs, with the full schema riding in the tool_search
> result.
> Design: [architecture.md §16 — Prompt Module](architecture.md#16-prompt-module).

**RAG & Query Repository — a config-driven, plug-and-play retrieval pipeline** over one unified search
corpus. A single query searches your cloud-drive files, Learning-Platform material, and chat Q&A
together. The pipeline's node list *is* the retrieval topology: compose, reorder, toggle, and tune every
stage from the admin console — live, no code, no restart. Feature walk-through in
[**Key Features**](docs/features.md); the design detail lives in
[§10 RAG Module (architecture.md)](architecture.md#10-rag-module-config-node-pipeline).

![RAG — config-driven node pipeline](./images/rag-architecture.png)

<details>
<summary>Mermaid source (for editing — regenerate via mermaid.ink)</summary>

```mermaid
flowchart TB
    %% ── ① sources → per-entry processing ───────────────────────────
    subgraph ENTRY["Three import entries"]
        DRIVE["Cloud Drive<br/>doc · pdf · txt · docx"]
        LEARN["Learning Platform<br/>sentences · articles"]
        CHAT["Chat<br/>reply pairs · whole sessions"]
        PDFT["PDF tools<br/>page text + tables → image → vision LLM text"]
        LEARNP["learning_import job<br/>sentences / articles → chunks"]
        CHATP["chat import<br/>LLM merges one question's turns · splits distinct Qs"]
        DRIVE --> PDFT
        LEARN --> LEARNP
        CHAT --> CHATP
    end

    %% ── ② query repository ────────────────────────────────────────
    subgraph REPO["Query Repository"]
        CHUNK["chunks table<br/>source_type file · learning · chat<br/>source_id · asset_id nullable · owner = importer"]
        EMBED["build_chunks + BGE-M3 embed → pgvector<br/>leaf-only · chunked per runtime config"]
        PDFT --> CHUNK
        LEARNP --> CHUNK
        CHATP --> CHUNK
        CHUNK --> EMBED
    end

    %% ── ③ retrieval node pipeline ─────────────────────────────────
    subgraph RETR["Retrieval — config-driven node pipeline"]
        Q["user query · top_k<br/>+ filters.user_id · tenant binding<br/>owner / workspace / ACL · guest → public links"]
        QC["QueryCache<br/>Redis · key(query, filters, top_k,<br/>config_version, corpus_version)<br/>pure accelerator · failure falls through"]
        REWRITE["QueryRewriter<br/>multi-query expansion + HyDE answer"]
        VEC["VectorRecaller<br/>pgvector cosine · BGE-M3 via TEI<br/>source-aware · LEFT JOIN assets"]
        KW["KeywordRecaller<br/>tsvector FTS · CJK via jieba<br/>source-aware · READY only for files"]
        RRF["rrf_fusion · k=60<br/>fuses cross-scale scores"]
        CE["CrossEncoderReranker<br/>BGE-reranker · optional"]
        PE["ParentExpandNode<br/>leaf → parent · optional"]
        CRG["CrgCheckNode<br/>LLM relevance gate · optional"]
        OUT["top_k SearchHits<br/>ground the chat answer"]
        FB["rag_feedback<br/>query · hits · rating → golden eval set"]
        Q --> REWRITE
        Q -. cache hit → .-> QC
        QC -.-> OUT
        REWRITE --> VEC
        REWRITE --> KW
        VEC --> RRF
        KW --> RRF
        RRF --> CE
        RRF -. no reranker .-> OUT
        CE --> OUT
        RRF -. optional extension .-> PE
        PE --> CRG
        CRG --> OUT
        OUT -. rated → .-> FB
    end

    EMBED -. source-aware recall<br/>leaf-only · LEFT JOIN assets .-> VEC
    EMBED -. source-aware recall .-> KW
    EMBED -. ingest / re-index<br/>bumps corpus_version .-> QC

    %% ── ④ RAG pipeline config ─────────────────────────────────────
    subgraph CFG["RAG Pipeline Config · admin console RAG module"]
        NODES["Nodes<br/>add / remove / reorder / toggle<br/>per-node params · live, no restart"]
        CHUNKCFG["Chunking<br/>size / overlap · contextualization"]
    end
    NODES -. drives topology .-> RETR
    CHUNKCFG -. drives chunking .-> EMBED
```

</details>

**Chat control plane — a turn's full path from transport to a terminal answer.**
`resolve_plan` keeps only the lifecycle: the pure L0 pass runs first, then a SINGLE
`intent_funnel.route()` call owns all routing (shadow hook + the Intent-Funnel
single-hop cascade — the legacy QIR lane was retired with migration 0014), and
`build_execution_plan` is the sole
policy mapper. Every PlanKind has a registered executor, every certified fast path
dispatches through the ONE shared execution waterfall, and every uncertifiable or
pre-commit-failing turn falls back to the Agent byte-identical. C1–C4 name the failure
classes (see the design section).

![Chat control plane — three-stage plan resolution, per-kind executors, and the ONE shared waterfall](./images/chat-control-plane.png)

<details>
<summary>Mermaid source (for editing — regenerate via mermaid.ink)</summary>

```mermaid
flowchart TB
    %% Invariants: fast-path fallbacks are byte-identical · the Agent is never a recovery channel
    %% for C2/C3/C4 · the commit point (first content delta) locks the channel · gates default OFF.

    subgraph transport["Transport — apps/api/routers/chat.py (unchanged)"]
        AUTH["POST /chat · /chat/stream<br/>auth · quota · channel pin · build_turn_context"]
    end

    subgraph resolve["TurnOrchestrator.resolve_plan — lifecycle only; routing absorbed into intent_funnel.route() (P0 move)"]
        L0["Source facts · resolve_source_facts<br/>private_only / external_ok · no model call"]
        RT["intent_funnel.route(ctx, deps, requirements)<br/>single path · no rollout switch"]
        FN["Intent Funnel — two lanes, one chain<br/>LIVE Registry tables → Matcher (exact-only) →<br/>HIT: Argument Acquisition → Binder<br/>MISS/AMBIGUOUS: Recall → Aggregation → cap_router →<br/>Argument Acquisition → Binder validate<br/>certified ACTION {capability_id · fingerprint TOCTOU stamp}"]
        MAP["build_execution_plan — sole policy mapper<br/>per-kind gates · sole-demand eligibility · source_policy"]
        L0 --> RT
        RT -- "funnel_live: deps wired (no entry veto; negation forces HIT→MISS)" --> FN --> MAP
        RT -- "abstain — the ORIGINAL requirements object" --> MAP
    end

    AUTH --> L0
    FN -. "ABSTAIN — NO_CANDIDATE · CAP_ROUTER_NONE / REJECT / UNAVAILABLE ·<br/>ACQUISITION_UNDECLARED / MISSING / MODEL_PENDING · DEFERRED ·<br/>POLICY_CONFLICT / HANDLER_MISSING ·<br/>BIND_MISSING / AMBIGUOUS / INVALID · REGISTRY / RECALL_UNAVAILABLE ·<br/>VERSION_MISMATCH · KIND_DISABLED · CASCADE_TIMEOUT — byte-identical" .-> AG

    subgraph gates["Feature gates (config.py)"]
        GQ["funnel: single path — funnel_live = deps wired<br/>(no rollout switch; the legacy chat_qir_* set was<br/>deleted with migration 0014)"]
        GF["fast_paths_enabled (master switch)<br/>+ per-kind: direct · viewer · retrieval<br/>· action (5A) · composite (5B)<br/>closed ⇒ that PlanKind never maps"]
    end
    GQ -. "single path: always live when deps are wired" .-> RT
    GF -. "gates build_execution_plan · all closed ⇒ every turn maps to AGENT" .-> MAP

    MAP --> K{"PlanKind"}
    K -- DIRECT --> DEX["DirectExecutor — single LLM answer"]
    K -- VIEWER --> VEX["ViewerExecutor — answer from injected screen text"]
    K -- LOCAL_RAG --> REX["RetrievalExecutor — staged private-corpus answer"]
    K -- ACTION --> AEX["ActionExecutor — deterministic one-shot dispatch"]
    K -- COMPOSITE --> CEX["CompositeExecutor — fixed grounded-read → task chain"]
    K -- "AGENT · unmapped kinds (structural Fail-Closed: no WEB executor exists)" --> AG["AgentExecutor — ReactLoopAgent step loop"]

    subgraph governance["ONE shared execution waterfall — ToolRuntime.execute (no second authority)"]
        RW["pre-execute ASK → approval bridge → sandbox / source-policy guards → tool body"]
        RW -- "rule-level DENY · approval denied / timeout ⇒ C3" --> TD["TERMINAL — decided-denial message"]
        RW -- "body entered then raised ⇒ C4" --> TU["TERMINAL — STATE_UNKNOWN · never blind-retry"]
        RW -- ok --> OUT["tool output"]
    end

    AEX -- "stage 0 funnel_registry_version (fingerprint) re-validation (stale ⇒ C3) · stage 0.5 binding_integrity (C2 ⇒ TERMINAL) · stage 1 schema (C1 ⇒ Agent) · then the _run_tool seam = side-effect boundary" --> RW
    CEX --> RW
    AG -- "every agent tool call" --> RW
    REX -. "insufficient evidence (pre-commit) + honest-disclosure note" .-> AG

    subgraph stream["Response — SSE frame contract unchanged"]
        OUT --> DONE["content → done · user + assistant rows persisted"]
        TD --> DONE
        TU --> DONE
        DEX --> DONE
        VEX --> DONE
        REX --> DONE
        AG --> DONE
        DONE -.- CP["Commit Point = first content delta<br/>before it EscalateToAgent is legal (zero pollution)<br/>after it: terminate only, never re-route"]
    end
```

</details>

> **Chat control-plane invariants** — a fast path takes only turns it can certify, and its demand
> must be the *sole* demand; abstention leaves the user text byte-identical; the routing chain
> (Intent-Funnel cascade over the LIVE Registry tables) produces routing metadata only —
> capability + fingerprint stamp — and never executes or binds at the dispatch side; live-table
> writes are atomic text+embedding and the
> stamped fingerprint is re-validated before dispatch; C2/C3/C4 are terminal by contract — the
> Agent clarifies user incompleteness (C1) and is never the recovery channel for system faults,
> decided denials, or mid-write uncertainty.
> Design: [architecture.md §24 — Chat Control Plane](architecture.md#24-chat-control-plane--plan-resolution-fast-paths--intent-routing),
> [§25 — Chat Intent Funnel](architecture.md#25-chat-intent-funnel--nodeized-routing-selector--shared-tool-runtime).

**Intent Funnel — two lanes, one chain, end to end.** `funnel.route` runs inside plan
resolution: the LIVE Registry tables load first, and the Matcher consults the curated query
corpus only (exact). A HIT is already decided, so it goes straight to argument acquisition and is
never re-selected. Every non-HIT runs Recall (candidates only, no adjudication), Candidate
Aggregation, then exactly ONE `cap_router` selection call (the deployed LayaChoice service; its
stub is a test double only). The Argument Path Router then decides HOW the chosen capability's
arguments are acquired, and the Binder only validates against the Registry schema.
Every abstain, denial or fault exits fail-open to the Agent byte-identical; execution is
never the funnel's — a certified turn dispatches through the same shared waterfall.

<details>
<summary>Mermaid source (for editing — regenerate via mermaid.ink)</summary>

```mermaid
flowchart TB
    %% Invariants: the funnel produces routing metadata only · every abstain/fault returns the
    %% ORIGINAL TurnRequirements object (byte-identical, zero pollution) · a HIT is never
    %% re-selected · one selector hop on every MISS/AMBIGUOUS turn · an off-card id is never a verdict.

    USER["chat turn · resolve_plan →<br/>funnel.route(ctx, deps, requirements)"]

    subgraph lane[" "]
        direction LR
        subgraph entry["Entry — single path"]
            direction TB
            GATE["funnel_live = deps wired (no rollout switch · no entry veto)<br/>every wired turn reaches the Matcher;<br/>guardrails.negated forces a would-be HIT to MISS"]
        end
        USER --> GATE

        subgraph cascade["Two lanes, one chain (wall-clock budget chat_funnel_timeout_seconds)"]
            direction TB
            REG["LIVE Registry tables (capabilities + query rows) —<br/>the routing truth, fingerprint-cached; Recall corpus index<br/>loads on the MISS lane<br/>missing ⇒ REGISTRY_UNAVAILABLE / RECALL_UNAVAILABLE"]
            M["Node 1 · Matcher — EXACT-only over the live query corpus<br/>(enabled Standard + Similar rows; patterns/aliases are inert<br/>legacy storage, never read — ruling 2026-09-25)<br/>negation guard before certifying (common layer)<br/>HIT · MISS · MATCH_AMBIGUOUS (one sentence under two<br/>caps — all claimants up)"]
            R["Node 2 · Recall — TWO independent vector searches<br/>(Standard + Similar rows, ONE query embedding)<br/>every hit ≥ min_score kept with provenance — quality gate,<br/>no merge/dedup, never adjudicates (origin=recall)"]
            CAND["Candidate Aggregation — ONE card per capability_id<br/>matcher_ambiguous 0.0 rides alongside recall · EMPTY set<br/>short-circuits to NO_CANDIDATE, zero selector hops"]
            SEL["Node 3 · cap_router — the ONE production selector<br/>(deployed LayaChoice service: K cards + frozen REJECT)<br/>SELECTED · NONE · REJECT; unavailable / timeout /<br/>illegal label ⇒ CAP_ROUTER_UNAVAILABLE, never a<br/>second model"]
            ACQ["Node 4 · Argument Acquisition (Argument Path Router)<br/>CONTEXT_DIRECT · QUERY_TO_EXTRACTOR ·<br/>QUERY_PLUS_5TURNS_TO_EXTRACTOR · MIXED · MISSING<br/>the extractor fills ONLY MODEL-owned slots ·<br/>then the kind gate"]
            B["Binder.validate — pure schema gate<br/>Registry canonical parameters · extracts nothing<br/>COMPLETE / MISSING / AMBIGUOUS / INVALID<br/>(non-COMPLETE exits straight to the Agent, BIND_*)"]
            REG --> M
            M -- "any non-HIT: MISS / AMBIGUOUS" --> R --> CAND --> SEL
            M -- "HIT — never re-selected" --> ACQ
        end
        GATE --> REG
    end
    style lane fill:none,stroke:none

    subgraph exits["Fail-open exits — reason codes carry the stage prefix"]
        direction LR
        AG["Agent fallback — ReactLoopAgent, full autonomy<br/>original query BYTE-IDENTICAL · never a<br/>side effect precedes the fallback"]
        REASONS["REGISTRY_UNAVAILABLE · RECALL_TIMEOUT /<br/>UNAVAILABLE · NO_CANDIDATE (empty set, zero hops) ·<br/>CAP_ROUTER_NONE / REJECT / UNAVAILABLE / TIMEOUT ·<br/>INTENT_FUNNEL_ALLOWLIST_REJECT ·<br/>ACQUISITION_UNDECLARED /<br/>MISSING / MODEL_PENDING / DEFERRED /<br/>POLICY_CONFLICT / HANDLER_MISSING · REGISTRY_VERSION_MISMATCH ·<br/>FUNNEL_KIND_DISABLED · BIND_MISSING / AMBIGUOUS / INVALID ·<br/>CASCADE_TIMEOUT / CASCADE_ERROR"]
        REASONS -.- AG
    end
    SEL -- "NONE / REJECT" --> AG
    POST["post-decision gates (orchestrator.py, before acquisition):<br/>entry still in the active table? · capability admitted by the Intent Funnel?<br/>— missing entry ⇒ registry mismatch · not admitted ⇒ Intent Funnel allowlist<br/>reject, the turn falls open to the Agent"]
    SEL -- "SELECTED" --> POST
    POST -. "REGISTRY_VERSION_MISMATCH / INTENT_FUNNEL_ALLOWLIST_REJECT" .-> AG
    POST --> ACQ
    ACQ -- "not COMPLETE" --> AG
    ACQ --> B
    B -- "not COMPLETE" --> AG

    CERT["Certified ACTION turn — NEW TurnRequirements<br/>requested_action {tool, args, capability_id,<br/>funnel_registry_version = Registry fingerprint<br/>(the single TOCTOU stamp), funnel_stage, funnel_kind}"]
    B -- COMPLETE --> CERT
    CERT --> EX["ActionExecutor._dispatch — the funnel executed NOTHING,<br/>the plan stage owns this: TOCTOU re-validate the stamped<br/>fingerprint / active entry / same tool / kind still ON<br/>(drift ⇒ decided TERMINAL, never an Agent retry) ·<br/>binding_integrity marker ⇒ TERMINAL · validate_action<br/>schema gate — malformed ⇒ proven pre-body, Agent clarifies"]

    subgraph runtime["ONE shared execution waterfall — no second authority"]
        RW["runtime.execute stages (fail-closed each):<br/>tools/pre-execute (approval ASK · sandbox ·<br/>source-policy) → monotonic guard →<br/>tools/execute (tool body) → tools/post-execute<br/>→ events / audit"]
    end
    EX --> RW
    AG -- "every agent tool call" --> RW

    OBS["funnel_trace + session event row (execution_mode-tagged)<br/>deepest stage · matcher · recall top · cap_router verdict ·<br/>acquisition strategy · final route · per-stage latency"]
    B -.- OBS
```

</details>

> **Intent-Funnel invariants** — nodes speak only through `contract.py` structures; the Matcher
> holds zero hardcoded business rules (safety vetoes are funnel-common-layer code); Recall filters,
> never decides; a Matcher HIT is never re-selected; exactly ONE `cap_router` selection call per
> non-HIT turn, and a selector failure is an `Unavailable` exit to the Agent — never a second model
> and never a fabricated verdict; capability determination and argument acquisition are separate
> concerns, so Qwen runs only when the Argument Path Router chose a MODEL strategy; every
> non-COMPLETE outcome leaves the Agent's input byte-identical; the funnel executes nothing —
> certified and agent turns traverse the same waterfall.
> Design: [architecture.md §25 — Chat Intent Funnel](architecture.md#25-chat-intent-funnel--nodeized-routing-selector--shared-tool-runtime).

## Parameter-processing flow

**English.** The diagram is the complete path from the user's query to an executed
tool or an Agent fallback. It shows the decision points the implementation owns:
capability admission, the per-capability acquisition strategy, acquisition
failure, the Binder contract gate, and the execution boundary. Every "generate a
structured context block" node publishes the turn-scoped record described in
§25.8; the record is routing context only and never grants a capability.

![Parameter-processing flow — English](./images/parameter-processing-flow-en.png)

<details>
<summary>Mermaid source (for editing — regenerate via mermaid.ink)</summary>

```mermaid
flowchart TD
    A["User submits query"] --> B["Determine capability_id"]
    B --> C{"Admitted by Intent Funnel allowlist?"}

    C -- No --> AG0["Route to Agent using existing semantics"]
    C -- Yes --> D["Dispatch parameter acquisition by capability_id"]

    D --> E{"Parameter acquisition strategy"}

    E -- "No arguments required" --> F["Construct valid empty parameter object"]

    E -- "Model-based extraction" --> I["Use local Qwen to extract permitted MODEL slots"]
    E -- "Hybrid strategy" --> J["Preserve existing parameters and extract remaining slots"]

    E -- "Rules / UI / existing context" --> G["Acquire parameters using corresponding rules"]
    E -- "Attachment required" --> H{"Attachment or context available?"}
    H -- Yes --> G
    H -- No --> X1

    G --> K{"Parameter acquisition successful?"}
    I --> K
    J --> K
    F --> L["Assemble final parameters"]

    K -- Yes --> L
    K -- No --> X1

    X1["Record acquisition failure status, error code, and reason"]
    X1 --> X2["Generate structured context block"]
    X2 --> AG1["Send to Agent fallback"]

    L --> M["Binder validates parameters"]
    M --> N{"Do parameters satisfy the capability contract?"}

    N -- No --> Y1["Record parameter mismatch reason"]
    N -- Yes --> O["VirtualExecutor / production execution"]

    Y1 --> Y2["Generate structured context block"]
    Y2 --> AG2["Send to Agent fallback"]

    O --> P{"Execution result"}
    P -- "Explicit parameter mismatch error" --> Y1
    P -- Success --> Q["Return result"]
    P -- "Other execution errors" --> R["Apply the corresponding error-handling policy and return an error message to the user"]

    classDef start fill:#DCEAFE,stroke:#3974C6,color:#17365D
    classDef decision fill:#FFF0CC,stroke:#C58A19,color:#654500
    classDef failure fill:#FCE2E2,stroke:#C64D4D,color:#7A2020
    classDef agent fill:#E9DDFB,stroke:#8355B5,color:#43276B
    classDef success fill:#DDF3E4,stroke:#39845A,color:#174D30

    class A,B,D start
    class C,E,H,K,M,N,P decision
    class X1,X2,Y1,Y2,R failure
    class AG0,AG1,AG2 agent
    class Q success
```

</details>

Reading the diagram against the code:

| Diagram node | Implementation |
|---|---|
| Determine `capability_id` | `intent_funnel/orchestrator.py::run_nodes` — Matcher HIT, or Recall → aggregation → cap_router |
| Admitted by the allowlist? | `registry/entry.py::is_intent_funnel_capability_allowed`; a reject publishes `ADMISSION_REJECT` |
| Dispatch by `capability_id` | `orchestrator.py::_acquisition_hop` reads `POLICIES` / `HANDLERS` by id — never infers from `parameters == {}` |
| Strategy branches | `NO_ARGS` / `RULE_CONTEXT` / `MIXED` / `MODEL` / `DEFERRED_AGENT` in `cap_handler/acquisition_policy.py` |
| Acquisition failed | `ACQUISITION_*` reason, kind `ACQUISITION_FAILURE`, stage `acquisition` |
| Binder gate | `binder.validate`; a miss publishes `CONTRACT_MISMATCH` at stage `binder` |
| Execution | `executors/action.py` — schema rejection publishes `CONTRACT_MISMATCH` at stage `executor`; a proven pre-body refusal publishes `EXECUTION_PREFLIGHT`; state-unknown is terminal and never replayed |
| Agent fallback | `publish_failure` → `turn_orchestrator.resolve_plan` → `AgentExecutor.run(context=…)` → `intent_funnel_reference` prompt section |

**中文。** 下图是从用户 query 到工具执行或 Agent 兜底的完整参数处理流程。图中保留了实现
真正拥有的决策点：能力准入、按 `capability_id` 分发的取参策略、取参失败、Binder 契约校验
以及执行边界。每个“生成结构化上下文信息块”节点都会发布 §25.8 描述的 turn 级记录；该记录
只是路由上下文，不授予任何能力。

![参数处理流程](./images/parameter-processing-flow-zh.png)

<details>
<summary>Mermaid source (for editing — regenerate via mermaid.ink)</summary>

```mermaid
flowchart TD
    A["用户提交 query"] --> B["确定 capability_id"]
    B --> C{"Intent Funnel 白名单准入？"}

    C -- 否 --> AG0["按现有语义转 Agent"]
    C -- 是 --> D["按 capability_id 分发取参规则"]

    D --> E{"取参策略"}

    E -- "无需参数" --> F["构造合法空参数对象"]

    E -- "需要模型抽取" --> I["本地 Qwen 抽取允许的 MODEL 槽位"]
    E -- "混合策略" --> J["保留已有参数并抽取剩余槽位"]

    E -- "规则 / UI / 已有上下文" --> G["按对应规则获取参数"]
    E -- "需要附件" --> H{"附件或上下文可用？"}
    H -- 是 --> G
    H -- 否 --> X1

    G --> K{"参数获取成功？"}
    I --> K
    J --> K
    F --> L["汇总最终参数"]

    K -- 是 --> L
    K -- 否 --> X1

    X1["记录取参失败状态、错误码与原因"]
    X1 --> X2["生成结构化上下文信息块"]
    X2 --> AG1["发送给 Agent 兜底"]

    L --> M["Binder 校验参数"]
    M --> N{"参数符合 capability 契约？"}

    N -- 否 --> Y1["记录参数不匹配原因"]
    N -- 是 --> O["VirtualExecutor / 正式执行"]

    Y1 --> Y2["生成结构化上下文信息块"]
    Y2 --> AG2["发送给 Agent 兜底"]

    O --> P{"执行结果"}
    P -- "明确的参数不匹配错误" --> Y1
    P -- 成功 --> Q["返回结果"]
    P -- "其他执行错误" --> R["按对应错误策略处理，并向用户返回错误信息"]

    classDef start fill:#DCEAFE,stroke:#3974C6,color:#17365D
    classDef decision fill:#FFF0CC,stroke:#C58A19,color:#654500
    classDef failure fill:#FCE2E2,stroke:#C64D4D,color:#7A2020
    classDef agent fill:#E9DDFB,stroke:#8355B5,color:#43276B
    classDef success fill:#DDF3E4,stroke:#39845A,color:#174D30

    class A,B,D start
    class C,E,H,K,M,N,P decision
    class X1,X2,Y1,Y2,R failure
    class AG0,AG1,AG2 agent
    class Q success
```

</details>

图中的节点与代码的对应关系：能力判定在 `run_nodes`（Matcher HIT，或 Recall → 聚合 →
cap_router）；准入判定是 `is_intent_funnel_capability_allowed`，拒绝时发布
`ADMISSION_REJECT`；取参分发在 `_acquisition_hop`，按 `capability_id` 读取 `POLICIES` /
`HANDLERS`，不从空参数推断策略；策略取值为 `NO_ARGS` / `RULE_CONTEXT` / `MIXED` /
`MODEL` / `DEFERRED_AGENT`；取参失败发布 `ACQUISITION_*`（kind `ACQUISITION_FAILURE`，
stage `acquisition`）；Binder 失败发布 `CONTRACT_MISMATCH`（stage `binder`）；执行器参数
门失败发布 `CONTRACT_MISMATCH`（stage `executor`），已证明未进入工具体的前置拒绝发布
`EXECUTION_PREFLIGHT`，执行状态未知则终止且绝不重放；结构化记录经
`turn_orchestrator.resolve_plan` 进入 `AgentExecutor.run(context=…)`，由
`intent_funnel_reference` 提示段落渲染。

> 说明：图中“规则 / UI / 已有上下文”对应第 1 类取参（确定性规则、Handler、可信上下文）。
> 文件类能力的 `pages` 不属于这一类，而是第 2 类的专用 `page_scope_resolver`（在线 LLM
> 只产出 `all` / `single` / `range`），由文件 Handler 在自己的 `acquire()` 内调用；非分页
> viewer 的跳过是 viewer kind 的结构化判断，不是用关键词规则替代模型分页语义。第 3 类是
> 共享 Qwen 抽取器。三类的职责边界、路由关系与失败处理见
> [phase4-acquisition-contract.md](phase4-acquisition-contract.md)。

> Design: [architecture.md §25.16 — Capability-Specific Acquisition Handlers](architecture.md#2516-capability-specific-acquisition-handlers).

> [architecture.md](architecture.md) is the single source of truth for the full design —
> tech stack, repository layout, agent-kernel internals, tool runtime, data model, and deployment
> topology (including what is implemented today vs. designed-only).

