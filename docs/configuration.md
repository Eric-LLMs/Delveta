# Configuration Reference

Environment variables and their defaults. Model inference never runs inside the API process, except the optional in-process reranker cross-encoder — LLM channels are managed in the admin console (Providers → Credentials); each is an OpenAI-compatible `base_url` + `api_key`, and a role bound to a channel routes chat straight to that provider (no code change). Web-search (provider + key + google engine id) and SMTP (host, credentials, TLS, enabled) are likewise configured in the admin console (Tools config) rather than `.env`. Model services are swapped through their deployment configuration, not by editing code. The embedding service loads BGE-M3 INT8 (ONNX) and the cap-router loads LayaChoice; both take a pinned manifest from `deploy/model_config/<service>.env` via Compose `env_file`, so a change is `edit the env file → docker compose up -d --force-recreate <service>` (a plain `restart` does not re-read `env_file`). TTS is changed with its Compose service definition and `TTS_*` variables. `--model-id` is a legacy TEI flag; neither current model service uses it. See [model-deployment.en.md](model-deployment.en.md) (§8) for the full procedure.

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://delveta:delveta@localhost:15432/delveta` | PostgreSQL + pgvector |
| `REDIS_URL` | `redis://localhost:16379/0` | cache / queue |
| `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` | `""` / `http://localhost:14000/v1` / `delveta-chat` | legacy default client (LiteLLM gateway); active channels + keys live in the DB and are managed in the admin console |
| `LLM_UPSTREAM_MODEL` / `LLM_UPSTREAM_BASE` / `LLM_UPSTREAM_KEY` | `openai/gpt-4o-mini` / `https://api.openai.com/v1` / `sk-xxx` | real upstream LLM (consumed by the gateway container) |
| `EMBEDDING_BASE_URL` / `EMBEDDING_MODEL` / `EMBEDDING_DIM` | `http://localhost:18080` / `Delveta-BGE-M3` / `1024` | BGE-M3 INT8 embedding service (ONNX, `deploy/embedding_int8`) |
| `TTS_BASE_URL` / `TTS_MODEL` / `TTS_VOICE` / `TTS_VOICE_ZH` | `http://localhost:18880/v1` / `kokoro` / `am_michael` / `zm_yunxi` | Kokoro-FastAPI TTS service (auto-switches to the Chinese voice for CJK text) |
| `RETRIEVAL_MODE` / `RETRIEVAL_GRPC_ADDR` | `in_process` / `localhost:15051` | capability seam: `in_process` or `grpc` |
| `WORKSPACE_DIR` | `.` | agent filesystem-tool root (`read_file` / `edit_file` / `bash`; path escape rejected) |
| `MEMORY_DIR` | `data/memory` | file memory directory (`MEMORY.md` index + one frontmatter `.md` per memory) |
| `SKILLS_DIR` | `skills` | `SKILL.md` skills directory at the repo root (version-controlled; lazy-loaded via the `skill` tool) |
| `PLUGINS_DIR` | `plugins` | plugin directory (`*/plugin.py`, version-controlled; auto-discovered at startup) |
| `MEMORY_NOTE_MAX_CHARS` | `4000` | `memory_save` content length cap (guardrail) |
| `MEMORY_RECALL_TOP_K` | `5` | proactive recall hits injected into the prompt memory section |
| `HISTORY_MAX_MESSAGES` / `HISTORY_KEEP_MESSAGES` | `40` / `20` | chat history length that triggers compaction / most-recent messages kept after compaction |
| `SESSION_EVENTS_RETENTION_DAYS` / `RETENTION_CRON` | `30` / `17 4 * * *` | daily worker cron purges `session_events` (audit log) older than this many days / its 5-field cron schedule |
| `WORKER_CONCURRENCY` / `WORKER_JOB_TIMEOUT` | `10` / `3600` | arq worker max concurrent jobs / per-job timeout (seconds); large-file RAG ingest needs a generous budget (PDF parse + embed) |
| `WEB_SEARCH_PROVIDER` / `WEB_SEARCH_API_KEY` / `WEB_SEARCH_ENGINE_ID` | `aggregate` / `""` / `""` | web-search provider (`aggregate`/`keyless` — no key, concurrent Bing + Google + Baidu + DDG fallback \| `duckduckgo` \| `tavily` \| `bing` \| `google`) + API key + google engine id; normally managed in admin → **Tools config**, these flat keys mirror that namespace |
| `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` / `REDDIT_USERNAME` / `REDDIT_PASSWORD` | `""` ×4 | `search_social` plugin reddit OAuth (free *script* app creds from `reddit.com/prefs/apps`); all four set → official `oauth.reddit.com` API, otherwise anonymous `search.json` (often 403) |
| `X_BEARER_TOKEN` | `""` | `search_social` plugin `x` platform — X API v2 bearer token; without it the `x` adapter degrades to a keyless site-scoped web search (`site:x.com` + `site:twitter.com`) |
| `OBJECT_STORE_ROOT` | `data/objects` | cloud-drive blob store root (SHA-256 content-addressed, ref-counted) |
| `DRIVE_CHUNK_SIZE` | `8MB` | cloud-drive upload chunk size |
| `DRIVE_MAX_CHUNKS` | `1024` | max chunks per upload session (cap on single-file size) |
| `DRIVE_MAX_FILE_SIZE` | `0` (unlimited) | per-file size limit for the cloud drive (bytes; `0` = no limit) |
| `INGEST_CHUNK_CHARS` / `INGEST_CHUNK_OVERLAP` | `1200` / `150` | text chunking window / overlap used when a cloud-drive file is ingested for RAG |
| `TOOLKIT_MAX_INPUT_TOKENS` / `TOOLKIT_MAX_FILE_BYTES` | `100000` / `20MB` | toolkit generation (mind map / slides / summary): one-shot **capacity check** of the complete raw input — at or below it the full text goes to the model in ONE call; above it `summary`/`mindmap` enter the explicit big-document multi-call flow (raw-grounded call per line-tracked batch + deterministic merge; never a silent digest), while `slides` keeps the complete text in its single call (`slides` does not consume the batch plan; an over-context input fails loudly at the call) / per-file size cap — files over the cap are refused and greyed out in the picker |
| `SLIDES_GENERATION_MODE` | `direct` | slides generation engine: `direct` = one semantic LLM call + local compiler (the five generate-stage nodes; default), `legacy` = the four-pass brief chain, kept as config-switchable fallback. A request may override per job via `generation_mode` |
| `TOOLKIT_LLM_TIMEOUT_S` | `300` | per-call deadline for toolkit generation calls; since 2026-09-15 these calls stream and accumulate, so it bounds IDLE time between chunks, not total generation time (the provider's ~300s non-stream gateway cutoff no longer kills a legitimate full-context Pass A) |
| `LLM_DISABLE_THINKING` | `true` | batch generation (`complete`/`complete_json`) sends the Qwen-compatible `enable_thinking: false` flag — reasoning tokens are pure latency for schema-validated JSON output; interactive chat (`chat_stream`) always keeps thinking |
| `LLM_POOL_MAX_CONNECTIONS` / `LLM_POOL_MAX_KEEPALIVE_CONNECTIONS` / `LLM_POOL_KEEPALIVE_EXPIRY_SECONDS` | `1000` / `100` / `300` | per-origin online-LLM HTTP connection pool ([architecture.md §12.8](architecture.md#128-online-llm-transport-pool--one-origin-one-transport-credentials-per-wrapper)): maximum simultaneous connections (in use + idle) / idle connections retained for reuse / idle keep-alive retention in seconds. These are transport limits only — per-request timeout and retry budgets are separate |
| `EMBED_BATCH_SIZE` | `16` | embedding batch size for indexing cloud-drive files into pgvector |

## Model deployment variables

These are consumed by the model containers, not by the API process. Their
defaults live in `docker-compose.yml`, and the two pinned manifests live in
`deploy/model_config/`.

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_SPEC_JSON` | set by `deploy/model_config/embedding.env` / `layachoice.env` | Complete model pin (repo, full commit revision, `load_path`, per-file size + SHA256) for `embedding` and `cap-router`. Injected by Compose `env_file` at container creation; changing it requires `up -d --force-recreate`, not `restart`. Never baked into an image |
| `TOOL_INTENT_MODEL_REPO` / `_REVISION` / `_FILE` / `_SHA256` / `_SIZE` | `eric-ml-nlp/Delveta-Qwen3-0.6B-PE-v1` / pinned SHA / `Q5_K_M/model.gguf` / pinned SHA256 / `444414752` | GGUF pin downloaded and verified by the one-shot `tool-intent-model` service before `tool-intent` starts |
| `TOOL_INTENT_OLLAMA_MODEL` | `delveta-pe:q5km` | Name the `tool-intent` entrypoint registers with `ollama create` and uses for its health check |
| `HF_ENDPOINT` | `https://huggingface.co` | Hugging Face endpoint (mirror or proxy) used by all model downloaders |
| `HF_TOKEN` | `""` | Hugging Face token for private repositories |
| `HF_HUB_DISABLE_XET` | `1` | Keeps downloads on Range-based HTTP so partial files persist in the model volume across container recreation |

## Model services in Compose

| Service | Role | Started by |
|---|---|---|
| `embedding` | BGE-M3 INT8 ONNX dense embeddings on port 18080 | Launchers and manual `docker compose up` |
| `cap-router` | LayaChoice capability selection on port 18092 | Launchers and manual `docker compose up` |
| `tool-intent-model` | One-shot GGUF download + verification | Only when `tool-intent` is requested |
| `tool-intent` | Ollama-served parameter extractor on port 18091 | Optional; `docker compose up -d tool-intent` |
| `tts` | Kokoro-FastAPI speech synthesis on port 18880 | Launchers and manual `docker compose up` |
| `stt` | FunASR SenseVoiceSmall transcription on loopback port 18881 | Desktop launcher and manual `docker compose up` |
| `reranker` | **Legacy** TEI reranker (port 18081) defined in Compose but not wired; the in-process cross-encoder is the actual rerank implementation | Not started by the launchers; leave stopped (a bare `docker compose up -d` would start it, so name services explicitly) |

See [docs/architecture.md](architecture.md) for the full topology and
[model-deployment.en.md](model-deployment.en.md) for model volumes, updates,
and offline export/import.
