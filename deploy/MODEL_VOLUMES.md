# Model volumes

The embedding, cap-router, and tool-intent sidecars load checkpoints that are far
too large to ship inside an image. Each service owns a Docker-managed named
volume instead. The image carries code only; the volume carries weights; a
startup bootstrap joins them.

For the full deployment procedure — configuration injection, startup order,
model updates, and offline transfer — see
[docs/model-deployment.en.md](../docs/model-deployment.en.md) and
[docs/model-deployment.zh-CN.md](../docs/model-deployment.zh-CN.md).

## Layout

```
                 image (code only)
   server.py · bootstrap.py · model_store
                    │
                    │ exec after the volume is ready
                    ▼
   ┌──────────── named volume (Docker-managed) ────────────┐
   │ versions/<revision>/...   immutable model snapshot    │
   │ current -> versions/<rev> the path the service loads  │
   │ .delveta-model.json       receipt: repo/rev/file pins │
   │ .fetch.lock               cross-process init lock     │
   │ .incoming-<revision>/     bytes in flight (resume)    │
   └───────────────────────────────────────────────────────┘
```

The tool-intent sidecar has one extra layer, because Ollama needs the file
registered before it can serve it:

```
   tool-intent-model (one-shot)          tool-intent (long-running)
   ───────────────────────────           ─────────────────────────
   .model-store/  ← download + verify    ollama serve
   .model-ready.json ← marker            ├─ ollama create (only if needed)
                                         └─ deletes .model-store once the
                                            blob digest is verified
```

## Startup contract

1. Acquire `<volume>/.fetch.lock` (two containers sharing a volume initialise it
   once).
2. Compare the receipt (repo, revision, per-file size) with the pin. On a match,
   return `current` immediately — no network, no hashing.
3. Otherwise download every required file into
   `.incoming-<revision>/`, resuming each file at the length already on disk.
4. Verify size and SHA256 for every file. A complete-length file with the wrong
   digest earns exactly one clean re-download, then fails.
5. Publish atomically: `versions/<revision>` ← staging, then swap `current`
   through a temporary symlink plus `os.replace`, then write the receipt and
   delete other versions.
6. `exec` the service process.

Any failure exits non-zero before the listening port is bound. There is no
fallback model and no "degraded" mode.

## What survives what

| Action | Image | Volume | Next start |
|---|---|---|---|
| `docker compose restart` | kept | kept | reuse, no network; does **not** re-read the model configuration |
| Container recreated (`up -d --force-recreate` after a config change) | same image | kept | reads the new configuration, then reuses or downloads |
| `docker compose build` / image rebuild | replaced | kept | reuse, no network |
| `docker compose down` | kept | kept | reuse, no network |
| `docker compose down -v` | kept | **deleted** | re-download |
| `docker volume prune`, Docker reset, uninstall | kept | **deleted** | re-download |
| New machine | pulled or rebuilt | **absent** | re-download, or import a bundle |
| Pinned revision changed | kept | kept | download the new revision only |

Code changes rebuild an image; model changes never do. A model change edits
`deploy/model_config/<service>.env` and recreates the container.

## Pinned parameters

| | embedding | cap-router | tool-intent |
|---|---|---|---|
| Configuration | `deploy/model_config/embedding.env` | `deploy/model_config/layachoice.env` | Compose environment variables |
| Volume | `delveta_embedding-model` | `delveta_laya-model` | `delveta_toolintent-cache` |
| Mount point | `/model` | `/models/layachoice` | `/root/.ollama` |
| Load path | `current/INT8` | `current` | Ollama blob store |
| Downloader | `bootstrap.py` | `bootstrap.py` | `tool-intent-model` service |
| Repository | `Delveta-BGE-M3-v1` | `Delveta-LayaChoice-v2` | `Delveta-Qwen3-0.6B-PE-v1` |
| Revision | `625d25d37490…` | `0e9c8facc4aa…` | `1a75945e12bf…` |
| Size | 0.57 GB | 1.29 GB | 0.44 GB |

Every file the service loads is listed with its size and SHA256: embedding and
cap-router read the list from `MODEL_SPEC_JSON` in their `env_file`, and
tool-intent from `fetch_model.py`. Anything else in the repository (training
code, READMEs, other precision variants) is never downloaded.

Base images are pinned by digest, not tag:

- `python:3.11-slim@sha256:6f31d6e9…` (embedding, cap-router, tool-intent init)
- `ollama/ollama@sha256:8262851b…` (tool-intent)
- `alpine@sha256:28bd5fe8…` (volume bundle runner)

## tool-intent: download and registration are separate

Ollama has no HTTP client and the stock image ships no `curl`. The split keeps
the serving image stock and the download logic in one place:

1. `tool-intent-model` (Python, shared `model_store`) downloads the GGUF into
   `<volume>/.model-store`, verifies size and SHA256, and exits 0.
2. Compose starts `tool-intent` only after that exit
   (`condition: service_completed_successfully`).
3. The entrypoint reads the registered model's blob digest from
   `ollama show --modelfile`. Ollama names the blob after the GGUF's SHA256, so
   this is an exact provenance check. A match means reuse; a mismatch or a
   missing model triggers `ollama create` from the file in the volume.
4. After the registration is verified, the entrypoint writes
   `.model-ready.json` and only then deletes `.model-store`. Ollama's blob store
   is the only persistent copy.
5. Later boots: the init service sees the marker (or reads Ollama's manifest
   directly when the marker is absent) and exits without downloading.

If the model is missing and `.model-store` is gone, the entrypoint removes the
stale marker and exits non-zero; the next `docker compose up` re-downloads.

## Moving a volume between machines

`deploy/model_volume_bundle.sh` wraps the volume in a single archive. Export
keeps `versions/`, the `current` symlink, and the receipt, so an imported volume
is reused as-is.

```bash
# on the source machine (use actual volume names, with the delveta_ prefix)
deploy/model_volume_bundle.sh export delveta_embedding-model embedding-model.tar.gz

# on the target machine (the volume must already exist)
docker compose create embedding
deploy/model_volume_bundle.sh import delveta_embedding-model embedding-model.tar.gz
docker compose up -d
```

Notes:

- Create the volume with `docker compose create <service>`. A manual
  `docker volume create` produces a volume that compose does not own, and
  `docker compose down -v` will not remove it.
- Partial downloads and the lock file are excluded from the archive.
- The archive is about as large as the model (safetensors compress poorly), so
  transfer time matters more than size.
- The revision inside the archive must match the pin, or the service will
  download the pinned revision anyway.

## Troubleshooting

| Symptom | Cause | Action |
|---|---|---|
| `model volume … is not writable` | volume permissions | recreate the container; the bootstrap runs as root |
| Repeated `resuming at N/M bytes` | network drops before a file completes | let it retry; progress is preserved |
| `sha256 mismatch` then a full retry | corrupt file at the pinned length | automatic; a second failure is a real pin error |
| `model … is not registered and … is absent` | model deleted without the marker | the marker is removed automatically; restart to re-download |
| Service healthy but new model still served | only `restart` was run, or the pin did not change | `up -d --force-recreate <service>`; confirm the revision in `deploy/model_config/<service>.env` |
| `MODEL_SPEC_JSON is not set` | missing `env_file`, or the variable is overridden | check the Compose `env_file` and variable precedence |
| Download works on one machine only | `HF_ENDPOINT` / proxy differences | set `HF_ENDPOINT` (and `HF_TOKEN` if private) in the compose environment |

## Invariants

- No host path is ever mounted for model data. Named volumes only.
- Model files are pinned by immutable revision plus explicit size and SHA256.
- Download, verification, locking, and resume exist once, in `model_store`.
- A service never binds its port before its checkpoint is verified.
- `HF_HUB_DISABLE_XET=1` is set for pullers: Range-based HTTP partials live
  inside the volume, so a recreated container resumes. Xet's chunk cache lives
  in the container filesystem and would be lost with it.
