# Model Deployment Pattern : Code in the Image, Models in Volumes

This document describes how Delveta deploys its three model services: embedding,
cap-router (LayaChoice), and tool-intent (Ollama). The core idea is that
**application images carry only code, the runtime, and dependencies, while model
weights live in Docker named volumes** populated by an initialisation step at
container start.

**Status labels**

| Label | Meaning |
|---|---|
| **[Implemented]** | Code and configuration are in place. |
| **[Pending Implementation]** | Agreed as necessary, but the code is not written. |
| **[Pending Verification]** | The code is complete, but it has not been verified beyond a single local host, or the verification scenarios are incomplete. |

An item counts as done only when it is both implemented and verified. Section 12
lists the current state item by item.

---

## 1. Overview

The relationship between the three service images and their volumes is shown
below. The volume names are the actual Docker names; the prefix comes from the
top-level `name: delveta` in `docker-compose.yml`.

```
                       docker compose (project: delveta)
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
   embedding                   cap-router               tool-intent-model
   delveta-embedding-int8      delveta-cap-router       delveta-tool-intent-init
        │                           │                   (one-shot initialiser)
        │ /model                    │ /models/layachoice         │ /root/.ollama
        ▼                           ▼                            ▼
 delveta_embedding-model     delveta_laya-model       delveta_toolintent-cache
                                                                  ▲
                                                                  │ same named volume
                                                            tool-intent
                                                          delveta-tool-intent
```

| Service | Responsibility | Model size |
|---|---|---|
| `embedding` | INT8 ONNX text embedding, `POST /embed` compatible | about 586 MB |
| `cap-router` | LayaChoice capability-selection model, `POST /v1/systemone` | about 1.32 GB |
| `tool-intent-model` + `tool-intent` | Argument extractor (GGUF, served by Ollama) | about 444 MB |

Each service loads only the files it declares. Training code, READMEs, and other
quantisation variants in the repository are never downloaded.

## 2. Architecture Principles

1. **Images do not contain weights.** An application image contains only code,
   the runtime, and dependencies. Models and images are decoupled, so image size
   does not grow with the model. **[Implemented]**
2. **Models live in Docker named volumes, not host absolute paths.** Deployment
   does not depend on drive letters, project directories, or pre-existing model
   files; moving to another machine needs only Docker and network access for the
   first download. **[Implemented]**
3. **The first start downloads, verifies, and persists automatically; later
   starts reuse.** Reuse is decided from the publish receipt and file lengths,
   without re-hashing every byte. **[Implemented]**
4. **There is only one download implementation.** Resume, integrity checks,
   concurrency locking, and atomic publish all live in `deploy/model_store`;
   each service's repository, revision, and file list come from its runtime
   configuration. **[Implemented]**
5. **Model versions and code versions are independent.** All three services take
   their model parameters at runtime: tool-intent uses `TOOL_INTENT_MODEL_*`
   environment variables; embedding and cap-router use `MODEL_SPEC_JSON`,
   injected from `deploy/model_config/*.env` through Compose `env_file`.
   Changing a model requires editing configuration and recreating the container,
   never rebuilding the image. **[Implemented]**
6. **Optional offline export/import.** Used to move models into environments
   without access to Hugging Face. **[Implemented]**; cross-machine verification
   is **[Pending Verification]**.
7. **An Ollama model is marked ready only after registration is verified.** The
   service does not report healthy and does not bind its public port while the
   model is unregistered. **[Implemented]**

## 3. Volume Layout

### 3.1 embedding and cap-router

Both use the same layout; only the mount point and load path differ.

```
<volume>/
├── versions/
│   └── <revision>/              # immutable snapshot named by commit SHA
│       └── ...                  # repository-relative paths are preserved
├── current -> versions/<revision>
├── .delveta-model.json          # publish receipt: repo, revision, per-file size/sha256
├── .fetch.lock                  # cross-process initialisation lock
└── .incoming-<revision>/        # download staging; absent after publish
```

| Entry | Purpose | Lifecycle |
|---|---|---|
| `versions/<revision>/` | Verified model snapshot | Current version is kept; other versions are deleted on publish |
| `current` | The path the service reads (symlink) | Swapped atomically with a temporary link plus `os.replace` |
| `.delveta-model.json` | Basis for reuse; matching the pin skips the download | Rewritten after every publish |
| `.fetch.lock` | Ensures a volume is initialised by one process at a time | Persistent; excluded from export |
| `.incoming-<revision>/` | Unfinished downloads, keyed by revision to support resume | Renamed on publish; stagings for other revisions are pruned |

### 3.2 tool-intent

The Ollama volume root is `/root/.ollama`, which holds the staging area, the
marker file, and Ollama's own model store.

```
delveta_toolintent-cache/
├── .model-store/                # initialiser staging (deleted after registration)
│   ├── versions/<revision>/
│   └── current -> versions/<revision>
├── .model-ready.json            # marker: these bytes are registered with Ollama
├── models/
│   ├── blobs/sha256-<digest>    # the only persistent copy; digest equals the GGUF SHA256
│   └── manifests/registry.ollama.ai/library/delveta-pe/q5km
└── .fetch.lock
```

`.model-store/` is transitional: it is deleted after successful registration and
verification, leaving Ollama's blob store as the only persistent copy. It may
legitimately be absent when the volume is exported.

## 4. Startup Sequence

### 4.1 embedding and cap-router

```
container start
   │
   ├─ read and validate MODEL_SPEC_JSON (injected by Compose env_file)
   │     └─ missing, invalid JSON, non-commit revision, bad path or digest → exit 1
   ├─ acquire <volume>/.fetch.lock (blocking)
   ├─ receipt matches the pin and current holds files of the right size? ── yes ──► exec the service
   │                                                              no
   ├─ download missing files into .incoming-<revision>/
   │     └─ each file resumes from its current length via an HTTP Range request
   ├─ verify size and sha256 for every file
   │     └─ correct length but wrong digest: discard and retry once; then fail
   ├─ publish: versions/<revision> ← staging directory
   │           atomically swap current
   │           write the receipt, delete other versions
   └─ exec the service process (uvicorn / python server.py)
```

Any failed step exits non-zero; the service process never starts and the port is
never bound. There is no fallback model.

The configuration is read again on every container start, so changing it only
requires recreating the container; the image itself is unchanged.

### 4.2 tool-intent

`tool-intent` is optional. This sequence runs only when `tool-intent` and its
initialiser are started explicitly; the launchers start `embedding` and
`cap-router` by default and leave the extractor stopped.

Download and registration are split across two containers, joined by Compose
with `depends_on: condition: service_completed_successfully`.

```
docker compose up -d tool-intent
   │
   ├─ tool-intent-model (one-shot)
   │    1. .model-ready.json matches the pin? ── yes ──► exit 0
   │    2. model layer digest in the Ollama manifest matches the pin? ── yes ──► write marker, exit 0
   │    3. download and verify the GGUF into .model-store/, exit 0
   │
   └─ only after the initialiser exits 0:
        tool-intent
          1. start an Ollama process used only for registration
          2. registered model blob digest equals the pin?
               yes → write marker, delete .model-store, exec ollama serve
               no  → ollama create from .model-store/current/<file>
                     re-check the blob digest
                     write marker, delete .model-store, exec ollama serve
          3. model not registered and .model-store absent
               → remove the marker, exit 1 (the next start re-downloads)
```

Ollama names the weight blob after the file's SHA256, so `FROM ...sha256-<digest>`
in `ollama show --modelfile` is direct provenance evidence, not merely evidence
that a tag with that name exists.

The health check passes only when the model is registered:
`ollama list | grep -q "$OLLAMA_MODEL"`.

### 4.3 Granularity of the reuse check

A normal start compares only the receipt and file lengths; it does not recompute
digests. Integrity is established once at download time, and
`versions/<revision>` is never modified after publish. This avoids hashing 1.3 GB
on every start.

## 5. Design Rationale

**5.1 Why weights are not baked into images**

Images are pushed to registries, pulled by every machine, and produce new layers
on every build. Embedding weights would push image size into the GB range, making
build and distribution far more expensive than downloading a model once; model
updates would also force new versions of the code image.

**5.2 Why named volumes instead of host directories**

Host directories bind deployment to a specific drive letter and project path, so
every machine needs a new path convention; named volumes are managed by Docker,
behave consistently across platforms, and are created and removed with the
Compose project. The cost is that volumes cannot be copied with a file manager,
which is why the export/import script exists.

**5.3 Why atomic publish**

The service can only ever see a complete model. Downloads are written to a
staging directory unrelated to the load path and exposed through an atomic swap
of `current` only after every file passes verification. A failure midway never
leaves a load path that exists but is incomplete.

**5.4 Why a deterministic staging directory and Range resume**

The staging directory is keyed by revision. After a dropped connection, the
bytes already written are the starting offset for the next attempt; if the
container is recreated, those bytes are still in the volume. The downloader uses
the standard-library `urllib` to issue Range requests, rather than a client
library that deletes its temporary file on failure.

**5.5 Why both size and sha256 are pinned**

A size check alone cannot detect content replacement at the same length; a
digest check alone cannot tell whether a download has finished. Pinning both and
verifying every file before publish rejects upstream artifacts that were silently
replaced.

**5.6 Why base images are pinned by digest**

Tags move; digests do not. `python:3.11-slim`, `ollama/ollama`, and the `alpine`
image used by the export script are pinned to specific digests, so builds on
different machines and at different times produce the same runtime.

**5.7 Why tool-intent is split into two containers**

The official Ollama image has no HTTP client, and installing curl/apt into it
just for downloads is undesirable. The split keeps download logic in the Python
`model_store`, leaves the Ollama image stock so it only registers and serves, and
lets a Compose dependency condition guarantee that the service never starts
before the bytes are ready.

**5.8 Why the Ollama manifest is read in addition to the marker**

A volume created before an upgrade may already have the model registered without
a marker file. Trusting only the marker would re-download 444 MB. Reading the
model-layer digest from the manifest confirms the provenance of the bytes without
trusting the marker, so the volume can be reused directly.

**5.9 Why configuration uses env_file instead of a mounted file**

Model parameters do not belong in the image, and they should not require an extra
host bind mount either. Compose `env_file` injects the configuration as
environment variables when the container is created: the image stays generic, the
configuration follows the deployment environment, no new mount point appears
inside the container, and changing a model never requires rebuilding the image.
The structured parameters (repository, revision, file list, sizes, SHA256s, load
path) live in a single `MODEL_SPEC_JSON`, which the startup code parses and
validates in one step, avoiding a dozen interdependent variables that can end up
half-configured.

The cost is that environment values are fixed at container creation: changing the
configuration requires recreating the container, and `restart` will not read the
new values. The update procedure states this explicitly.

**5.10 Costs and boundaries**

- The first start needs network access and extra time (download plus
  verification).
- A volume is valid only within its current Docker environment; migration
  requires export/import.
- Only the current version is retained; rolling back requires a re-download.
- Import uses a length-only fast path and does not re-hash (see section 12).

## 6. Offline Export and Import

The export script packs a named volume into a single `tar.gz`. The archive keeps
`versions/`, the `current` symlink, and the receipt, so an imported volume can be
reused directly.

**Important: the script takes the actual Docker volume name, not the Compose
volume key.** This project's Compose project name is `delveta`, so the actual
volumes are `delveta_embedding-model`, `delveta_laya-model`, and
`delveta_toolintent-cache`.

### 6.1 Export (source machine)

```bash
deploy/model_volume_bundle.sh export delveta_embedding-model embedding-model.tar.gz
deploy/model_volume_bundle.sh export delveta_laya-model laya-model.tar.gz
deploy/model_volume_bundle.sh export delveta_toolintent-cache toolintent-cache.tar.gz
```

- The script refuses to overwrite an existing archive.
- `.fetch.lock`, `.incoming-*`, and `.current.tmp` are excluded.
- On Windows, Git Bash is required; the script converts host paths with
  `cygpath`.

### 6.2 Import (target machine)

The volume must already exist. Create it with Compose, not with
`docker volume create`; otherwise `docker compose down -v` will not manage it.

```bash
docker compose create embedding
deploy/model_volume_bundle.sh import delveta_embedding-model embedding-model.tar.gz
docker compose up -d embedding

docker compose create cap-router
deploy/model_volume_bundle.sh import delveta_laya-model laya-model.tar.gz
docker compose up -d cap-router

docker compose create tool-intent-model
deploy/model_volume_bundle.sh import delveta_toolintent-cache toolintent-cache.tar.gz
docker compose up -d tool-intent-model tool-intent
```

The first start after import should hit the receipt and reuse the model without
any network request.

### 6.3 Status and verification

- Export, import, and post-import reuse: **[Implemented]**.
- A local round trip between two independent volumes confirmed that the archive
  preserves the published layout and that the first start after import reuses
  the model without downloading.
- Import across real machines (different hosts, different architectures):
  **[Pending Verification]**.
- Re-verifying sha256 on import: **[Pending Implementation]**. The fast path
  currently compares file lengths only.

## 7. Day-to-Day Operations

| Operation | Command | Effect on model volumes |
|---|---|---|
| Default start (embedding + cap-router) | `docker compose up -d embedding cap-router` | Downloads if absent, reuses if present |
| On-demand start (adds the argument extractor) | `docker compose up -d tool-intent-model tool-intent` | Downloads the GGUF if absent, reuses if present |
| View initialiser logs | `docker compose logs --tail=100 tool-intent-model` | None |
| View status | `docker compose ps` | None |
| Restart a container | `docker compose restart embedding` | Preserved; does **not** re-read env_file |
| Change the model configuration | Edit `deploy/model_config/*.env`, then `docker compose up -d --force-recreate <service>` | Old version is kept until the new one is verified and switched |
| Rebuild an image after code changes | `docker compose build embedding`, then `up -d` | Preserved; no re-download |
| Stop and keep volumes | `docker compose down` | Preserved |
| Stop and delete volumes | `docker compose down -v` | **Deleted**; the next start re-downloads |
| Inspect volume contents | `docker compose exec embedding ls -la /model` | Read-only |
| Inspect volume usage | `docker system df -v` | Read-only |

Names to keep straight:

- Compose services: `embedding`, `cap-router`, `tool-intent-model`,
  `tool-intent`
- Container names: `delveta-embedding`, `delveta-cap-router`,
  `delveta-tool-intent` (`tool-intent-model` is a one-shot container that remains
  in the Exited state)
- Compose volume keys: `embedding-model`, `laya-model`, `toolintent-cache`
- Actual volume names: prefixed with `delveta_`

## 8. Updating a Model

### 8.1 tool-intent

All parameters come from environment variables; recreating the containers is
enough, with no image rebuild:

```
TOOL_INTENT_MODEL_REPO
TOOL_INTENT_MODEL_REVISION
TOOL_INTENT_MODEL_FILE
TOOL_INTENT_MODEL_SHA256
TOOL_INTENT_MODEL_SIZE
OLLAMA_MODEL
```

```bash
docker compose up -d --force-recreate tool-intent-model tool-intent
```

### 8.2 embedding and cap-router

**Configuration files (on the host):**

- `deploy/model_config/embedding.env`
- `deploy/model_config/layachoice.env`

Compose injects the file contents as `MODEL_SPEC_JSON` when the container is
created. The value is not written into the image and is not exposed through a
bind mount; `bootstrap.py` reads it from the environment.

**Update procedure:**

1. Edit the relevant `.env` file. Keep the double-quoted value and the `\"`
   escapes; change `revision`, `files`, `size`, `sha256`, and `load_path` when
   needed.
2. Recreate the container:

```bash
docker compose up -d --force-recreate embedding
# or
docker compose up -d --force-recreate cap-router
```

`up -d` also detects the environment change and recreates the container, but the
explicit `--force-recreate` is unambiguous. **`docker compose restart` does not
read the new env values**, because it reuses the environment of the existing
container.

No `docker compose build` and no image rebuild is involved.

**Behaviour at startup:**

1. Parse and validate `MODEL_SPEC_JSON`; a missing or malformed value exits 1.
2. If the receipt matches the configuration and the file lengths are correct,
   reuse the volume without network access.
3. Otherwise download into `.incoming-<revision>/` (resumable) and verify every
   file's size and sha256.
4. Only after every check passes, publish `versions/<revision>` and atomically
   switch `current`.
5. Only after `current` has switched and the receipt has been written are other
   version directories removed.

**Confirming that the new version is published before the old one is removed:**

```bash
docker compose logs embedding | grep -E 'spec:|verified|ready:|reusing'
docker compose exec embedding python -c "import os;print(os.readlink('/model/current'))"
docker compose exec embedding python -c "import os;print(sorted(os.listdir('/model/versions')))"
```

The log order must be: `spec:` → one `verified` per file → `ready: ... ->
versions/<new revision>`. If `ready:` never appears, the process failed during
verification and exited; `current` still points at the old version and the old
directory is still present. Other versions are removed only after `ready:`
appears.

### 8.3 Obtaining pins for a new version

The Hugging Face tree API returns both the file size and the LFS object ID, the
latter being the SHA256:

```bash
curl -s "https://huggingface.co/api/models/<repo>/tree/<revision>?recursive=true" \
  | jq -r '.[] | select(.type=="file") | [.path, .size, (.lfs.oid // "-")] | @tsv'
```

List only the files the service actually loads; do not mirror the whole
repository. `load_path` decides which subdirectory under `current` the service
loads from (`INT8` for embedding, `.` for cap-router).

### 8.4 Rollback

Set the revision back to the old value and recreate the container:
`docker compose up -d --force-recreate embedding` for embedding and cap-router;
`docker compose up -d --force-recreate tool-intent-model tool-intent` for
tool-intent. Because only one version is retained, the old version must be
downloaded again; use a previously exported archive when there is no network
access.

## 9. Caveats and Troubleshooting

- `docker compose down -v` deletes all three model volumes. It is the only
  routine operation that automatically triggers re-downloads.
- `docker volume prune`, a Docker reset, or an uninstall also deletes volumes.
- Export/import commands must use the actual volume names (with the `delveta_`
  prefix).
- Transfer archives as binary; the script does not extract to the host and does
  not verify the archive's own integrity.
- The embedding and cap-router configuration is injected through `env_file`;
  after editing it the container must be recreated, because `restart` has no
  effect.
- When `MODEL_SPEC_JSON` is missing or malformed, initialisation exits 1 during
  validation and `current` is not touched.
- With base images pinned by digest, upgrading a base image requires changing the
  digest explicitly.
- `HF_ENDPOINT` is for mirrors or proxies; `HF_TOKEN` is for private
  repositories. Both are passed through the Compose environment.
- Keep `HF_HUB_DISABLE_XET=1`: Xet's chunk cache lives in the container
  filesystem and is lost when the container is recreated.
- Only the current model version is retained; rollback requires a re-download or
  an archive.
- There is no disk-space preflight before downloading (see section 12).

| Symptom | Cause | Action |
|---|---|---|
| `model volume ... is not writable` | Volume permission problem | Recreate the container; initialisation runs as root |
| Repeated `resuming at N/M bytes` | The network drops before a file completes | Let it retry; progress is preserved |
| `sha256 mismatch` followed by one full retry | Pinned length but corrupt content | Handled automatically; a second failure indicates a bad pin or upstream change |
| `model ... is not registered and ... is absent` | The model was deleted and staging is gone | The marker is removed automatically; restart to download |
| Service is healthy but serves the old model | Only `restart` was run, or the configuration did not actually change | `up -d --force-recreate <service>`; confirm the `revision` was changed |
| `MODEL_SPEC_JSON is not set` | The service has no `env_file`, or the variable is overridden | Check Compose's `env_file` and variable precedence |
| Only one machine can download | Proxy or `HF_ENDPOINT` differs | Align the Compose environment variables |

## 10. Onboarding a New Model

### 10.1 Required parameters

| Parameter | Meaning | Example |
|---|---|---|
| `repo_id` | Hugging Face repository | `eric-ml-nlp/Delveta-BGE-M3-v1` |
| `revision` | Immutable commit SHA; branches and movable tags are forbidden | `625d25d3…` |
| `files` | Files required to start: path, byte size, SHA256 | `INT8/model.onnx`, `568511234`, `0e0ada7f…` |
| `load_path` | Subdirectory under `current` that the service loads | `INT8` (embedding), `.` (cap-router) |
| Volume | Compose volume key and mount point | `embedding-model` → `/model` |
| Load path | Directory the service reads, determined by `load_path` | `/model/current/INT8` |
| Health check | Time before the first health verdict; must cover the download | `start_period: 900s` |

### 10.2 Steps

1. Confirm which files the service actually loads, and list only those.
2. Obtain each file's `size` and `lfs.oid` (SHA256) from the HF tree API.
3. Add a `bootstrap.py` that calls `load_model_spec()` to obtain the
   configuration, then calls `ensure_model()`.
4. In the Dockerfile, copy `model_store` and `bootstrap.py`, set the entrypoint
   to the bootstrap, and keep the original service command as `CMD`. The
   configuration is not copied into the image.
5. Create `deploy/model_config/<service>.env` holding the parameters as
   `MODEL_SPEC_JSON`; declare an `env_file` and a named volume in Compose. Model
   weights live only in the named volume, and the configuration introduces no
   bind mount.
6. Set the health check's `start_period` to cover the worst case of download +
   verification + load.
7. Verify the first download on an empty volume and reuse on a populated volume.
8. For offline deployment, perform one round trip with the export/import script.

### 10.3 Configuration injection

The model parameters are neither inside the image nor provided through a bind
mount. Compose injects the `env_file` as `MODEL_SPEC_JSON` when it creates the
container, and `bootstrap.py` parses it at startup. The same image can therefore
load different model versions in different environments, and the configuration
does not become fixed by distributing the image.

The only requirement for changing the configuration is to recreate the
container; `restart` has no effect.

## 11. Configuration of the Three Model Services

### 11.1 Summary

| Item | embedding | cap-router | tool-intent |
|---|---|---|---|
| Compose service | `embedding` | `cap-router` | `tool-intent-model` + `tool-intent` |
| Image | `delveta-embedding-int8:latest` | `delveta-cap-router:latest` | `delveta-tool-intent-init:latest` + `delveta-tool-intent:latest` |
| Build context | `./deploy` | `./deploy` | `./deploy` |
| Dockerfile | `embedding_int8/Dockerfile` | `laya/Dockerfile` | `tool_intent/init.Dockerfile` + `tool_intent/Dockerfile` |
| Model configuration | `deploy/model_config/embedding.env` | `deploy/model_config/layachoice.env` | Environment variables (Compose) |
| Volume (Compose key) | `embedding-model` | `laya-model` | `toolintent-cache` |
| Volume (actual name) | `delveta_embedding-model` | `delveta_laya-model` | `delveta_toolintent-cache` |
| Mount point | `/model` | `/models/layachoice` | `/root/.ollama` |
| Load path | `/model/current/INT8` | `/models/layachoice/current` | `delveta-pe:q5km` (Ollama blob) |
| Repository | `eric-ml-nlp/Delveta-BGE-M3-v1` | `eric-ml-nlp/Delveta-LayaChoice-v2` | `eric-ml-nlp/Delveta-Qwen3-0.6B-PE-v1` |
| Revision | `625d25d3749076cee90abcd1904da3a6e7ce4c9b` | `0e9c8facc4aa72eb222caac490a020b01745ba89` | `1a75945e12bfeb203910bdb765e7cdf07f6a1126` |
| Total download | 585,611,532 bytes (about 586 MB) | 1,322,020,109 bytes (about 1.32 GB) | 444,414,752 bytes (about 444 MB) |
| Host port | `18080 → 80` | `127.0.0.1:18092 → 8000` | `127.0.0.1:18091 → 11434` |
| Health start period | `start_period: 900s` | `start_period: 1800s` | `start_period: 1800s` |

### 11.2 Environment variables

| Variable | embedding | cap-router | tool-intent |
|---|---|---|---|
| `MODEL_SPEC_JSON` (from `env_file`) | ✓ | ✓ | |
| `TOOL_INTENT_MODEL_REPO` / `_REVISION` / `_FILE` / `_SHA256` / `_SIZE` | | | ✓ |
| `OLLAMA_MODEL` | | | ✓ |
| `HF_ENDPOINT` | ✓ | ✓ | ✓ |
| `HF_TOKEN` | ✓ | ✓ | ✓ |
| `HF_HUB_DISABLE_XET` | ✓ | ✓ | ✓ |
| `MODEL_DIR` | Set by the bootstrap according to `load_path` | | |
| `LAYA_CHOICE_MODEL_DIR` | | Set by the bootstrap according to `load_path` | |

### 11.3 Base images

| Purpose | Pinned value |
|---|---|
| embedding, cap-router, tool-intent initialiser | `python:3.11-slim@sha256:6f31d6e9ba2b0a787a3f81c37b004155b87b9efa1b771182bd550c1615745be5` |
| tool-intent service | `ollama/ollama@sha256:8262851b2846b87c649eddf3e76beb270c52f4d1bc94559f47efde16b0841551` |
| Volume export/import script | `alpine@sha256:28bd5fe8b56d1bd048e5babf5b10710ebe0bae67db86916198a6eec434943f8b` |

### 11.4 The two tool-intent containers

| | `tool-intent-model` | `tool-intent` |
|---|---|---|
| Type | One-shot (`restart: "no"`) | Long-running (`restart: unless-stopped`) |
| Network | Required, to reach Hugging Face | Not required |
| Responsibility | Download, verify, and publish the GGUF | Register the model and serve inference |
| Success signal | Exit code 0 | Health check passes |
| Shared | `delveta_toolintent-cache` | `delveta_toolintent-cache` |

## 12. Implementation Status

### 12.1 Implemented

| Capability | Status | Local verification evidence |
|---|---|---|
| Image/volume separation and three named volumes | Implemented | `docker compose config` shows model services mounting named volumes only, with no host model paths |
| Runtime injection of model configuration | Implemented | With the same image, changing only the `revision` configuration completed download, publish, and cleanup of the old version; the image ID did not change |
| Automatic first-boot download, verification, atomic publish | Implemented | All three models completed download, per-file verification, and publish on first start |
| Resume via HTTP Range | Implemented | All three models logged `resuming at` and continued from the offset to the full length |
| Per-file size + SHA256 pins | Implemented | Pins come from `MODEL_SPEC_JSON`; per-file `verified` output before publish |
| Concurrency lock | Implemented | Volume-level mutual exclusion via `fcntl.flock`; concurrent two-container initialisation has not been measured (see 12.3) |
| Base images pinned by digest | Implemented | All four images built and ran from digests |
| Independent tool-intent initialisation | Implemented | The service container starts only after the initialiser exits 0; the production migration performed no re-download |
| Ollama registration provenance check | Implemented | Blob digest equals the GGUF SHA256; a mismatch triggers re-registration |
| Manifest fallback when the marker is absent | Implemented | The upgraded production volume reused the existing model without a 444 MB download |
| Failure self-healing | Implemented | After deleting the model, restart exits 1 and removes the marker |
| Volume export/import | Implemented | A local round trip preserved the published layout; the first start after import reused the model without downloading |
| Health checks cover initialisation | Implemented | All three services report healthy; embedding and cap-router health endpoints respond normally |

### 12.2 Pending Implementation

1. **Re-verify SHA256 on import.** Imported volumes take the length-only fast
   path, so an archive corrupted in transit at unchanged length cannot be
   detected. An explicit `--verify` mode is needed.
2. **Automated tests for `model_store`.** There are no unit or integration tests
   covering resume, locking, atomic publish, or corruption retry.
3. **Disk-space preflight before download.** Insufficient space fails during the
   write, without a clear check before the start.

### 12.3 Pending Verification

1. **Export/import across real machines.** Only two volumes on the same machine
   have been verified.
2. **Compose gating when initialisation fails.** The initialiser has been
   confirmed to exit 1; it has not been verified that `tool-intent` remains
   unstarted in that case.
3. **Private repositories, proxies, and mirrors.** Combinations of `HF_TOKEN` and
   `HF_ENDPOINT` are unverified.
4. **Non-amd64 architectures.** Digests have been resolved only on the local
   amd64 host.
5. **Xet on/off comparison.** It is currently forced off; behaviour with it
   enabled has not been compared.
6. **Concurrent initialisation of one volume.** The lock is implemented, but two
   containers initialising the same empty volume simultaneously has not been
   measured.

These three status classes correspond to the architecture principles in section
2; no capability should be described externally as complete until it has passed
verification.
