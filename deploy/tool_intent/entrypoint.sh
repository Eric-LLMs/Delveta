#!/bin/sh
# Register the pre-downloaded argument-extractor GGUF with Ollama, then serve.
#
# This script never downloads anything. The one-shot `tool-intent-model` service
# (deploy/tool_intent/fetch_model.py, backed by the shared model_store) has
# already put the pinned, SHA256-verified GGUF at
# $TOOL_INTENT_MODEL_STORE/current/$TOOL_INTENT_MODEL_FILE and exited 0 before
# compose starts this container.
#
# What this script does:
#   1. starts a private loopback Ollama server for the registration phase,
#   2. reuses the registered model when its blob digest equals the pinned
#      GGUF SHA256 (Ollama names the blob after the file's SHA256, so this is an
#      exact provenance check, not just "a model with that tag exists"),
#   3. otherwise runs `ollama create` from the volume file and re-checks it,
#   4. writes .model-ready.json and only then deletes the staging tree, so the
#      only persistent copy is Ollama's blob store,
#   5. execs the real `ollama serve` in the foreground.
#
# Any failure exits non-zero before the public port is served, and removes the
# marker so the next `docker compose up` re-downloads instead of reusing a
# model whose bytes are gone.
set -eu

MODEL="${OLLAMA_MODEL:-delveta-pe:q5km}"
REPO="${TOOL_INTENT_MODEL_REPO:-eric-ml-nlp/Delveta-Qwen3-0.6B-PE-v1}"
REVISION="${TOOL_INTENT_MODEL_REVISION:-1a75945e12bfeb203910bdb765e7cdf07f6a1126}"
MODEL_FILE="${TOOL_INTENT_MODEL_FILE:-Q5_K_M/model.gguf}"
MODEL_SHA256="${TOOL_INTENT_MODEL_SHA256:-17772c20f3bf356f251c29816b24000dfdbf4b412755024dc087ae63591b98f6}"
STORE="${TOOL_INTENT_MODEL_STORE:-/root/.ollama/.model-store}"
MARKER="${TOOL_INTENT_MODEL_MARKER:-/root/.ollama/.model-ready.json}"
GGUF="$STORE/current/$MODEL_FILE"
SERVER_LOG="/tmp/ollama-init.log"
MODELFILE="/tmp/Modelfile"

log() { echo "[tool-intent] $*" >&2; }
die() { log "ERROR: $*"; exit 1; }

write_marker() {
  printf '{\n  "repo_id": "%s",\n  "revision": "%s",\n  "sha256": "%s",\n  "model": "%s"\n}\n' \
    "$REPO" "$REVISION" "$MODEL_SHA256" "$MODEL" > "$MARKER.tmp"
  mv "$MARKER.tmp" "$MARKER"
}

# The blob Ollama stores for `FROM <file>.gguf` is named after the file's
# SHA256, so the digest in the generated Modelfile identifies the exact bytes
# the model was created from.
registered_blob() {
  _from=$(ollama show --modelfile "$MODEL" 2>/dev/null | awk '/^FROM /{print $2; exit}')
  case "$_from" in
    *sha256-*) printf '%s' "${_from##*sha256-}" ;;
    *) printf '' ;;
  esac
}

# Private server for the registration phase; the public bind happens after exec.
ollama serve >"$SERVER_LOG" 2>&1 &
server_pid=$!
cleanup() {
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

attempt=0
until ollama list >/dev/null 2>&1; do
  attempt=$((attempt + 1))
  if [ "$attempt" -gt 120 ]; then
    cat "$SERVER_LOG" >&2 || true
    die "ollama server did not become ready"
  fi
  sleep 0.5
done

if ollama show "$MODEL" >/dev/null 2>&1; then
  if [ "$(registered_blob)" = "$MODEL_SHA256" ]; then
    log "reusing $MODEL (blob $MODEL_SHA256) from the volume"
    write_marker
    cleanup
    trap - EXIT INT TERM
    log "starting server"
    exec /bin/ollama serve
  fi
  log "$MODEL is registered from different bytes; re-registering from the pinned GGUF"
fi

if [ ! -f "$GGUF" ]; then
  rm -f "$MARKER"
  die "model $MODEL is not registered and $GGUF is absent; " \
      "the tool-intent-model init service must provide it"
fi

log "registering $MODEL from $GGUF"
cat >"$MODELFILE" <<EOF
FROM $GGUF
PARAMETER temperature 0
EOF

ollama create "$MODEL" -f "$MODELFILE" || die "ollama create failed for $MODEL"
[ "$(registered_blob)" = "$MODEL_SHA256" ] \
  || die "registered blob does not match the pinned GGUF after create"

# Registration is verified; the marker lets the init service skip future
# downloads, and the source GGUF can go — Ollama's blob store is the only copy.
write_marker
rm -rf "$STORE"

cleanup
trap - EXIT INT TERM
log "model $MODEL ready; starting server"
exec /bin/ollama serve
