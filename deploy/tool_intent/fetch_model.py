"""Download the pinned argument-extractor GGUF into the Ollama volume.

This module is the entrypoint of the one-shot ``tool-intent-model`` service. It
runs before ``tool-intent`` starts (compose waits for it to exit 0) and owns the
only network-dependent phase of that sidecar: all download, verification,
locking, and resume logic lives in :mod:`model_store`, so there is exactly one
implementation of it in the repository.

Ollama itself needs no download at all: the GGUF is stored in this volume, and
the tool-intent entrypoint registers it locally with ``ollama create``. Once
that registration is verified, the entrypoint writes ``.model-ready.json`` and
deletes the staging tree, leaving Ollama's blob store as the only copy. On later
boots the marker means this service has nothing to fetch and exits immediately.

Any failure exits non-zero, so compose never starts a serving container on top
of a missing or unverified checkpoint.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/app")

from model_store import FileSpec, ModelFetchError, ensure_model  # noqa: E402

DEFAULT_REPO = "eric-ml-nlp/Delveta-Qwen3-0.6B-PE-v1"
DEFAULT_REVISION = "1a75945e12bfeb203910bdb765e7cdf07f6a1126"
DEFAULT_FILE = "Q5_K_M/model.gguf"
DEFAULT_SHA256 = "17772c20f3bf356f251c29816b24000dfdbf4b412755024dc087ae63591b98f6"
DEFAULT_SIZE = 444_414_752

DEFAULT_MODEL = "delveta-pe:q5km"
DEFAULT_MODELS_DIR = "/root/.ollama/models"
DEFAULT_STORE = "/root/.ollama/.model-store"
DEFAULT_MARKER = "/root/.ollama/.model-ready.json"

# The media type Ollama uses for the weights layer of a `FROM <file>.gguf`
# model; its digest is the SHA256 of that file.
_MODEL_LAYER = "application/vnd.ollama.image.model"


def _split_tag(model: str) -> tuple[str, str]:
    """Split ``name:tag`` the way Ollama does (a colon in the last segment only)."""
    if ":" in model.rsplit("/", 1)[-1]:
        name, _, tag = model.rpartition(":")
        return name, tag
    return model, "latest"


def _registered_blob(models_dir: Path, model: str, sha256: str) -> bool:
    """Whether ``model`` is registered from exactly the pinned bytes.

    Reads Ollama's on-disk manifest so this service can skip the download even
    when the marker is absent (first boot after upgrading, marker deleted, or a
    volume populated by an older deployment). Without this check the marker's
    absence would force a redundant 444 MB re-download.
    """
    name, tag = _split_tag(model)
    if "/" not in name:
        name = f"library/{name}"
    manifest = models_dir / "manifests" / "registry.ollama.ai" / name / tag
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    layers = payload.get("layers")
    if not isinstance(layers, list):
        return False
    return any(
        isinstance(layer, dict)
        and layer.get("mediaType") == _MODEL_LAYER
        and layer.get("digest") == f"sha256:{sha256}"
        for layer in layers
    )


def _marker_matches(marker: Path, repo_id: str, revision: str, sha256: str,
                    model: str) -> bool:
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        isinstance(payload, dict)
        and payload.get("repo_id") == repo_id
        and payload.get("revision") == revision
        and payload.get("sha256") == sha256
        and payload.get("model") == model
    )


def _write_marker(marker: Path, repo_id: str, revision: str, sha256: str,
                  model: str) -> None:
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "repo_id": repo_id,
        "revision": revision,
        "sha256": sha256,
        "model": model,
    }
    tmp = marker.with_name(marker.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, marker)


def main() -> None:
    repo_id = os.environ.get("TOOL_INTENT_MODEL_REPO") or DEFAULT_REPO
    revision = os.environ.get("TOOL_INTENT_MODEL_REVISION") or DEFAULT_REVISION
    model = os.environ.get("OLLAMA_MODEL") or DEFAULT_MODEL
    model_file = os.environ.get("TOOL_INTENT_MODEL_FILE") or DEFAULT_FILE
    sha256 = os.environ.get("TOOL_INTENT_MODEL_SHA256") or DEFAULT_SHA256
    size = int(os.environ.get("TOOL_INTENT_MODEL_SIZE") or DEFAULT_SIZE)
    models_dir = Path(os.environ.get("OLLAMA_MODELS") or DEFAULT_MODELS_DIR)

    store = Path(os.environ.get("TOOL_INTENT_MODEL_STORE") or DEFAULT_STORE)
    marker = Path(os.environ.get("TOOL_INTENT_MODEL_MARKER") or DEFAULT_MARKER)

    if _marker_matches(marker, repo_id, revision, sha256, model):
        print(f"[tool-intent-model] {repo_id}@{revision[:12]} is already registered; "
              "nothing to download", flush=True)
        return

    if _registered_blob(models_dir, model, sha256):
        print(f"[tool-intent-model] {model} is already registered from the pinned "
              "bytes; writing the marker and skipping the download", flush=True)
        _write_marker(marker, repo_id, revision, sha256, model)
        return

    try:
        current = ensure_model(
            store,
            repo_id=repo_id,
            revision=revision,
            files=(FileSpec(model_file, size, sha256),),
            token=os.environ.get("HF_TOKEN") or None,
        )
    except ModelFetchError as exc:
        print(f"[tool-intent-model] initialisation failed: {exc}",
              file=sys.stderr, flush=True)
        raise SystemExit(1) from exc

    print(f"[tool-intent-model] ready: {current / model_file}", flush=True)


if __name__ == "__main__":
    main()
