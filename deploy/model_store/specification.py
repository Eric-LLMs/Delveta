"""Parse the runtime model specification injected through the environment.

The specification is delivered as ``MODEL_SPEC_JSON`` by the service's Compose
``env_file``. It is therefore read at container start from outside the image:
changing a model version means editing the env file and recreating the
container, with no image rebuild and no host bind mount.

The document is validated strictly and completely before any download starts, so
a typo fails the boot instead of producing a half-configured volume. Only the
schema this project actually uses is accepted; unknown keys are ignored so the
document can grow without breaking older images.

Example::

    {
      "repo_id": "eric-ml-nlp/Delveta-BGE-M3-v1",
      "revision": "625d25d3749076cee90abcd1904da3a6e7ce4c9b",
      "load_path": "INT8",
      "files": [
        {"path": "INT8/model.onnx", "size": 568511234, "sha256": "0e0a..."}
      ]
    }
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from .fetcher import FileSpec

SPEC_ENV_VAR = "MODEL_SPEC_JSON"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$|^[0-9a-f]{64}$")


class ModelSpecError(RuntimeError):
    """The injected model specification is missing, malformed, or inconsistent."""


@dataclass(frozen=True)
class ModelSpec:
    """Everything a service needs in order to initialise its model volume.

    ``load_path`` is relative to the published ``current`` directory and names
    the directory the service actually loads, so artifact layout is configuration
    rather than code.
    """

    repo_id: str
    revision: str
    load_path: str
    files: tuple[FileSpec, ...]


def load_model_spec(environ: dict[str, str] | None = None) -> ModelSpec:
    """Read and validate ``MODEL_SPEC_JSON`` from the environment."""
    source = os.environ if environ is None else environ
    raw = (source.get(SPEC_ENV_VAR) or "").strip()
    if not raw:
        raise ModelSpecError(
            f"{SPEC_ENV_VAR} is not set; the service must receive its model "
            "specification through Compose env_file or the environment"
        )
    return parse_model_spec(raw)


def parse_model_spec(raw: str) -> ModelSpec:
    """Validate a specification document and return its typed form."""
    try:
        document = json.loads(raw)
    except ValueError as exc:
        raise ModelSpecError(f"{SPEC_ENV_VAR} is not valid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise ModelSpecError(f"{SPEC_ENV_VAR} must be a JSON object")

    repo_id = _required_text(document, "repo_id")
    revision = _required_text(document, "revision").lower()
    if not _COMMIT_SHA.fullmatch(revision):
        raise ModelSpecError(
            f"{SPEC_ENV_VAR}: revision must be a full 40- or 64-character hex "
            f"commit SHA, got {revision!r}; branches and tags are not allowed"
        )

    load_path = _optional_text(document, "load_path", default=".")
    _require_relative(load_path, "load_path")

    entries = document.get("files")
    if not isinstance(entries, list) or not entries:
        raise ModelSpecError(f"{SPEC_ENV_VAR}: files must be a non-empty list")

    files: list[FileSpec] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        label = f"files[{index}]"
        if not isinstance(entry, dict):
            raise ModelSpecError(f"{SPEC_ENV_VAR}: {label} must be an object")
        path = _required_text(entry, "path", label=label)
        _require_relative(path, f"{label}.path")
        if path in seen:
            raise ModelSpecError(f"{SPEC_ENV_VAR}: duplicate file path {path!r}")
        seen.add(path)

        size = entry.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ModelSpecError(f"{SPEC_ENV_VAR}: {label}.size must be a positive integer")

        digest = _required_text(entry, "sha256", label=label).lower()
        if not _SHA256.fullmatch(digest):
            raise ModelSpecError(
                f"{SPEC_ENV_VAR}: {label}.sha256 must be 64 hexadecimal characters"
            )
        files.append(FileSpec(path, size, digest))

    return ModelSpec(
        repo_id=repo_id,
        revision=revision,
        load_path=load_path,
        files=tuple(files),
    )


def _required_text(container: dict, key: str, *, label: str | None = None) -> str:
    prefix = f"{label}." if label else ""
    value = container.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ModelSpecError(f"{SPEC_ENV_VAR}: {prefix}{key} must be a non-empty string")
    return value.strip()


def _optional_text(container: dict, key: str, *, default: str) -> str:
    value = container.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ModelSpecError(f"{SPEC_ENV_VAR}: {key} must be a non-empty string")
    return value.strip()


def _require_relative(path: str, label: str) -> None:
    candidate = PurePosixPath(path)
    if candidate.is_absolute() or ".." in candidate.parts or "\\" in path:
        raise ModelSpecError(
            f"{SPEC_ENV_VAR}: {label} must be a relative POSIX path without '..', "
            f"got {path!r}"
        )
