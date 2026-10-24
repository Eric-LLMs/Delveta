"""Initialise the embedding model volume from the injected spec, then exec the service.

The model specification arrives as ``MODEL_SPEC_JSON`` through Compose's
``env_file``, so it is read at container start from outside the image: changing a
model version means editing ``deploy/model_config/embedding.env`` and recreating
the container. No image rebuild and no host bind mount is involved, and the
weights themselves stay in the Docker-managed ``/model`` volume.

Any failure exits non-zero before uvicorn can bind the port, so an incomplete
download is never mistaken for a serving container.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, "/app")

from model_store import ModelFetchError, ModelSpecError, ensure_model, load_model_spec  # noqa: E402


def main() -> None:
    root = Path(os.environ.get("MODEL_VOLUME") or "/model")
    try:
        spec = load_model_spec()
        print(
            f"[embedding] spec: {spec.repo_id}@{spec.revision[:12]} "
            f"({len(spec.files)} files, load_path={spec.load_path})",
            flush=True,
        )
        current = ensure_model(
            root,
            repo_id=spec.repo_id,
            revision=spec.revision,
            files=spec.files,
            token=os.environ.get("HF_TOKEN") or None,
        )
    except (ModelSpecError, ModelFetchError) as exc:
        print(f"[embedding] model initialisation failed: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc

    load_dir = current / spec.load_path
    if not load_dir.is_dir():
        print(
            f"[embedding] model initialisation failed: {load_dir} does not exist "
            "after publish; check load_path in the model specification",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1)

    os.environ["MODEL_DIR"] = str(load_dir)
    print(f"[embedding] serving from {load_dir}", flush=True)

    if len(sys.argv) < 2:
        raise SystemExit("[embedding] no service command supplied after bootstrap")
    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    main()
