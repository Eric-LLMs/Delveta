"""Initialise the cap-router model volume from the injected spec, then exec the service.

The model specification arrives as ``MODEL_SPEC_JSON`` through Compose's
``env_file``, so it is read at container start from outside the image: changing a
model version means editing ``deploy/model_config/layachoice.env`` and recreating
the container. No image rebuild and no host bind mount is involved, and the
checkpoint stays in the Docker-managed ``/models/layachoice`` volume.

Any failure exits non-zero before the HTTP server starts, so a missing or
half-written checkpoint is never reported as a healthy service.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, "/app")

from model_store import ModelFetchError, ModelSpecError, ensure_model, load_model_spec  # noqa: E402


def main() -> None:
    root = Path(os.environ.get("LAYA_MODEL_VOLUME") or "/models/layachoice")
    try:
        spec = load_model_spec()
        print(
            f"[cap-router] spec: {spec.repo_id}@{spec.revision[:12]} "
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
        print(f"[cap-router] model initialisation failed: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc

    load_dir = current / spec.load_path
    if not load_dir.is_dir():
        print(
            f"[cap-router] model initialisation failed: {load_dir} does not exist "
            "after publish; check load_path in the model specification",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1)

    os.environ["LAYA_CHOICE_MODEL_DIR"] = str(load_dir)
    print(f"[cap-router] serving from {load_dir}", flush=True)

    if len(sys.argv) < 2:
        raise SystemExit("[cap-router] no service command supplied after bootstrap")
    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    main()
