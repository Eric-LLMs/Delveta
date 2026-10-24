"""Fetch, verify, and publish a pinned Hugging Face checkpoint inside a volume.

A model service that owns a Docker-managed volume calls :func:`ensure_model`
before it starts serving. The volume layout is::

    <root>/versions/<revision>/...      immutable downloaded snapshot
    <root>/current -> versions/<rev>    the path the service loads from
    <root>/.fetch.lock                  cross-process initialisation lock
    <root>/.delveta-model.json          receipt of the last successful publish

Downloads land in a deterministic ``.incoming-<revision>`` directory and only
become visible by an atomic swap of ``current`` after every required file has
been size- and SHA256-verified. Bytes are fetched with plain HTTP Range requests
(no downloader library): each file is written in place, so a dropped connection
or a killed container leaves a prefix of the file behind and the next boot
continues from that offset instead of starting over. The staging directory is
keyed by revision and is never exposed under the load path until verified. The
lock makes two containers sharing one volume initialise it once instead of
racing.

Nothing here knows about Delveta business logic; each service owns its own
repo/revision/file specification and passes it in.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

_RECEIPT_NAME = ".delveta-model.json"
_LOCK_NAME = ".fetch.lock"
_CURRENT_LINK = "current"
_VERSIONS_DIR = "versions"
_STAGING_PREFIX = ".incoming-"
_DEFAULT_ENDPOINT = "https://huggingface.co"
# One request per file. The timeout applies to connect and to each blocking
# read, so a stalled transfer fails the boot instead of hanging forever.
_HTTP_TIMEOUT_SECONDS = 120
# A partial file smaller than its pin is resumed; the same file at the pinned
# size but with the wrong SHA256 is treated as corrupt and re-fetched once.
_DOWNLOAD_ATTEMPTS = 2


class ModelFetchError(RuntimeError):
    """The pinned checkpoint could not be downloaded, verified, or published."""


@dataclass(frozen=True)
class FileSpec:
    """One file the service cannot start without.

    ``path`` is relative to the repository root (so a subfolder snapshot keeps
    its subfolder), and both ``size`` and ``sha256`` are pinned so a silently
    changed upstream artifact is rejected instead of served.
    """

    path: str
    size: int
    sha256: str


def ensure_model(
    root: Path,
    *,
    repo_id: str,
    revision: str,
    files: Sequence[FileSpec],
    token: str | None = None,
    endpoint: str | None = None,
) -> Path:
    """Ensure ``root`` holds ``repo_id@revision`` and return the load path.

    Returns ``<root>/current``. Raises :class:`ModelFetchError` with an
    operator-readable message when the snapshot cannot be produced.
    """
    if not files:
        raise ModelFetchError("no required files were specified")
    if not revision:
        raise ModelFetchError("revision must be pinned; refusing to download an unpinned checkpoint")

    root = Path(root)
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ModelFetchError(f"model volume {root} is not writable: {exc}") from exc

    with _volume_lock(root / _LOCK_NAME):
        current = root / _CURRENT_LINK
        receipt = _read_receipt(root)
        if _receipt_matches(receipt, repo_id, revision, files) and _is_complete(current, files):
            _log(f"reusing {repo_id}@{revision[:12]} from {current}")
            return current

        _log(f"initialising {repo_id}@{revision[:12]} into {root}")
        _download(repo_id=repo_id, revision=revision, root=root, files=files,
                  token=token, endpoint=endpoint or os.environ.get("HF_ENDPOINT")
                  or _DEFAULT_ENDPOINT)
        if not _is_complete(current, files):
            raise ModelFetchError("published snapshot failed the post-publish completeness check")
        _write_receipt(root, repo_id, revision, files)
        _log(f"ready: {current} -> versions/{revision}")
        return current


def _download(
    *,
    repo_id: str,
    revision: str,
    root: Path,
    files: Sequence[FileSpec],
    token: str | None,
    endpoint: str,
) -> None:
    # Keyed by revision and never removed up front: whatever bytes a previous
    # attempt already wrote are the resume offset for this one.
    staging = root / f"{_STAGING_PREFIX}{revision}"
    # The lock is held, so nothing else is mid-download. Any other staging
    # directory belongs to a revision that is no longer pinned and cannot be
    # resumed, so drop it.
    for leftover in root.glob(f"{_STAGING_PREFIX}*"):
        if leftover != staging:
            _remove_tree(leftover)
    staging.mkdir(parents=True, exist_ok=True)

    # Attempt 1 resumes (or finishes) whatever is on disk. A failure that
    # leaves a correct prefix — a dropped connection, a killed container — is
    # re-raised as-is so the next boot continues from that byte offset. Only a
    # complete-length file with the wrong SHA256 is treated as corrupt, and that
    # earns exactly one clean re-download before failing closed.
    for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
        _fetch_all(staging, files=files, repo_id=repo_id, revision=revision,
                   token=token, endpoint=endpoint)
        broken = _first_integrity_failure(staging, files)
        if broken is None:
            break
        if attempt >= _DOWNLOAD_ATTEMPTS:
            raise ModelFetchError(broken)
        _log(f"{broken}; discarding {staging.name} and retrying once")
        _remove_tree(staging)
        staging.mkdir(parents=True, exist_ok=True)

    versions = root / _VERSIONS_DIR
    versions.mkdir(exist_ok=True)
    target = versions / revision
    _remove_tree(target)
    staging.rename(target)

    _swap_current(root, f"{_VERSIONS_DIR}/{revision}")
    _prune_versions(versions, keep=revision)


def _fetch_all(
    staging: Path,
    *,
    files: Sequence[FileSpec],
    repo_id: str,
    revision: str,
    token: str | None,
    endpoint: str,
) -> None:
    base = f"{endpoint.rstrip('/')}/{repo_id}/resolve/{revision}"
    for spec in files:
        _fetch_file(f"{base}/{spec.path}", staging / spec.path,
                    token=token, expected_size=spec.size)


def _fetch_file(url: str, dest: Path, *, token: str | None, expected_size: int) -> None:
    """Fetch ``url`` into ``dest``, resuming at whatever length already exists."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    if have > expected_size:
        _log(f"{dest.name}: partial file is larger than the pin; restarting it")
        dest.unlink()
        have = 0
    if have == expected_size:
        return
    if have:
        _log(f"{dest.name}: resuming at {have}/{expected_size} bytes")

    headers = {"User-Agent": "delveta-model-store/1", "Accept-Encoding": "identity"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if have:
        headers["Range"] = f"bytes={have}-"

    try:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT_SECONDS) as response:
            if response.status == 206:
                start = _content_range_start(response.headers.get("Content-Range", ""))
                if start != have:
                    dest.unlink(missing_ok=True)
                    raise ModelFetchError(
                        f"{dest.name}: server resumed at {start}, expected {have}; "
                        "deleted the partial file"
                    )
                mode = "ab"
            else:
                # The server ignored Range: this response is the whole file.
                mode = "wb"
                have = 0
            with dest.open(mode) as handle:
                shutil.copyfileobj(response, handle, 1 << 20)
    except ModelFetchError:
        raise
    except urllib.error.HTTPError as exc:
        if exc.code == 416:
            # The local file does not fit the remote range: start it clean.
            dest.unlink(missing_ok=True)
            raise ModelFetchError(
                f"{dest.name}: server rejected resume offset {have}; deleted the "
                "partial file so the next attempt starts clean"
            ) from exc
        raise ModelFetchError(f"{dest.name}: HTTP {exc.code} from {url}") from exc
    except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
        raise ModelFetchError(
            f"{dest.name}: download interrupted at "
            f"{dest.stat().st_size if dest.exists() else 0}/{expected_size} bytes: "
            f"{type(exc).__name__}: {exc}; the next attempt resumes"
        ) from exc

    size = dest.stat().st_size
    if size != expected_size:
        raise ModelFetchError(
            f"{dest.name}: expected {expected_size} bytes, got {size}; "
            "the next attempt resumes from here"
        )


def _content_range_start(value: str) -> int:
    """Return the first byte of ``Content-Range: bytes <start>-<end>/<total>``."""
    try:
        unit, span = value.split(" ", 1)
        if unit != "bytes":
            return -1
        return int(span.split("-", 1)[0])
    except (ValueError, IndexError):
        return -1


def _first_integrity_failure(staging: Path, files: Sequence[FileSpec]) -> str | None:
    """Return the first pin violation, or None when every file is exact."""
    for spec in files:
        path = staging / spec.path
        if not path.is_file():
            return f"required file missing after download: {spec.path}"
        size = path.stat().st_size
        if size != spec.size:
            return f"{spec.path}: expected {spec.size} bytes, got {size}"
        digest = _sha256(path)
        if digest != spec.sha256:
            return f"{spec.path}: sha256 mismatch (expected {spec.sha256}, got {digest})"
        _log(f"verified {spec.path} ({spec.size} bytes)")
    return None


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _is_complete(load_path: Path, files: Sequence[FileSpec]) -> bool:
    # Fast path: sizes only. Integrity is established once, at download time,
    # over an immutable versions/<revision> directory; re-hashing 1.3 GB on
    # every boot would add minutes for no new information.
    base = load_path.resolve() if load_path.exists() else None
    if base is None or not base.is_dir():
        return False
    for spec in files:
        candidate = base / spec.path
        if not candidate.is_file() or candidate.stat().st_size != spec.size:
            return False
    return True


def _swap_current(root: Path, relative_target: str) -> None:
    link = root / _CURRENT_LINK
    tmp = root / f".{_CURRENT_LINK}.tmp"
    try:
        if tmp.is_symlink() or tmp.exists():
            tmp.unlink()
    except OSError:
        pass
    os.symlink(relative_target, tmp)
    os.replace(tmp, link)


def _prune_versions(versions: Path, *, keep: str) -> None:
    for entry in versions.iterdir():
        if entry.is_dir() and entry.name != keep:
            _remove_tree(entry)


def _read_receipt(root: Path) -> dict | None:
    try:
        return json.loads((root / _RECEIPT_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_receipt(root: Path, repo_id: str, revision: str, files: Sequence[FileSpec]) -> None:
    payload = {
        "repo_id": repo_id,
        "revision": revision,
        "files": [{"path": f.path, "size": f.size, "sha256": f.sha256} for f in files],
    }
    tmp = root / f".{_RECEIPT_NAME}.tmp"
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, root / _RECEIPT_NAME)


def _receipt_matches(
    receipt: dict | None, repo_id: str, revision: str, files: Sequence[FileSpec]
) -> bool:
    if not isinstance(receipt, dict):
        return False
    expected = {
        "repo_id": repo_id,
        "revision": revision,
        "files": [{"path": f.path, "size": f.size, "sha256": f.sha256} for f in files],
    }
    return receipt == expected


def _remove_tree(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path, ignore_errors=True)


@contextmanager
def _volume_lock(lock_path: Path) -> Iterator[None]:
    import fcntl  # POSIX only; these containers are Linux by construction.

    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        _log(f"waiting for {lock_path.name}")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _log(message: str) -> None:
    print(f"[model-store] {message}", file=sys.stderr, flush=True)
