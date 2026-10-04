#!/usr/bin/env python3
"""Strict row-alignment check between a dual-target JSONL and its manifest.

The dual-target training set is two row-aligned files:

  * ``bge_m3_<split>.jsonl``          -- training rows ``{query, pos, neg}``
  * ``bge_m3_<split>_manifest.jsonl`` -- provenance, one row per training row

Row ``i`` of the manifest describes row ``i`` of the data file. Training never
reads the manifest as a signal: ``task`` / ``declared_lang`` / ``language_profile``
/ ``source`` / ``seed_family`` / ``negative_provenance`` are recorded provenance
only. The single manifest field the training path consumes is
``gold_capability_id``, and only to keep at most one record per capability in a
batch.

That sampler contract holds only if the training index ``i`` maps to manifest
row ``i``. This module recomputes the alignment from bytes on disk and fails
loudly on any mismatch, so a silently reordered / truncated / regenerated data
file can never make the sampler's index -> capability map wrong.

Read-only: opens files, returns a report, raises on mismatch. Writes nothing.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Union

# Row / manifest keys that must be present. The data row keys are exactly the
# training-target fields; the manifest keys below are the ones training relies
# on for alignment and grouping.
DATA_REQUIRED_KEYS = ("query", "pos", "neg")
MANIFEST_REQUIRED_KEYS = ("gold_capability_id", "task", "query_sha256")

# Provenance fields recorded (never branched on) in the report.
MANIFEST_METADATA_KEYS = (
    "task",
    "declared_lang",
    "language_profile",
    "source",
    "seed_family",
    "negative_provenance",
)


class AlignmentError(ValueError):
    """Raised when a data file and its manifest are not strictly row-aligned."""


def sha256_text(text: str) -> str:
    """SHA-256 of ``text`` encoded as UTF-8 (the manifest's ``query_sha256``)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_jsonl(path: Union[str, Path]) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise AlignmentError(f"{path}: line {lineno} is not valid JSON: {exc}") from exc
    return rows


def validate(
    data_path: Union[str, Path],
    manifest_path: Union[str, Path],
) -> dict:
    """Verify ``data_path`` and ``manifest_path`` are strictly row-aligned.

    Raises :class:`AlignmentError` on the first violation. Returns a
    JSON-serialisable report on success.

    Checks, in order:
      1. equal row counts;
      2. every data row has the ``query/pos/neg`` keys, with non-empty
         ``pos``/``neg`` lists and a non-empty ``query`` string;
      3. every manifest row has ``gold_capability_id``/``task``/``query_sha256``;
      4. row ``i`` alignment: ``sha256(data[i]["query"]) == manifest[i]["query_sha256"]``.
    """
    data_path = Path(data_path)
    manifest_path = Path(manifest_path)

    rows = _read_jsonl(data_path)
    manifest = _read_jsonl(manifest_path)

    if len(rows) != len(manifest):
        raise AlignmentError(
            f"row count mismatch: {data_path.name}={len(rows)} vs "
            f"{manifest_path.name}={len(manifest)}"
        )

    for i, (row, meta) in enumerate(zip(rows, manifest)):
        missing_data = [k for k in DATA_REQUIRED_KEYS if k not in row]
        if missing_data:
            raise AlignmentError(f"{data_path.name} row {i}: missing keys {missing_data}")
        if not isinstance(row["query"], str) or not row["query"]:
            raise AlignmentError(f"{data_path.name} row {i}: 'query' must be a non-empty string")
        for field in ("pos", "neg"):
            value = row[field]
            if not isinstance(value, list) or not value:
                raise AlignmentError(f"{data_path.name} row {i}: '{field}' must be a non-empty list")

        missing_meta = [k for k in MANIFEST_REQUIRED_KEYS if k not in meta]
        if missing_meta:
            raise AlignmentError(f"{manifest_path.name} row {i}: missing keys {missing_meta}")
        if not isinstance(meta["gold_capability_id"], str) or not meta["gold_capability_id"]:
            raise AlignmentError(
                f"{manifest_path.name} row {i}: 'gold_capability_id' must be a non-empty string"
            )

        actual = sha256_text(row["query"])
        if actual != meta["query_sha256"]:
            raise AlignmentError(
                f"row {i} is misaligned: sha256({data_path.name} query)={actual} "
                f"!= {manifest_path.name}.query_sha256={meta['query_sha256']}"
            )

    report = {
        "data_file": str(data_path),
        "manifest_file": str(manifest_path),
        "rows": len(rows),
        "sha256_checked": len(rows),
        "sha256_mismatches": 0,
        # Provenance only -- recorded, never used as a training signal.
        "metadata": {
            key: dict(Counter(m.get(key) for m in manifest))
            for key in MANIFEST_METADATA_KEYS
        },
    }
    return report


if __name__ == "__main__":  # pragma: no cover - thin CLI
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", help="path to bge_m3_<split>.jsonl")
    parser.add_argument("manifest", help="path to bge_m3_<split>_manifest.jsonl")
    ns = parser.parse_args()
    print(json.dumps(validate(ns.data, ns.manifest), ensure_ascii=False, indent=2))
