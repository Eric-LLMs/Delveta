#!/usr/bin/env python3
"""LayaChoice-v2 dataset contract — adapter + loader.

The V2 bundle is the accepted 4-way task: every row carries three capability
cards plus one REJECT card. This module owns the dataset contract ONLY — it never
touches a tokenizer or a model.

Two responsibilities live here because they share one frozen schema:

  * the ADAPTER  (``V2Example`` / ``adapt_row``) turns one frozen JSON row into a
    validated record whose MODEL INPUT is exactly ``{state, order, options}``.
    Every other frozen key (``part``, ``source_query_id``, ``src_sha``,
    ``shuffle``, ``gold_index``, ...) is metadata: carried for validation, audit
    and the target, and structurally excluded from what a renderer may read.
  * the LOADER   (``read_manifest`` / ``load_split``) reads ``v2_{split}.jsonl``
    and proves each split still describes the frozen manifest (row count, split
    tag, and every row's source sha declared for that split).

``gold_index`` (0..3) is the FINAL target position — the slot the model must pick.
For a ``capability`` row that slot holds the gold capability; for a ``reject`` row
it holds the REJECT card (and equals ``reject_index``).
"""
from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import layachoice_v2_spec as S  # noqa: E402

DATA = HERE / "data"
SPLITS = ("train", "val", "test")

# The exact frozen schema. An unexpected key means the bundle is not the frozen
# revision (or carries construction metadata that must never reach the model).
ROW_KEYS = frozenset({
    "id", "source_query_id", "split", "part", "lang", "state", "order", "options",
    "gold_index", "target_kind", "gold_capability_id", "reject_index", "shuffle", "src_sha",
})

# The ONLY keys a renderer may ever read to build the model input.
MODEL_INPUT_KEYS = ("state", "order", "options")

TARGET_KINDS = frozenset({"capability", "reject"})
PARTITIONS = frozenset({"p1", "p2"})


@dataclass(frozen=True)
class V2Example:
    """One frozen row, adapted. ``order``/``options`` are tuples so the record is
    hashable and cannot be mutated by a downstream renderer."""
    id: str
    source_query_id: str
    split: str
    part: str
    lang: str
    state: str
    order: tuple[str, ...]
    options: tuple[str, ...]
    gold_index: int
    target_kind: str
    gold_capability_id: str | None
    reject_index: int
    shuffle_seed_prefix: str
    shuffle_permutation: tuple[int, ...]
    src_sha: str

    @property
    def model_input(self) -> dict:
        """The closed set of fields a renderer is allowed to read."""
        return {"state": self.state, "order": self.order, "options": self.options}

    @property
    def target_index(self) -> int:
        """The one-hot target position (== ``gold_index``)."""
        return self.gold_index

    @property
    def is_reject(self) -> bool:
        return self.target_kind == "reject"

    @property
    def gold_id(self) -> str:
        """The label at the target slot (a capability id, or ``REJECT``)."""
        return self.order[self.gold_index]


def adapt_row(raw: dict) -> V2Example:
    """Adapter: one frozen JSON row -> a validated ``V2Example``.

    Fails loudly on any schema or semantic drift, so a bundle that is not the
    frozen revision can never be trained on by accident.
    """
    keys = set(raw)
    if keys != ROW_KEYS:
        missing = ROW_KEYS - keys
        extra = keys - ROW_KEYS
        raise ValueError(f"{raw.get('id')}: schema drift (missing={sorted(missing)}, "
                         f"extra={sorted(extra)})")

    order = tuple(raw["order"])
    options = tuple(raw["options"])
    if len(order) != S.OPTION_SLOTS or len(options) != S.OPTION_SLOTS:
        raise ValueError(f"{raw['id']}: expected {S.OPTION_SLOTS} options, "
                         f"got order={len(order)} options={len(options)}")
    if any(not str(o).strip() for o in options):
        raise ValueError(f"{raw['id']}: an option card is empty")

    reject_slots = [i for i, cid in enumerate(order) if cid == S.REJECT_LABEL]
    if reject_slots != [raw["reject_index"]]:
        raise ValueError(f"{raw['id']}: REJECT must appear exactly once at reject_index "
                         f"{raw['reject_index']}, found at {reject_slots}")

    gi = raw["gold_index"]
    if not (0 <= gi < S.OPTION_SLOTS):
        raise ValueError(f"{raw['id']}: gold_index {gi} out of range")
    kind = raw["target_kind"]
    if kind not in TARGET_KINDS:
        raise ValueError(f"{raw['id']}: unknown target_kind {kind!r}")
    gold_cap = raw["gold_capability_id"]
    if kind == "reject":
        if gi != raw["reject_index"]:
            raise ValueError(f"{raw['id']}: reject row gold_index {gi} != "
                             f"reject_index {raw['reject_index']}")
        if gold_cap is not None:
            raise ValueError(f"{raw['id']}: reject row carries gold_capability_id {gold_cap!r}")
    else:
        if gi == raw["reject_index"]:
            raise ValueError(f"{raw['id']}: capability row points at the REJECT slot")
        if order[gi] != gold_cap:
            raise ValueError(f"{raw['id']}: order[gold_index]={order[gi]!r} != "
                             f"gold_capability_id {gold_cap!r}")

    if raw["part"] not in PARTITIONS:
        raise ValueError(f"{raw['id']}: unknown part {raw['part']!r}")

    shuf = raw["shuffle"]
    perm = tuple(shuf["permutation"])
    if sorted(perm) != list(range(S.OPTION_SLOTS)):
        raise ValueError(f"{raw['id']}: shuffle permutation {perm} is not a permutation "
                         f"of 0..{S.OPTION_SLOTS - 1}")

    return V2Example(
        id=str(raw["id"]), source_query_id=str(raw["source_query_id"]),
        split=str(raw["split"]), part=str(raw["part"]), lang=str(raw["lang"]),
        state=str(raw["state"]), order=order, options=options,
        gold_index=int(gi), target_kind=str(kind),
        gold_capability_id=(None if gold_cap is None else str(gold_cap)),
        reject_index=int(raw["reject_index"]),
        shuffle_seed_prefix=str(shuf["seed_prefix"]), shuffle_permutation=perm,
        src_sha=str(raw["src_sha"]),
    )


def read_manifest(data_dir: Path = DATA) -> dict:
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("bench") != S.V2_BENCH:
        raise RuntimeError(f"manifest bench {manifest.get('bench')!r} != {S.V2_BENCH!r}")
    if manifest.get("option_slots") != S.OPTION_SLOTS:
        raise RuntimeError(f"manifest option_slots {manifest.get('option_slots')} "
                           f"!= {S.OPTION_SLOTS}")
    return manifest


def load_split(split: str, data_dir: Path = DATA, limit: int | None = None) -> list[V2Example]:
    """Load one split and prove it still describes the frozen manifest."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    path = data_dir / f"v2_{split}.jsonl"
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(adapt_row(json.loads(line)))
    if limit is not None:
        return rows[:limit]

    manifest = read_manifest(data_dir)
    expected = manifest["rows"][split]
    if len(rows) != expected:
        raise RuntimeError(f"{split}: {len(rows)} rows, manifest says {expected}")
    if {r.split for r in rows} != {split}:
        raise RuntimeError(f"{split}: rows carry a foreign split tag")
    # A split may legitimately draw from more than one frozen source (e.g. after the
    # 2026-10-02 Test->Train move, Train carries both the train and the test source
    # sha). When the manifest declares the split's provenances, every row must match
    # one; without a declaration the original single-source invariant still holds.
    declared = manifest.get("split_provenance", {}).get(split)
    present = {r.src_sha for r in rows}
    if declared is None:
        if len(present) != 1:
            raise RuntimeError(f"{split}: rows do not carry a single frozen source sha")
    elif present - set(declared):
        raise RuntimeError(f"{split}: rows carry undeclared source sha(s) "
                           f"{sorted(present - set(declared))}; declared {sorted(declared)}")
    return rows


def calibration_split(n_train: int) -> tuple[list[int], list[int]]:
    """The fixed held-out calibration indices: min(CALIB_MAX, n//10) rows, seed 20260922."""
    order = list(range(n_train))
    random.Random(S.CALIB_SEED).shuffle(order)
    n_calib = min(S.CALIB_MAX, n_train // 10)
    return sorted(order[:n_calib]), sorted(order[n_calib:])


def frozen_calibration(examples: list[V2Example],
                       manifest: dict) -> tuple[list[V2Example], list[V2Example]]:
    """Split train into (calibration, effective-train) by the manifest's recorded
    indices, after proving those indices are reproducible."""
    calib_idx, train_idx = calibration_split(len(examples))
    recorded = manifest["calibration"]["indices"]
    if calib_idx != recorded:
        raise RuntimeError(
            f"calibration indices are not reproducible: recomputed {len(calib_idx)} rows "
            f"differ from the {len(recorded)} recorded in manifest.json")
    ids = [examples[i].id for i in calib_idx]
    if ids != manifest["calibration"]["ids"]:
        raise RuntimeError("calibration ids are not reproducible from the recorded indices")
    return [examples[i] for i in calib_idx], [examples[i] for i in train_idx]
