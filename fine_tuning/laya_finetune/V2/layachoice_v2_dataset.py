#!/usr/bin/env python3
"""LayaChoice-v2 dataset contract — adapter + loader.

The installed bundle is a plain supervised dataset: every row is exactly

    {"message_id": "<uuid4>", "language": "<lang>",
     "input": {"query": "<request text>",
               "options": ["<capability_id>: <card>", ...]},
     "label": <0-based index of the correct option>}

``input.options`` is the EXACT text the model sees, one string per option, in
option order. ``label`` points at one of them: a capability card, or the REJECT
card. There is no padding: K recalled candidates + REJECT = K+1 options (1..4).

``message_id`` and ``language`` are sample metadata. They never enter the model
input, the target, calibration or cross-version alignment; ``message_id`` is a
fresh uuid4 in every threshold bundle. ``split`` is not stored in the row — the
file name is the authority and ``load_split`` injects it.

This module owns the dataset contract ONLY — it never touches a tokenizer or a
model. The adapter derives the training-time view (``order``, ``options`` cards,
``gold_index``) from ``input`` so the renderer keeps the same shape it always
had:

  * ``order``      — the capability id of each option, in option order;
  * ``options``    — the card text of each option (the part after ``"<id>: "``);
  * ``gold_index`` — ``label``.

Metadata fields such as ``target_kind`` / ``gold_capability_id`` /
``reject_index`` are properties derived from those three, never stored.
"""
from __future__ import annotations

import json
import random
import sys
import uuid
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
ROW_KEYS = frozenset({"message_id", "language", "input", "label"})
INPUT_KEYS = frozenset({"query", "options"})

# The ONLY keys a renderer may ever read to build the model input.
MODEL_INPUT_KEYS = ("state", "order", "options")

TARGET_KINDS = frozenset({"capability", "reject"})


@dataclass(frozen=True)
class V2Example:
    """One frozen row, adapted. ``order``/``options`` are tuples so the record is
    hashable and cannot be mutated by a downstream renderer."""
    message_id: str
    split: str
    language: str
    state: str
    order: tuple[str, ...]
    options: tuple[str, ...]
    gold_index: int

    @property
    def model_input(self) -> dict:
        """The closed set of fields a renderer is allowed to read."""
        return {"state": self.state, "order": self.order, "options": self.options}

    @property
    def rendered_options(self) -> tuple[str, ...]:
        """The row's ``input.options`` strings, reconstructed exactly."""
        return tuple(cid if not card else f"{cid}: {card}"
                     for cid, card in zip(self.order, self.options))

    @property
    def target_index(self) -> int:
        """The one-hot target position (== ``label``)."""
        return self.gold_index

    @property
    def gold_id(self) -> str:
        """The label at the target slot (a capability id, or ``REJECT``)."""
        return self.order[self.gold_index]

    @property
    def target_kind(self) -> str:
        return "reject" if self.gold_id == S.REJECT_LABEL else "capability"

    @property
    def is_reject(self) -> bool:
        return self.target_kind == "reject"

    @property
    def gold_capability_id(self) -> str | None:
        return None if self.is_reject else self.gold_id

    @property
    def reject_index(self) -> int:
        return self.order.index(S.REJECT_LABEL)


def _parse_option(option: str, row_id: str) -> tuple[str, str]:
    """Split one rendered option back into (capability_id, card text).

    ``render_options`` emits ``"<id>: <card>"`` for a described option and the
    bare ``"<id>"`` when the card is empty. Anything that does not round-trip
    through that rule is not a valid model input.
    """
    cid, sep, card = option.partition(": ")
    if not sep:
        cid, card = option, ""
    if not cid.strip():
        raise ValueError(f"{row_id}: option {option!r} has no capability id")
    # Capability ids never contain ":" and never pad with whitespace; anything
    # else cannot be split back into (id, card) unambiguously.
    if ":" in cid or cid != cid.strip():
        raise ValueError(f"{row_id}: option {option!r} is not canonical "
                         f"(capability id {cid!r})")
    rebuilt = cid if not card else f"{cid}: {card}"
    if rebuilt != option:
        raise ValueError(f"{row_id}: option {option!r} is not canonical "
                         f"(expected {rebuilt!r})")
    return cid, card


def _valid_uuid4(value: str, row_id: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError(f"{row_id}: message_id {value!r} is not a canonical uuid4") from exc
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError(f"{row_id}: message_id {value!r} is not a canonical uuid4")
    return value


def adapt_row(raw: dict, split: str) -> V2Example:
    """Adapter: one frozen JSON row -> a validated ``V2Example``.

    ``split`` is the file-level fact and must be passed by the loader; it is
    never read from the row. Fails loudly on any schema or semantic drift.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    keys = set(raw)
    if keys != ROW_KEYS:
        missing = ROW_KEYS - keys
        extra = keys - ROW_KEYS
        raise ValueError(f"schema drift (missing={sorted(missing)}, "
                         f"extra={sorted(extra)})")

    message_id = _valid_uuid4(raw["message_id"], str(raw["message_id"]))
    language = raw["language"]
    if not isinstance(language, str) or not language.strip():
        raise ValueError(f"{message_id}: language must be a non-empty string")

    inp = raw["input"]
    if not isinstance(inp, dict) or set(inp) != INPUT_KEYS:
        raise ValueError(f"{message_id}: input must carry exactly {sorted(INPUT_KEYS)}")
    state = inp["query"]
    if not isinstance(state, str) or not state.strip():
        raise ValueError(f"{message_id}: input.query must be a non-empty string")
    raw_options = inp["options"]
    if not isinstance(raw_options, list):
        raise ValueError(f"{message_id}: input.options must be a list")
    n_options = len(raw_options)
    if not (S.MIN_OPTION_SLOTS <= n_options <= S.MAX_OPTION_SLOTS):
        raise ValueError(f"{message_id}: expected {S.MIN_OPTION_SLOTS}.."
                         f"{S.MAX_OPTION_SLOTS} options, got {n_options}")
    if any(not isinstance(o, str) or not o.strip() for o in raw_options):
        raise ValueError(f"{message_id}: an option is empty or not a string")

    parsed = [_parse_option(o, message_id) for o in raw_options]
    order = tuple(cid for cid, _ in parsed)
    if len(set(order)) != len(order):
        raise ValueError(f"{message_id}: duplicate capability ids in options {order}")
    reject_slots = [i for i, cid in enumerate(order) if cid == S.REJECT_LABEL]
    if len(reject_slots) != 1:
        raise ValueError(f"{message_id}: REJECT must appear exactly once, "
                         f"found at {reject_slots}")

    label = raw["label"]
    if isinstance(label, bool) or not isinstance(label, int):
        raise ValueError(f"{message_id}: label must be an integer")
    if not (0 <= label < n_options):
        raise ValueError(f"{message_id}: label {label} out of range for "
                         f"{n_options} options")

    return V2Example(
        message_id=message_id, split=split, language=language, state=state,
        order=order, options=tuple(card for _, card in parsed),
        gold_index=label,
    )


def option_slot_bounds(manifest: dict) -> tuple[int, int]:
    """The option-count contract stated by a manifest.

    Frozen V2 declares an integer (4). A dynamic bundle declares the same rule
    production uses, ``K+1``. Both are validated; an unknown shape is rejected.
    """
    slots = manifest.get("option_slots")
    if isinstance(slots, int) and not isinstance(slots, bool):
        if slots != S.OPTION_SLOTS:
            raise RuntimeError(f"manifest option_slots {slots} != {S.OPTION_SLOTS}")
        return slots, slots
    if isinstance(slots, dict) and slots.get("rule") == "K+1":
        lo, hi = slots.get("min"), slots.get("max")
        if lo != S.MIN_OPTION_SLOTS or hi != S.MAX_OPTION_SLOTS:
            raise RuntimeError(f"manifest option_slots range {lo}..{hi} != "
                               f"{S.MIN_OPTION_SLOTS}..{S.MAX_OPTION_SLOTS}")
        return lo, hi
    raise RuntimeError(f"manifest option_slots is not a valid contract: {slots!r}")


def read_manifest(data_dir: Path = DATA) -> dict:
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("bench") != S.V2_BENCH:
        raise RuntimeError(f"manifest bench {manifest.get('bench')!r} != {S.V2_BENCH!r}")
    if set(manifest.get("training_schema_keys", [])) != ROW_KEYS:
        raise RuntimeError(f"manifest training_schema_keys "
                           f"{manifest.get('training_schema_keys')!r} != {sorted(ROW_KEYS)}")
    option_slot_bounds(manifest)
    return manifest


def split_path(split: str, data_dir: Path = DATA) -> Path:
    """Resolve one split file for either installed layout.

    The formal flat bundle uses ``{split}.jsonl``; the archived threshold
    bundles keep the historical ``v2_{split}.jsonl`` names. The file bytes are
    the same contract in both layouts, so callers never branch on the name.
    """
    formal = data_dir / f"{split}.jsonl"
    if formal.is_file():
        return formal
    archived = data_dir / f"v2_{split}.jsonl"
    if archived.is_file():
        return archived
    raise FileNotFoundError(
        f"{split}: neither {formal.name} nor {archived.name} under {data_dir}")


def load_split(split: str, data_dir: Path = DATA, limit: int | None = None) -> list[V2Example]:
    """Load one split and prove it still describes the frozen manifest."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    path = split_path(split, data_dir)
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(adapt_row(json.loads(line), split))
    if limit is not None:
        return rows[:limit]

    manifest = read_manifest(data_dir)
    expected = manifest["rows"][split]
    if len(rows) != expected:
        raise RuntimeError(f"{split}: {len(rows)} rows, manifest says {expected}")
    ids = [r.message_id for r in rows]
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"{split}: duplicate message_id in the split")
    lo, hi = option_slot_bounds(manifest)
    out_of_contract = [r.message_id for r in rows if not lo <= len(r.options) <= hi]
    if out_of_contract:
        raise RuntimeError(f"{split}: option count outside manifest contract "
                           f"{lo}..{hi} for {out_of_contract[:5]}")
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
    indices, after proving those indices are reproducible.

    Calibration membership is positional. ``message_id`` is deliberately not
    consulted: it is a random per-bundle identifier and has no role in training.
    """
    calib_idx, train_idx = calibration_split(len(examples))
    recorded = manifest["calibration"]["indices"]
    if calib_idx != recorded:
        raise RuntimeError(
            f"calibration indices are not reproducible: recomputed {len(calib_idx)} rows "
            f"differ from the {len(recorded)} recorded in manifest.json")
    return [examples[i] for i in calib_idx], [examples[i] for i in train_idx]
