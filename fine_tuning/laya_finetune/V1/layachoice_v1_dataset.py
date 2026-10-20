#!/usr/bin/env python3
"""LayaChoice-v1 dataset contract — adapter + loader.

Every row is exactly four fields:

    {"message_id": "<sample id>", "language": "<lang>",
     "input": {"query": "<request text>",
               "options": ["<capability_id>: <card>", ...]},
     "label": <0-based index of the correct option>}

``input.options`` is the EXACT model input, one string per option in option
order; ``label`` points at one of them. ``K = 3`` is frozen for v1 (gold + two
distractors); there is no REJECT option and no padding.

``message_id`` and ``language`` are sample metadata. They never enter the model
input, the target, calibration or alignment. ``split`` is not stored in the
row — the file name (``train.jsonl`` / ``val.jsonl`` / ``test.jsonl``) is the
authority and ``load_bundle`` injects it.

This module owns the dataset contract only — it never touches a tokenizer, a
model or a registry. ``laya.common.render_options`` renders a choice option as ``"<id>: <card>"``,
so the stored option string IS the model-visible option; the adapter splits it
so the v1 renderer keeps its ``order`` + card shape and rebuilds the identical
string. The adapter also derives ``d1``/``d2`` so the report keeps its shape:

  * ``order``      — the capability id of each option, in option order;
  * ``options``    — the card text of each option (the part after ``"<id>: "``);
  * ``gold``       — ``label``;
  * ``d1``/``d2``  — the frozen distractor slots (scores 0.62 / 0.55), a pure
                     function of the gold capability.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SPLITS = ("train", "val", "test")

# The exact frozen schema. An unexpected key means the bundle is not the frozen
# four-field revision (or carries pre-migration metadata).
ROW_KEYS = frozenset({"message_id", "language", "input", "label"})
INPUT_KEYS = frozenset({"query", "options"})
OPTION_SLOTS = 3

# The frozen distractor roles (score 0.62 / 0.55). These were stored per row as
# d1/d2 before the four-field migration; they are a pure function of the gold
# capability, so the bundle no longer carries them.
DISTRACTORS = {
    "cap-add-term":          ("cap-create-folder", "cap-edit-file"),
    "cap-artifact":          ("cap-slides", "cap-summary"),
    "cap-bash":              ("cap-edit-file", "cap-read-file"),
    "cap-create-folder":     ("cap-add-term", "cap-bash"),
    "cap-edit-file":         ("cap-bash", "cap-read-file"),
    "cap-mindmap":           ("cap-slides", "cap-summary"),
    "cap-pdf-extract-text":  ("cap-pdf-table-to-text", "cap-read-document"),
    "cap-pdf-table-to-text": ("cap-pdf-extract-text", "cap-summary"),
    "cap-rag-search":        ("cap-web-search", "cap-social-search"),
    "cap-read-document":     ("cap-pdf-extract-text", "cap-read-file"),
    "cap-read-file":         ("cap-read-document", "cap-edit-file"),
    "cap-slides":            ("cap-mindmap", "cap-artifact"),
    "cap-social-search":     ("cap-web-search", "cap-rag-search"),
    "cap-summary":           ("cap-mindmap", "cap-artifact"),
    "cap-translate":         ("cap-summary", "cap-read-document"),
    "cap-vision":            ("cap-read-document", "cap-pdf-extract-text"),
    "cap-web-search":        ("cap-social-search", "cap-rag-search"),
}


def _parse_option(option: str, row_id: str) -> tuple[str, str]:
    """Split one stored option back into (capability_id, card text).

    The bundle stores ``"<id>: <card>"``. Anything that does not round-trip
    through that rule is not a valid model input.
    """
    cid, sep, card = option.partition(": ")
    if not sep or not cid.strip() or ":" in cid:
        raise RuntimeError(f"{row_id}: option {option[:80]!r} is not '<id>: <card>'")
    return cid, card


def adapt_row(raw: dict, split: str) -> dict:
    """Adapter: one frozen JSON row -> the renderer's compact row shape.

    The stored contract is four fields only; ``order``/``options``/``d1``/``d2``
    are derived here. The renderer rebuilds ``"<id>: <card>"`` from ``order`` and
    the card, so the model sees exactly the stored option string.
    """
    if split not in SPLITS:
        raise RuntimeError(f"unknown split {split!r}; expected one of {SPLITS}")
    keys = set(raw)
    if keys != ROW_KEYS:
        raise RuntimeError(f"schema drift (missing={sorted(ROW_KEYS - keys)}, "
                           f"extra={sorted(keys - ROW_KEYS)})")
    message_id = str(raw["message_id"])
    language = raw["language"]
    if not language or not isinstance(language, str):
        raise RuntimeError(f"{message_id}: language must be a non-empty string")
    inp = raw["input"]
    if not isinstance(inp, dict) or set(inp) != INPUT_KEYS:
        raise RuntimeError(f"{message_id}: input must carry exactly {sorted(INPUT_KEYS)}")
    state = inp["query"]
    if not isinstance(state, str) or not state.strip():
        raise RuntimeError(f"{message_id}: input.query must be a non-empty string")
    stored = inp["options"]
    if not isinstance(stored, list) or len(stored) != OPTION_SLOTS:
        raise RuntimeError(f"{message_id}: expected {OPTION_SLOTS} options, "
                           f"got {len(stored) if isinstance(stored, list) else type(stored)}")
    parsed = [_parse_option(o, message_id) for o in stored]
    order = [cid for cid, _ in parsed]
    if len(set(order)) != len(order):
        raise RuntimeError(f"{message_id}: duplicate capability ids in options {order}")
    label = raw["label"]
    if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < OPTION_SLOTS:
        raise RuntimeError(f"{message_id}: label {label!r} is not a slot in 0..{OPTION_SLOTS - 1}")
    pair = DISTRACTORS.get(order[label])
    if pair is None or pair[0] not in order or pair[1] not in order:
        raise RuntimeError(f"{message_id}: frozen distractors {pair} absent for gold {order[label]}")
    return {
        "id": message_id, "split": split, "lang": language, "state": state,
        "order": order, "options": [card for _, card in parsed], "gold": label,
        "d1": order.index(pair[0]), "d2": order.index(pair[1]),
    }


def read_manifest(data_dir: Path = DATA) -> dict:
    return json.loads((Path(data_dir) / "manifest.json").read_text(encoding="utf-8"))


def load_bundle(split: str, data_dir: Path = DATA, limit: int | None = None) -> list[dict]:
    """Read one split and prove it still describes the frozen manifest."""
    if split not in SPLITS:
        raise RuntimeError(f"unknown split {split!r}; expected one of {SPLITS}")
    data_dir = Path(data_dir)
    rows = [adapt_row(json.loads(line), split)
            for line in (data_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if limit is not None:
        return rows[:limit]
    manifest = read_manifest(data_dir)
    if set(manifest.get("training_schema_keys", [])) != ROW_KEYS:
        raise RuntimeError(f"manifest training_schema_keys != {sorted(ROW_KEYS)}")
    if manifest.get("option_slots") != OPTION_SLOTS:
        raise RuntimeError(f"manifest option_slots {manifest.get('option_slots')!r} "
                           f"!= {OPTION_SLOTS}")
    expected = manifest["rows"][split]
    if len(rows) != expected:
        raise RuntimeError(f"{split}: {len(rows)} rows, manifest says {expected}")
    if len({r["id"] for r in rows}) != len(rows):
        raise RuntimeError(f"{split}: duplicate message_id in the split")
    return rows
