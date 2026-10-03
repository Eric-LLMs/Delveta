#!/usr/bin/env python3
"""Stage 1 - freeze LayaChoice-v1 into self-contained training bundles.

Runs on the REPO HOST, once. The GPU box has no registry/DB, so everything the
training and evaluation stages need is materialised here and committed under
scripts/laya_finetune/data/.

Sources (read-only, never modified):
  train  logs/_laya_ds/LayaChoice_v1_train.jsonl          (rendered, 856 rows)
  test   logs/_laya_ds/LayaChoice_v1_test_v3.jsonl        (rendered, 900 rows)
  val    data/LayaChoice_v1_validation_raw_v3.jsonl       (RAW, 150 rows)

Train and test are already rendered, so their cards are read straight out of
`views.B_noprov.tool`. Validation was never ingested, so it is rendered HERE, in
memory, through the same shared contract the other two went through
(logs/_laya_choice_row.build_views + logs/_cap_only_eval.DISTRACTORS) and only the
resulting compact bundle is written - `data/LayaChoice_v1_validation_raw_v3.jsonl`
itself is never touched, and no `LayaChoice_v1_validation.jsonl` is produced.

Formal card view is B_noprov (frozen decision): the production card with
the `query examples:` block and the `evidence:` provenance line removed. Reading
`questions.tool.criteria` instead would silently train on view B, which still
carries the provenance line that names the answer in 856/856 train rows.

Output per split, gzipped JSONL, one object per line:
    {"id", "lang", "state", "order", "options", "gold", "d1", "d2", "src_sha"}
where `order` is the capability id per option slot and `options` are the card texts
in that same order - the order `laya.common.render_options` iterates, so
`options[gold]` is the gold card. `d1`/`d2` are the two frozen distractors
(score 0.62 / 0.55).

Runs in the repo venv (`.venv`), which carries sqlalchemy/httpx for the registry;
`.venv-laya` is inference-only and lacks them.

Usage:
    .venv/Scripts/python.exe scripts/laya_finetune/layachoice_prepare.py
    .venv/Scripts/python.exe scripts/laya_finetune/layachoice_prepare.py --smoke
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import importlib.util
import json
import random
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LOGS = ROOT / "logs"

# logs/ supplies _laya_choice_row and _cap_only_eval; packages/ makes the flat
# `core` package importable (core/ lives directly under packages/, not packages/core/core).
for p in (str(LOGS), str(ROOT / "packages")):
    if p not in sys.path:
        sys.path.insert(0, p)

import _laya_choice_row as choice_row  # noqa: E402
from _laya_choice_row import INSTRUCTIONS, classify_leak  # noqa: E402

DATA = HERE / "data"
RAW_DIR = DATA / "raw"
SMOKE_DIR = DATA / "smoke"

TRAIN_RENDERED = LOGS / "_laya_ds" / "LayaChoice_v1_train.jsonl"
TEST_RENDERED = LOGS / "_laya_ds" / "LayaChoice_v1_test_v3.jsonl"
VAL_RAW = ROOT / "data" / "LayaChoice_v1_validation_raw_v3.jsonl"
TEST_RAW = ROOT / "data" / "LayaChoice_v1_final_test_raw_v3.jsonl"
DESC_TSV = LOGS / "_laya_capdesc.tsv"

FORMAL_VIEW = "B_noprov"
CALIB_SEED = 20260922
CALIB_MAX = 400
DEFAULT_DISTRACTORS = ("cap-add-term", "cap-create-folder")
OPTION_SLOTS = 3
SMOKE_ROWS = 8

# Frozen shas of the sources this bundle is derived from. A mismatch means the
# frozen inputs moved under us and the bundle would silently stop describing them.
EXPECTED_SRC_SHA = {
    TRAIN_RENDERED: "bf4139262b9a398aa2821e55d5abf5a0a28384feeab6d8a85b88334557561fa3",
    TEST_RENDERED: "5a001c6a6b02a3c2fbe67b229d61e6c0b0ff023b7d542b9601697c2a48201f96",
    VAL_RAW: "974a3b99cf5883c527677373b53539f3ac859ddfb0ba59ea793058d03a6710ff",
    TEST_RAW: "e0738991d193cae39531a384305dc5d3547ffdd69f9e80b062c91079f51b4613",
}
EXPECTED_ROWS = {"train": 856, "val": 150, "test": 900}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_mod(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_desc_tsv(path: Path) -> dict[str, str]:
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            cid, _, desc = line.rstrip("\n").partition("\t")
            if cid.strip():
                out[cid.strip()] = desc
    return out


def check_sources() -> dict[str, str]:
    """Refuse to build from inputs that are not the frozen revision."""
    shas = {}
    for path, expected in EXPECTED_SRC_SHA.items():
        if not path.exists():
            raise FileNotFoundError(f"frozen source missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"frozen source changed: {path}\n  expected sha256 {expected}\n"
                f"  actual   sha256 {actual}\n"
                "Refusing to build a bundle that no longer describes the frozen dataset.")
        shas[path.name] = actual
    return shas


def bundle_from_rows(rows: list[dict], src_sha: str, split: str) -> tuple[list[dict], int]:
    """Rendered LayaChoice rows -> compact B_noprov bundle rows.

    Returns the bundle rows plus the number of rows whose `candidate_order` differs
    from the rendered option order (see the note below).
    """
    out = []
    rotated = 0
    for row in rows:
        cid = row["id"]
        gold = row["expected"]["tool"]
        crit = row["views"][FORMAL_VIEW]["tool"]
        order = list(crit.keys())
        if len(order) != OPTION_SLOTS:
            raise RuntimeError(f"{cid}: {FORMAL_VIEW} has {len(order)} options, expected {OPTION_SLOTS}")
        if gold not in crit:
            raise RuntimeError(f"{cid}: gold {gold} is not one of the {FORMAL_VIEW} options")
        # `laya.common.render_options` iterates `q["crit"].items()`, so the option slots
        # ARE this dict's insertion order. `build_views` fills every view from the same
        # sorted role_map, so B_noprov's key order must equal the row's own
        # `questions.tool.criteria` order - that is what a production renderer would emit.
        # `candidate_order` is NOT the option order: it is the corpus908 candidate list,
        # preserved as provenance by _laya_ds_build.py and never read by any renderer.
        # (It is rotated - gold balanced 287/283/286 - while the rendered criteria are
        # always sorted, gold 297/205/354.) Assert against the renderer's own source of
        # truth and merely count the inert divergence for the manifest.
        rendered = list(row["questions"]["tool"]["criteria"].keys())
        if rendered != order:
            raise RuntimeError(
                f"{cid}: questions.tool.criteria order {rendered} != {FORMAL_VIEW} key order {order}")
        if list(row.get("candidate_order") or rendered) != order:
            rotated += 1

        meta = row["candidate_meta"]
        negatives = sorted((m for m in meta if m["role"] != "gold"), key=lambda m: -m["score"])
        if len(negatives) != 2:
            raise RuntimeError(f"{cid}: expected 2 frozen distractors, got {len(negatives)}")
        scores = tuple(round(float(m["score"]), 2) for m in negatives)
        if scores != (0.62, 0.55):
            raise RuntimeError(f"{cid}: distractor scores {scores} != (0.62, 0.55)")
        d1 = order.index(negatives[0]["capability_id"])
        d2 = order.index(negatives[1]["capability_id"])
        if len({order.index(gold), d1, d2}) != OPTION_SLOTS:
            raise RuntimeError(f"{cid}: gold/d1/d2 do not occupy three distinct slots")

        options = [crit[c] for c in order]
        for slot, card in enumerate(options):
            if not str(card).strip():
                raise RuntimeError(f"{cid}: option slot {slot} ({order[slot]}) is empty")
            # LayaChoice-v1 is leak-audited: a frozen row must not hand the model the
            # query verbatim. `example`/`substring` mean the query reached the card text;
            # a `description` hit is the already-documented confound (the curated `does:`
            # text quoting the phrasing) and is counted, not fatal.
            leak = classify_leak(card, row["state"])
            if leak in ("example", "substring"):
                raise RuntimeError(f"{cid}: {FORMAL_VIEW} card for {order[slot]} leaks the query ({leak})")

        out.append({
            "id": cid,
            "split": split,
            "lang": row.get("language") or ("zh" if any("\u4e00" <= ch <= "\u9fff" for ch in row["state"]) else "en"),
            "state": row["state"],
            "order": order,
            "options": options,
            "gold": order.index(gold),
            "d1": d1,
            "d2": d2,
            "src_sha": src_sha,
        })
    return out, rotated


def render_validation(entries: dict, desc_tsv: dict[str, str], cap_only) -> list[dict]:
    """Render the frozen validation v3 RAW rows in memory, on the shared contract."""
    raw = read_jsonl(VAL_RAW)
    rows = []
    for i, o in enumerate(raw, 1):
        cap = str(o.get("capability_id") or "").strip()
        if cap not in entries:
            raise RuntimeError(f"validation raw line {i}: unknown capability {cap!r}")
        query = str(o.get("query") or "").strip()
        if not query:
            raise RuntimeError(f"validation raw line {i}: empty query")
        d0, d1 = cap_only.DISTRACTORS.get(cap, DEFAULT_DISTRACTORS)
        conf = str(o.get("confusable_with") or "").strip()
        if conf and conf not in (d0, d1):
            raise RuntimeError(
                f"validation raw line {i}: confusable_with {conf!r} not in {(d0, d1)}")
        role_map = choice_row.candidate_roles(cap, cap_only.DISTRACTORS.get(cap, DEFAULT_DISTRACTORS))
        positives, all_positives, negatives = {}, {}, {}
        for cid in role_map:
            e = entries[cid]
            corpus = [str(x).strip() for x in e.intent_corpus if str(x).strip()]
            positives[cid] = all_positives[cid] = corpus
            negatives[cid] = [f"- {str(x).strip()}" for x in (e.negatives or ()) if str(x).strip()]
        criteria = choice_row.build_views(entries, desc_tsv, role_map, positives, negatives,
                                         all_positives)
        rid = f"gen-val-{hashlib.sha256(query.encode()).hexdigest()[:12]}"
        rows.append(choice_row.build_row(rid, query, cap, role_map, criteria,
                                        origin="generated_validation", split="validation"))
    return rows


def calibration_split(n_train: int) -> tuple[list[int], list[int]]:
    """The fixed held-out calibration indices: min(CALIB_MAX, n//10) rows, seed 20260922.

    Mirrors the official recipe, but the chosen row indices are frozen here and
    recorded in the manifest, so the exclusion is reproducible and auditable.
    """
    order = list(range(n_train))
    random.Random(CALIB_SEED).shuffle(order)
    n_calib = min(CALIB_MAX, n_train // 10)
    calib = sorted(order[:n_calib])
    train = sorted(order[n_calib:])
    return calib, train


def write_gz_jsonl(path: Path, rows: list[dict]) -> None:
    """Gzipped JSONL with mtime=0 so the artifact hash is reproducible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    with open(path, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            gz.write(payload.encode("utf-8"))


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="emit only 8 rows per split (full structural validation still runs)")
    args = ap.parse_args()

    src_shas = check_sources()
    cap_only = load_mod("_cap_only_eval", LOGS / "_cap_only_eval.py")
    desc_tsv = load_desc_tsv(DESC_TSV)
    entries = await cap_only.load_entries()
    entries = {k: v for k, v in entries.items() if k != "cap-research"}
    print(f"registry entries: {len(entries)} capabilities")

    train_rows = read_jsonl(TRAIN_RENDERED)
    test_rows = read_jsonl(TEST_RENDERED)
    val_rows = render_validation(entries, desc_tsv, cap_only)
    print(f"rows: train {len(train_rows)} / val {len(val_rows)} / test {len(test_rows)}")

    for split, rows in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
        if len(rows) != EXPECTED_ROWS[split]:
            raise RuntimeError(f"{split}: {len(rows)} rows, expected {EXPECTED_ROWS[split]}")

    bundles, rotated = {}, {}
    for split, rows, sha_name in (
        ("train", train_rows, "LayaChoice_v1_train.jsonl"),
        ("val", val_rows, "LayaChoice_v1_validation_raw_v3.jsonl"),
        ("test", test_rows, "LayaChoice_v1_test_v3.jsonl"),
    ):
        bundles[split], rotated[split] = bundle_from_rows(rows, src_shas[sha_name], split)

    for split, rows in bundles.items():
        langs = {r["lang"] for r in rows}
        if not langs <= {"zh", "en"}:
            raise RuntimeError(f"{split}: unexpected languages {langs}")
        holes = [r["id"] for r in rows if len({r["gold"], r["d1"], r["d2"]}) != OPTION_SLOTS]
        if holes:
            raise RuntimeError(f"{split}: {len(holes)} rows with duplicate slots, e.g. {holes[:3]}")

    n_train = len(bundles["train"])
    calib_idx, train_idx = calibration_split(n_train)
    calib_ids = [bundles["train"][i]["id"] for i in calib_idx]

    out_dir = SMOKE_DIR if args.smoke else DATA
    limit = SMOKE_ROWS if args.smoke else None
    for split, rows in bundles.items():
        write_gz_jsonl(out_dir / f"layachoice_v1_{split}_b_noprov.jsonl.gz", rows[:limit] if limit else rows)

    if not args.smoke:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        for src in (VAL_RAW, TEST_RAW):
            dst = RAW_DIR / src.name
            shutil.copy2(src, dst)
            if sha256_file(dst) != sha256_file(src):
                raise RuntimeError(f"copy corrupted: {dst}")

    manifest = {
        "bench": "LayaChoice-v1",
        "formal_card_view": FORMAL_VIEW,
        "instructions": INSTRUCTIONS,
        "option_slots": OPTION_SLOTS,
        "option_order_rule": "insertion order of views.B_noprov.tool (= questions.tool.criteria "
                             "order), which laya.common.render_options iterates",
        "gold_slot_distribution": {s: {str(i): sum(1 for r in rows if r["gold"] == i)
                                       for i in range(OPTION_SLOTS)} for s, rows in bundles.items()},
        # Rows whose inert `candidate_order` (corpus908 provenance, never read by a
        # renderer) differs from the rendered option order. Recorded, not used.
        "candidate_order_diverges_from_rendered": rotated,
        "source_files": {p.name: src_shas[p.name] for p in EXPECTED_SRC_SHA},
        "rows": {"train": len(bundles["train"]), "val": len(bundles["val"]), "test": len(bundles["test"])},
        "language": {s: {l: sum(1 for r in rows if r["lang"] == l) for l in ("zh", "en")}
                     for s, rows in bundles.items()},
        "calibration": {
            "seed": CALIB_SEED,
            "source_train_rows": n_train,
            "calibration_rows": len(calib_idx),
            "effective_train_rows": len(train_idx),
            "indices": calib_idx,
            "ids": calib_ids,
        },
        "base_checkpoint": {
            "repo_id": "convaiinnovations/laya",
            "revision": "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
            "subfolder": "multilingual",
        },
        "smoke": bool(args.smoke),
    }
    if not args.smoke:
        (DATA / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        lines = ["# base_checkpoint convaiinnovations/laya@55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851/multilingual"]
        for f in sorted(list(DATA.glob("*.gz")) + list(RAW_DIR.glob("*.jsonl"))):
            lines.append(f"{sha256_file(f)}  {f.relative_to(DATA).as_posix()}")
        (DATA / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"written to {out_dir}")


if __name__ == "__main__":
    asyncio.run(main())
