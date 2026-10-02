#!/usr/bin/env python3
"""LayaChoice-v2 bundle verifier — data contract, splits, labels and frozen hashes.

Runs with the standard library only (no torch / no tokenizer / no model, no network),
so it can be run anywhere before spending GPU time:

    python layachoice_v2_verify.py

It checks, against the real files in ``data/`` and ``frozen_backup/``:

  * every bundle file's SHA256 against ``data/SHA256SUMS`` (computed, not assumed);
  * manifest identity (``bench``, ``option_slots``) and the frozen budget / base
    constants in ``layachoice_v2_spec``;
  * the full 4-way row schema and label legality for EVERY row (reusing the training
    adapter ``layachoice_v2_dataset.adapt_row``, so the verifier and the trainer
    cannot disagree about what a valid row is);
  * the per-split counts against the manifest (rows, language, part, gold slot);
  * that the three splits are disjoint by ``id`` and by ``source_query_id``;
  * that the ``REJECT`` card text is identical across all rows and equals the
    manifest's frozen card;
  * that the calibration split is reproducible and disjoint from the effective train
    split;
  * the ``frozen_backup`` pre-adjustment Test split against its recorded SHA256;
  * two static checks that the training stage never loads the Test split and that the
    Test split is never used to select a checkpoint.

Exits non-zero on the first class of failure found; prints a PASS/FAIL summary.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import layachoice_v2_spec as S        # noqa: E402
import layachoice_v2_dataset as D     # noqa: E402

DATA = HERE / "data"
BACKUP = HERE / "frozen_backup"
SPLITS = ("train", "val", "test")

_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    mark = "ok  " if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        _failures.append(name)
    return ok


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_sums(path: Path) -> dict[str, str]:
    sums = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, name = line.split(None, 1)
        sums[name.strip()] = digest
    return sums


# ── 1. frozen hashes ────────────────────────────────────────────────────────
def verify_hashes() -> None:
    print("[1] bundle SHA256")
    sums = read_sums(DATA / "SHA256SUMS")
    on_disk = sorted(p.name for p in DATA.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    check("SHA256SUMS lists exactly the bundle files", sorted(sums) == on_disk,
          f"sums={sorted(sums)} disk={on_disk}")
    for name in on_disk:
        actual = sha256(DATA / name)
        check(f"{name} sha256", sums.get(name) == actual,
              f"{actual}" + ("" if sums.get(name) == actual else f" != recorded {sums.get(name)}"))


# ── 2. constants / manifest identity ────────────────────────────────────────
def verify_identity(manifest: dict) -> None:
    print("[2] constants and manifest identity")
    check("OPTION_SLOTS == 4", S.OPTION_SLOTS == 4, str(S.OPTION_SLOTS))
    check("token budget 256/768/1024",
          (S.OPTION_MAX_TOKENS, S.HEAD_MAX_LEN, S.MAX_LEN) == (256, 768, 1024),
          f"{S.OPTION_MAX_TOKENS}/{S.HEAD_MAX_LEN}/{S.MAX_LEN}")
    check("REJECT label", S.REJECT_LABEL == "REJECT", S.REJECT_LABEL)
    check("manifest bench", manifest.get("bench") == S.V2_BENCH, str(manifest.get("bench")))
    check("manifest option_slots == 4", manifest.get("option_slots") == S.OPTION_SLOTS)
    check("manifest card view == B_noprov", manifest.get("formal_card_view") == "B_noprov")
    check("base revision matches spec",
          manifest.get("base_checkpoint", {}).get("revision") == S.BASE_REVISION,
          manifest.get("base_checkpoint", {}).get("revision", ""))
    check("base subfolder matches spec",
          manifest.get("base_checkpoint", {}).get("subfolder") == S.BASE_SUBFOLDER)


# ── 3. rows: schema, labels, distributions ──────────────────────────────────
def verify_rows(manifest: dict) -> dict[str, list[D.V2Example]]:
    print("[3] per-row schema and label legality (via the training adapter)")
    loaded: dict[str, list[D.V2Example]] = {}
    reject_card = manifest["reject"]["card"]

    for split in SPLITS:
        rows = D.load_split(split, DATA)          # adapt_row on every row; also checks manifest counts
        loaded[split] = rows
        check(f"{split}: row count == manifest {manifest['rows'][split]}",
              len(rows) == manifest["rows"][split], str(len(rows)))
        check(f"{split}: every row adapted without error (schema + labels)",
              all(isinstance(r, D.V2Example) for r in rows))
        check(f"{split}: exactly one REJECT card per row",
              all(sum(1 for c in r.order if c == S.REJECT_LABEL) == 1 for r in rows))
        check(f"{split}: REJECT card text == manifest frozen card",
              all(r.options[r.reject_index] == reject_card for r in rows))
        # gold slot distribution
        dist = {str(k): v for k, v in sorted(Counter(r.gold_index for r in rows).items())}
        want = {k: v for k, v in sorted(manifest["gold_slot_distribution"][split].items())}
        check(f"{split}: gold slot distribution == manifest", dist == want, str(dist))
        # language / part
        lang = dict(Counter(r.lang for r in rows))
        check(f"{split}: language counts == manifest", lang == manifest["language"][split], str(lang))
        part = dict(Counter(r.part for r in rows))
        want_part = {k.replace("part", "p"): v for k, v in manifest["part_rows"][split].items()}
        check(f"{split}: part counts == manifest", part == want_part, str(part))

    raw_keys = {
        split: {frozenset(json.loads(l)) for l in
                (DATA / f"v2_{split}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
        for split in SPLITS
    }
    check("every row carries exactly the frozen 14-key schema",
          all(ks == {D.ROW_KEYS} for ks in raw_keys.values()),
          str({s: sorted(next(iter(ks))) for s, ks in raw_keys.items()}))
    check("all splits share one field schema",
          len({next(iter(ks)) for ks in raw_keys.values()}) == 1)

    for attr, label in (("id", "id"), ("source_query_id", "source_query_id")):
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            inter = {getattr(r, attr) for r in loaded[a]} & {getattr(r, attr) for r in loaded[b]}
            check(f"{a} & {b} disjoint by {label}", not inter,
                  f"{len(inter)} shared" if inter else "")
    return loaded


# ── 4. calibration ──────────────────────────────────────────────────────────
def verify_calibration(manifest: dict, loaded: dict[str, list[D.V2Example]]) -> None:
    print("[4] calibration split")
    calib, fit = D.frozen_calibration(loaded["train"], manifest)   # raises if not reproducible
    check("calibration split reproducible from the recorded indices",
          len(calib) == manifest["calibration"]["calibration_rows"], str(len(calib)))
    check("effective train rows == manifest",
          len(fit) == manifest["calibration"]["effective_train_rows"], str(len(fit)))
    check("calibration disjoint from effective train",
          not ({r.id for r in calib} & {r.id for r in fit}))
    check("calibration rows carry the train split tag",
          all(r.split == "train" for r in calib))


# ── 5. frozen_backup ────────────────────────────────────────────────────────
def verify_backup() -> None:
    print("[5] frozen_backup (pre-adjustment Test split)")
    meta = json.loads((BACKUP / "FROZEN_SHA256.json").read_text(encoding="utf-8"))
    backup = BACKUP / "v2_test_1800_pre_adjustment.jsonl"
    recorded = meta["files"]["v2_test.jsonl"]
    actual = sha256(backup)
    check("backup sha256 == recorded pre-adjustment v2_test.jsonl",
          recorded == actual, f"{actual}" + ("" if recorded == actual else f" != {recorded}"))
    n = sum(1 for line in backup.read_text(encoding="utf-8").splitlines() if line.strip())
    check("backup row count == 1800", n == 1800, str(n))
    print("  note: backup is the PRE-adjustment Test and is NOT the current Test split "
          f"({sha256(DATA / 'v2_test.jsonl')[:8]}…); the current Test is 1400 rows.")


# ── 6. static checks: Test split never used for training or selection ───────
def verify_static() -> None:
    """The Test split must never train the model or select a checkpoint. The only
    permitted read of the Test split is the 5-row reload smoke inside the export
    stage; a read anywhere before ``mode_export`` / ``arm_benchmark`` is a failure."""
    print("[6] static checks (Test split never trains or selects)")
    fin = (HERE / "layachoice_v2_finetune.py").read_text(encoding="utf-8")
    ev = (HERE / "layachoice_v2_eval.py").read_text(encoding="utf-8")
    fin_export = fin.find("def mode_export")
    ev_select, ev_bench = ev.find("def arm_select"), ev.find("def arm_benchmark")
    fin_tests = [i for i in range(len(fin)) if fin.startswith('load_split("test"', i)]
    ev_tests = [i for i in range(len(ev)) if ev.startswith('load_split("test"', i)]
    check("finetune.py trains on the train split", 'load_split("train"' in fin)
    check("finetune.py never reads Test before the export stage",
          fin_export != -1 and fin_tests and all(i > fin_export for i in fin_tests),
          f"test reads at {fin_tests}, mode_export at {fin_export}")
    check("eval.py selects on val", 'load_split("val"' in ev)
    # Test may be read only by the baseline arm (before arm_select) and the benchmark
    # arm (after it) — never inside arm_select, which is what chooses the checkpoint.
    check("eval.py selection never reads Test",
          ev_select != -1 and ev_bench != -1
          and not [i for i in ev_tests if ev_select < i < ev_bench],
          f"test reads at {ev_tests}, select at {ev_select}, benchmark at {ev_bench}")


def main() -> int:
    print("LayaChoice-v2 bundle verification\n")
    if not (DATA / "manifest.json").exists():
        print(f"missing bundle: {DATA/'manifest.json'}", file=sys.stderr)
        return 2
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    verify_hashes()
    verify_identity(manifest)
    loaded = verify_rows(manifest)
    verify_calibration(manifest, loaded)
    verify_backup()
    verify_static()
    print()
    if _failures:
        print(f"FAILED: {len(_failures)} check(s): {', '.join(_failures)}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
