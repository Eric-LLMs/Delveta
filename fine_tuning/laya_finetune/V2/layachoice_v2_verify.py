#!/usr/bin/env python3
"""LayaChoice bundle verifier — data contract, splits, labels and frozen hashes.

Runs with the standard library only (no torch / no tokenizer / no model, no network),
so it can be run anywhere before spending GPU time:

    python layachoice_v2_verify.py

It accepts either installed layout under ``data/``:

  * flat — ``data/manifest.json`` plus the three splits at the root (the installed
    ``recall_threshold_082`` training bundle);
  * dynamic V4 — ``data/recall_threshold_{082,085,088}/`` bundles (plus the
    optional ``shared/`` data files) covered by one checksum file, either
    ``data/SHA256SUMS.all_thresholds`` or ``data/SHA256SUMS``.

Passing a directory as the only argument verifies that directory instead of the
default ``data/`` (``run.sh`` passes ``--data`` through this way). A single
``recall_threshold_0XX`` directory is verified on its own: its checksums are
read from the parent ``SHA256SUMS.all_thresholds`` and filtered to that
directory.

It checks, against the real files in ``data/``:

  * every bundle file's SHA256 against ``data/SHA256SUMS`` (computed, not assumed);
  * manifest identity (``bench``, ``option_slots``) and the frozen budget / base
    constants in ``layachoice_v2_spec``; an ``option_slots`` contract is accepted
    only as the frozen flat integer or the ``K+1`` dynamic declaration;
  * the full row schema and label legality for EVERY row (reusing the training
    adapter ``layachoice_v2_dataset.adapt_row``, so the verifier and the trainer
    cannot disagree about what a valid row is);
  * the per-split counts against the manifest (rows, language, gold slot) and
    each row's option count against the declared bounds;
  * that the three splits are disjoint by ``message_id`` and by rendered content
    (query + options + label) — the real leakage guard, since random ids cannot
    prove provenance;
  * that ``message_id`` is a canonical uuid4 and unique across the whole bundle;
  * for dynamic bundles, that the three thresholds carry the same ordered query
    list in each split;
  * that the ``REJECT`` card text is identical across all rows and equals the
    manifest's frozen card;
  * that the calibration split is reproducible and disjoint from the effective train
    split;
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

def _data_dir() -> Path:
    args = sys.argv[1:]
    if not args:
        return HERE / "data"
    if len(args) == 1:
        return Path(args[0])
    raise SystemExit("usage: layachoice_v2_verify.py [DATA_DIR]")


DATA = _data_dir()
SPLITS = ("train", "val", "test")
DYNAMIC_TAGS = ("recall_threshold_082", "recall_threshold_085", "recall_threshold_088")

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


def bundle_layout() -> tuple[str, dict[str, Path]]:
    """Installed layout: ``flat`` (one bundle at the data root) or ``dynamic``
    (the three V4 threshold bundles)."""
    if (DATA / "manifest.json").is_file():
        return "flat", {"v2": DATA}
    present = [t for t in DYNAMIC_TAGS if (DATA / t / "manifest.json").is_file()]
    if not present:
        return "missing", {}
    missing = [t for t in DYNAMIC_TAGS if t not in present]
    if missing:
        raise RuntimeError(f"incomplete dynamic bundle: missing {missing}")
    return "dynamic", {t: DATA / t for t in DYNAMIC_TAGS}


# ── 1. frozen hashes ────────────────────────────────────────────────────────
SUMS_NAMES = ("SHA256SUMS", "SHA256SUMS.all_thresholds")
# Repository control files that may sit beside a flat bundle on the Hub but are
# not part of the data. They are neither hashed nor required by the release.
REPO_CONTROL_FILES = (".gitattributes",)


def sums_source(layout: str) -> tuple[Path, str | None]:
    """The checksum file covering this layout, plus an optional path prefix.

    A dynamic bundle ships either ``SHA256SUMS.all_thresholds`` (the archived
    ``data_versions/`` copy) or ``SHA256SUMS`` (a staged release). A single
    threshold directory may be verified on its own: it has no local checksum
    file, so the parent ``SHA256SUMS.all_thresholds`` is used and its entries
    are filtered to this directory. Only these locations are candidates — a
    checksum file elsewhere is never picked up implicitly.
    """
    if layout == "dynamic":
        for name in ("SHA256SUMS.all_thresholds", "SHA256SUMS"):
            if (DATA / name).is_file():
                return DATA / name, None
        raise SystemExit(f"missing checksum file under {DATA}")
    if (DATA / "SHA256SUMS").is_file():
        return DATA / "SHA256SUMS", None
    parent_all = DATA.parent / "SHA256SUMS.all_thresholds"
    if DATA.name in DYNAMIC_TAGS and parent_all.is_file():
        return parent_all, DATA.name
    raise SystemExit(f"missing checksum file under {DATA}")


def bundle_files(layout: str) -> list[str]:
    """Exactly the files the checksum file must cover.

    ``flat``: every file at the data root except the checksum files and the
    repository control files. ``dynamic``: every file in the three threshold
    dirs plus the optional ``shared/`` dir. Other things a working/archive
    directory may contain (``message_id_map.jsonl``, ``build/``, ``smoke/``,
    audits, staging notes, the README) are NOT bundle files and are never
    hashed or required.
    """
    if layout == "flat":
        return sorted(p.name for p in DATA.iterdir()
                      if p.is_file() and p.name not in SUMS_NAMES
                      and p.name not in REPO_CONTROL_FILES)
    files = [p for tag in DYNAMIC_TAGS for p in (DATA / tag).iterdir()
             if p.is_file()]
    shared = DATA / "shared"
    if shared.is_dir():
        files += [p for p in shared.iterdir() if p.is_file()]
    return sorted(p.relative_to(DATA).as_posix() for p in files)


def verify_hashes(layout: str) -> None:
    print("[1] bundle SHA256")
    sums_file, prefix = sums_source(layout)
    sums = read_sums(sums_file)
    if prefix is not None:
        want = f"{prefix}/"
        sums = {k[len(want):]: v for k, v in sums.items() if k.startswith(want)}
    on_disk = bundle_files(layout)
    check("checksum file lists exactly the bundle files", sorted(sums) == on_disk,
          f"sums={sorted(sums)} disk={on_disk}")
    for name in on_disk:
        actual = sha256(DATA / name)
        check(f"{name} sha256", sums.get(name) == actual,
              f"{actual}" + ("" if sums.get(name) == actual else f" != recorded {sums.get(name)}"))


# ── 2. constants / manifest identity ────────────────────────────────────────
def verify_identity(manifest: dict, bundle: str) -> None:
    print(f"[2] constants and manifest identity — {bundle}")
    check("OPTION_SLOTS == 4", S.OPTION_SLOTS == 4, str(S.OPTION_SLOTS))
    check("token budget 256/768/1024",
          (S.OPTION_MAX_TOKENS, S.HEAD_MAX_LEN, S.MAX_LEN) == (256, 768, 1024),
          f"{S.OPTION_MAX_TOKENS}/{S.HEAD_MAX_LEN}/{S.MAX_LEN}")
    check("REJECT label", S.REJECT_LABEL == "REJECT", S.REJECT_LABEL)
    check("manifest bench", manifest.get("bench") == S.V2_BENCH, str(manifest.get("bench")))
    check("manifest training_schema_keys == 4-field schema",
          frozenset(manifest.get("training_schema_keys", [])) == D.ROW_KEYS,
          str(manifest.get("training_schema_keys")))
    check("manifest training fields == input.query + input.options + label",
          manifest.get("training_fields") == ["input.query", "input.options", "label"],
          str(manifest.get("training_fields")))
    check("manifest marks message_id/language as not used in training",
          sorted(manifest.get("not_used_in_training", [])) == ["language", "message_id"],
          str(manifest.get("not_used_in_training")))
    slots = manifest.get("option_slots")
    try:
        lo, hi = D.option_slot_bounds(manifest)
        slots_ok, slots_detail = True, f"{lo}..{hi}"
    except RuntimeError as exc:
        slots_ok, slots_detail = False, str(exc)
    check(f"{bundle}: manifest option_slots is the declared contract",
          slots_ok, slots_detail)
    if isinstance(slots, dict) and "observed" in slots:
        observed = slots["observed"]
        check(f"{bundle}: observed option counts within bounds",
              isinstance(observed, list) and all(lo <= n <= hi for n in observed),
              str(observed))
    check("manifest card view == B_noprov", manifest.get("formal_card_view") == "B_noprov")
    check("base revision matches spec",
          manifest.get("base_checkpoint", {}).get("revision") == S.BASE_REVISION,
          manifest.get("base_checkpoint", {}).get("revision", ""))
    check("base subfolder matches spec",
          manifest.get("base_checkpoint", {}).get("subfolder") == S.BASE_SUBFOLDER)


# ── 3. rows: schema, labels, distributions ──────────────────────────────────
def verify_rows(manifest: dict, data_dir: Path,
                bundle: str) -> dict[str, list[D.V2Example]]:
    print(f"[3] per-row schema and label legality — {bundle}")
    loaded: dict[str, list[D.V2Example]] = {}
    reject_card = manifest["reject"]["card"]
    lo, hi = D.option_slot_bounds(manifest)
    observed: set[int] = set()

    for split in SPLITS:
        # adapt_row on every row; also checks manifest counts and option bounds
        rows = D.load_split(split, data_dir)
        loaded[split] = rows
        check(f"{bundle}/{split}: row count == manifest {manifest['rows'][split]}",
              len(rows) == manifest["rows"][split], str(len(rows)))
        check(f"{bundle}/{split}: every row adapted without error (schema + labels)",
              all(isinstance(r, D.V2Example) for r in rows))
        check(f"{bundle}/{split}: every option count within {lo}..{hi}",
              all(lo <= len(r.options) <= hi for r in rows))
        observed.update(len(r.options) for r in rows)
        check(f"{bundle}/{split}: exactly one REJECT card per row",
              all(sum(1 for c in r.order if c == S.REJECT_LABEL) == 1 for r in rows))
        check(f"{bundle}/{split}: REJECT card text == manifest frozen card",
              all(r.options[r.reject_index] == reject_card for r in rows))
        # gold slot distribution
        dist = {str(k): v for k, v in sorted(Counter(r.gold_index for r in rows).items())}
        want = {k: v for k, v in sorted(manifest["gold_slot_distribution"][split].items())}
        check(f"{bundle}/{split}: gold slot distribution == manifest", dist == want, str(dist))
        # language
        lang = dict(Counter(r.language for r in rows))
        check(f"{bundle}/{split}: language counts == manifest",
              lang == manifest["language"][split], str(lang))

    declared = manifest.get("option_slots")
    if isinstance(declared, dict) and "observed" in declared:
        check(f"{bundle}: observed option counts == manifest",
              sorted(observed) == sorted(declared["observed"]),
              f"disk={sorted(observed)} manifest={sorted(declared['observed'])}")

    raw_keys = {
        split: {frozenset(json.loads(l)) for l in
                D.split_path(split, data_dir).read_text(encoding="utf-8").splitlines()
                if l.strip()}
        for split in SPLITS
    }
    check(f"{bundle}: every row carries exactly the frozen 4-key schema",
          all(ks == {D.ROW_KEYS} for ks in raw_keys.values()),
          str({s: sorted(next(iter(ks))) for s, ks in raw_keys.items()}))
    check(f"{bundle}: manifest training_schema_keys == disk schema",
          frozenset(manifest["training_schema_keys"]) == D.ROW_KEYS)
    check(f"{bundle}: all splits share one field schema",
          len({next(iter(ks)) for ks in raw_keys.values()}) == 1)

    ids = [r.message_id for rows in loaded.values() for r in rows]
    check(f"{bundle}: message_id unique across the bundle",
          len(set(ids)) == len(ids), f"{len(ids)} rows, {len(set(ids))} ids")

    def content(row: D.V2Example) -> tuple:
        return (row.state, row.rendered_options, row.gold_index)

    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        inter = {r.message_id for r in loaded[a]} & {r.message_id for r in loaded[b]}
        check(f"{bundle}/{a} & {b} disjoint by message_id", not inter,
              f"{len(inter)} shared" if inter else "")
        leaked = {content(r) for r in loaded[a]} & {content(r) for r in loaded[b]}
        check(f"{bundle}/{a} & {b} disjoint by query+options+label", not leaked,
              f"{len(leaked)} shared" if leaked else "")
    return loaded


def verify_threshold_consistency(loaded: dict[str, dict[str, list[D.V2Example]]]) -> None:
    """Threshold variants select the same source rows; only the candidates differ.

    ``message_id`` is random per bundle, so alignment is positional: the ordered
    query list of a split must be identical across thresholds. The label and the
    candidate options may legitimately differ, so only the query is compared.
    """
    print("[3b] dynamic thresholds share one source population")
    first_tag = next(iter(loaded))
    for split in SPLITS:
        base = [r.state for r in loaded[first_tag][split]]
        for tag in list(loaded)[1:]:
            other = [r.state for r in loaded[tag][split]]
            if base == other:
                check(f"{tag}/{split} ordered query list == {first_tag}/{split}", True)
                continue
            diff = sum(1 for x, y in zip(base, other) if x != y)
            check(f"{tag}/{split} ordered query list == {first_tag}/{split}", False,
                  f"{diff} differing rows, {len(base)} vs {len(other)} rows")


# ── 4. calibration ──────────────────────────────────────────────────────────
def verify_calibration(manifest: dict, loaded: dict[str, list[D.V2Example]],
                       bundle: str) -> None:
    print(f"[4] calibration split — {bundle}")
    calib, fit = D.frozen_calibration(loaded["train"], manifest)   # raises if not reproducible
    check(f"{bundle}: calibration split reproducible from the recorded indices",
          len(calib) == manifest["calibration"]["calibration_rows"], str(len(calib)))
    check(f"{bundle}: effective train rows == manifest",
          len(fit) == manifest["calibration"]["effective_train_rows"], str(len(fit)))
    check(f"{bundle}: calibration disjoint from effective train",
          not ({r.message_id for r in calib} & {r.message_id for r in fit}))
    check(f"{bundle}: calibration rows carry the train split tag",
          all(r.split == "train" for r in calib))


# ── 5. static checks: Test split never used for training or selection ───────
def verify_static() -> None:
    """The Test split must never train the model or select a checkpoint. The only
    permitted read of the Test split is the 5-row reload smoke inside the export
    stage; a read anywhere before ``mode_export`` / ``arm_benchmark`` is a failure."""
    print("[5] static checks (Test split never trains or selects)")
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
    print("LayaChoice bundle verification\n")
    layout, bundles = bundle_layout()
    if layout == "missing":
        print(f"missing bundle under {DATA}", file=sys.stderr)
        return 2
    print(f"layout: {layout} ({', '.join(bundles)})\n")
    verify_hashes(layout)
    all_loaded: dict[str, dict[str, list[D.V2Example]]] = {}
    for bundle, data_dir in bundles.items():
        manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
        verify_identity(manifest, bundle)
        loaded = verify_rows(manifest, data_dir, bundle)
        verify_calibration(manifest, loaded, bundle)
        all_loaded[bundle] = loaded
    if layout == "dynamic":
        verify_threshold_consistency(all_loaded)
    verify_static()
    print()
    if _failures:
        print(f"FAILED: {len(_failures)} check(s): {', '.join(_failures)}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
