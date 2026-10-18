import csv
import json
import hashlib
import shutil
from pathlib import Path
from collections import Counter

BASE = Path(r"e:/Work Space/Git Repositories/DeepDive/scripts/laya_finetune/data_v4_build")
MAIN = BASE / "full_r1"
REF = BASE / "full_r1_repaired"
ARCHIVE = BASE / "archive" / "full_r1_repaired_superseded"
AUD = BASE / "official_version_unification_audit"
TAGS = {0.82: "threshold_082", 0.85: "threshold_085", 0.88: "threshold_088"}
SPLITS = ("train", "test", "val")
REPAIRED = {
    0.82: {"ALL": 152, "train": 107, "test": 38, "val": 7},
    0.85: {"ALL": 603, "train": 421, "test": 149, "val": 33},
    0.88: {"ALL": 2466, "train": 1792, "test": 527, "val": 147},
}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


# ── 0. audit dirs ───────────────────────────────────────────────────────────
for sub in ("manifest_backup_pre_update", "pre_repair_aux_csv", "archived_listing"):
    (AUD / sub).mkdir(parents=True, exist_ok=True)

# ── 1. archive the superseded bundle ────────────────────────────────────────
archive_report = {"source": str(REF), "archive": str(ARCHIVE), "moved": False}
if REF.exists():
    if ARCHIVE.exists():
        raise SystemExit(f"archive target already exists: {ARCHIVE}")
    listing = {}
    for p in sorted(REF.rglob("*")):
        if p.is_file():
            listing[str(p.relative_to(REF))] = {
                "size": p.stat().st_size, "sha256": sha(p)}
    (AUD / "archived_listing" / "full_r1_repaired_files.json").write_text(
        json.dumps(listing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(REF), str(ARCHIVE))
    archive_report.update({"moved": True, "file_count": len(listing)})
    (ARCHIVE / "ARCHIVED.md").write_text(
        "# Archived: full_r1_repaired (superseded)\n\n"
        "This directory is NOT a training input.\n\n"
        "- Superseded by: `scripts/laya_finetune/data_v4_build/full_r1` (in-place K=0→K=1 repair).\n"
        "- Kept only as a read-only cross-check reference for the repaired target rows.\n"
        "- Differences: its repair used seed prefix `reject-shuffle-v1` and it rewrote "
        "`construction_audit.jsonl`; the official bundle uses "
        "`k0-to-k1-inplace-v1:<threshold>:<split>` and preserves the original audit.\n"
        "- Archive listing and hashes: `../official_version_unification_audit/archived_listing/`.\n",
        encoding="utf-8")
(AUD / "archive_report.json").write_text(
    json.dumps(archive_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

# ── 2. move stale pre-repair CSV exports out of the official bundle ─────────
csv_report = {}
for t, tag in TAGS.items():
    dst = AUD / "pre_repair_aux_csv" / tag
    dst.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        src = MAIN / "data" / tag / f"{split}.csv"
        rec = {"exists": src.exists()}
        if src.exists():
            rec.update({"sha256": sha(src), "size": src.stat().st_size})
            shutil.move(str(src), str(dst / f"{split}.csv"))
        csv_report[f"{tag}/{split}"] = rec
(AUD / "aux_csv_archive_report.json").write_text(
    json.dumps(csv_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

# ── 3. update manifests (backup first) ──────────────────────────────────────
manifest_report = {}
for t, tag in TAGS.items():
    d = MAIN / "data" / tag
    mp = d / "manifest.json"
    bak_dir = AUD / "manifest_backup_pre_update" / tag
    bak_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(mp, bak_dir / "manifest.json")
    before_sha = sha(bak_dir / "manifest.json")

    m = json.loads(mp.read_text(encoding="utf-8"))
    rows = {s: list(jsonl(d / f"v2_{s}.jsonl")) for s in SPLITS}
    gold_slot = {
        s: dict(sorted(Counter(str(r["gold_index"]) for r in rows[s]).items()))
        for s in SPLITS
    }
    source_files = dict(m["source_files"])
    for s in SPLITS:
        source_files[f"v2_{s}.jsonl"] = sha(d / f"v2_{s}.jsonl")
    source_files["construction_audit.jsonl"] = sha(d / "construction_audit.jsonl")

    repaired_by_split = {s: REPAIRED[t][s] for s in SPLITS}
    m["dataset_variant"] = "V4-dynamic-official-r1"
    m["option_order_rule"] = (
        "per-row deterministic shuffle; seed prefix reject-shuffle-v1 over the row id; "
        "in-place K=0->K=1 repaired rows use seed prefix "
        "k0-to-k1-inplace-v1:<threshold>:<split> over the row id; "
        "option count = capability candidates + 1"
    )
    m["gold_slot_distribution"] = gold_slot
    m["source_files"] = source_files
    m["repair"] = {
        "mode": "in_place",
        "rule": "K=0, is_action=true, candidate_count=0, and source_gold_capability_id is a valid capability",
        "action": "inject the source gold as the only capability candidate; keep REJECT; K=1; options=2",
        "candidate_origin": "source_gold_injected",
        "retrieval_score_available": False,
        "seed_prefix": f"k0-to-k1-inplace-v1:{t}:<split>",
        "base_order": ["<gold capability>", "REJECT"],
        "repaired_rows": REPAIRED[t]["ALL"],
        "repaired_by_split": repaired_by_split,
        "audit_dir": str(BASE / "full_r1_k0_to_k1_inplace_audit"),
        "construction_audit_note": (
            "construction_audit.jsonl preserves the pre-repair retrieval evidence and "
            "is intentionally not rewritten; row-level repaired state is recorded in "
            "the in-place repair audit detail."
        ),
    }
    mp.write_text(json.dumps(m, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_report[tag] = {
        "before_sha256": before_sha,
        "after_sha256": sha(mp),
        "rows": m["rows"],
        "repaired_rows": REPAIRED[t]["ALL"],
        "gold_slot_distribution": gold_slot,
        "source_files": source_files,
    }

# ── 4. official hashes ──────────────────────────────────────────────────────
hashes = {"data_root": str(MAIN / "data"), "bundles": {}}
for t, tag in TAGS.items():
    d = MAIN / "data" / tag
    hashes["bundles"][tag] = {
        "threshold": t,
        "files": {p.name: {"size": p.stat().st_size, "sha256": sha(p)}
                  for p in sorted(d.iterdir()) if p.is_file()},
    }
(AUD / "official_data_hashes.json").write_text(
    json.dumps(hashes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

print(json.dumps({"archive": archive_report,
                  "csv_archived": len(csv_report),
                  "manifests": {k: {"before": v["before_sha256"][:16],
                                    "after": v["after_sha256"][:16]}
                                for k, v in manifest_report.items()}},
                 ensure_ascii=False, indent=2))
