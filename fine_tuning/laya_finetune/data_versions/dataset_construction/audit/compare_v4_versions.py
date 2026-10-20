import json
import hashlib
from collections import Counter
from pathlib import Path

BASE = Path(r"e:/Work Space/Git Repositories/DeepDive/scripts/laya_finetune/data_v4_build")
MAIN = BASE / "full_r1"
REF = BASE / "full_r1_repaired"
AUD = BASE / "full_r1_k0_to_k1_inplace_audit"
THRESHOLDS = (0.82, 0.85, 0.88)
TAGS = {0.82: "threshold_082", 0.85: "threshold_085", 0.88: "threshold_088"}
SPLITS = ("train", "test", "val")
REJECT = "REJECT"

# authoritative target ids from in-place audit detail
target = {}
with open(AUD / "k0_to_k1_repair_detail.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        target.setdefault((r["threshold"], r["split"]), set()).add(r["id"])

errors = []
report = {"row_alignment": {}, "non_target_bytes": {}, "target_rows": {},
          "audit_files": {}, "manifests": {}, "csv_staleness": {}}

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

for t in THRESHOLDS:
    tag = TAGS[t]
    for split in SPLITS:
        main_p = MAIN / "data" / tag / f"v2_{split}.jsonl"
        ref_p = REF / "data" / tag / f"v2_{split}.jsonl"
        main_lines = main_p.read_bytes().split(b"\n")
        ref_lines = ref_p.read_bytes().split(b"\n")
        key = f"{t}/{split}"
        report["row_alignment"][key] = {"main": len(main_lines), "ref": len(ref_lines),
                                        "bytes_equal": main_p.read_bytes() == ref_p.read_bytes(),
                                        "main_sha": sha(main_p), "ref_sha": sha(ref_p)}
        if len(main_lines) != len(ref_lines):
            errors.append(f"{key}: line count mismatch")
            continue
        ids_main, ids_ref = [], []
        non_target_diff = []
        target_diff = []
        for i, (ml, rl) in enumerate(zip(main_lines, ref_lines)):
            if not ml and not rl:
                continue
            m = json.loads(ml)
            r = json.loads(rl)
            ids_main.append(m["id"]); ids_ref.append(r["id"])
            if m["id"] != r["id"]:
                errors.append(f"{key}: id order mismatch at line {i+1}")
                break
            is_target = m["id"] in target.get((t, split), set())
            if not is_target:
                if ml != rl:
                    non_target_diff.append({"line": i + 1, "id": m["id"]})
            else:
                d = {"id": m["id"], "same_order": m["order"] == r["order"],
                     "same_gold": m["gold_capability_id"] == r["gold_capability_id"],
                     "same_option_set": set(m["order"]) == set(r["order"]),
                     "main_order": m["order"], "ref_order": r["order"],
                     "main_shuffle": m.get("shuffle"), "ref_shuffle": r.get("shuffle"),
                     "same_gold_text": None, "same_reject_text": None}
                gm = m["gold_capability_id"]
                if gm in m["order"] and gm in r["order"]:
                    d["same_gold_text"] = (m["options"][m["order"].index(gm)]
                                           == r["options"][r["order"].index(gm)])
                if REJECT in m["order"] and REJECT in r["order"]:
                    d["same_reject_text"] = (m["options"][m["order"].index(REJECT)]
                                             == r["options"][r["order"].index(REJECT)])
                target_diff.append(d)
        if ids_main != ids_ref:
            errors.append(f"{key}: id sequence differs")
        report["row_alignment"][key]["target_count"] = len(target.get((t, split), set()))
        report["non_target_bytes"][key] = {"diff_count": len(non_target_diff),
                                           "samples": non_target_diff[:5]}
        report["target_rows"][key] = {
            "count": len(target_diff),
            "same_order": sum(d["same_order"] for d in target_diff),
            "same_gold": sum(d["same_gold"] for d in target_diff),
            "same_option_set": sum(d["same_option_set"] for d in target_diff),
            "same_gold_text": sum(1 for d in target_diff if d["same_gold_text"]),
            "same_reject_text": sum(1 for d in target_diff if d["same_reject_text"]),
            "seed_prefix_main": sorted({str(d["main_shuffle"].get("seed_prefix")) for d in target_diff}),
            "seed_prefix_ref": sorted({str(d["ref_shuffle"].get("seed_prefix")) for d in target_diff}),
            "sample": target_diff[0] if target_diff else None,
        }
        if non_target_diff:
            errors.append(f"{key}: non-target rows differ in ref ({len(non_target_diff)})")

    # audit-ish files
    files = ["construction_audit.jsonl", "direct_reject.jsonl", "manifest.json"]
    for fn in files:
        mp = MAIN / "data" / tag / fn
        rp = REF / "data" / tag / fn
        rec = {"exists": [mp.exists(), rp.exists()]}
        if mp.exists() and rp.exists():
            rec.update({"main_size": mp.stat().st_size, "ref_size": rp.stat().st_size,
                        "main_sha": sha(mp), "ref_sha": sha(rp),
                        "identical": mp.read_bytes() == rp.read_bytes()})
        report["audit_files"][f"{t}/{fn}"] = rec

    # manifests detail
    mm = json.loads((MAIN / "data" / tag / "manifest.json").read_text(encoding="utf-8"))
    rm = json.loads((REF / "data" / tag / "manifest.json").read_text(encoding="utf-8"))
    report["manifests"][str(t)] = {
        "main_variant": mm.get("dataset_variant"),
        "ref_variant": rm.get("dataset_variant"),
        "main_repair": mm.get("repair"),
        "ref_repair": rm.get("repair"),
        "main_option_slots": mm.get("option_slots"),
        "ref_option_slots": rm.get("option_slots"),
        "main_option_order_rule": mm.get("option_order_rule"),
        "ref_option_order_rule": rm.get("option_order_rule"),
        "main_gold_slot": mm.get("gold_slot_distribution"),
        "ref_gold_slot": rm.get("gold_slot_distribution"),
        "rows_equal": mm.get("rows") == rm.get("rows"),
        "calibration_equal": mm.get("calibration") == rm.get("calibration"),
    }
    # csv staleness: CSV must reflect pre-repair state
    for split in SPLITS:
        csv_p = MAIN / "data" / tag / f"{split}.csv"
        bak_p = AUD / "backups" / tag / f"v2_{split}.jsonl"
        stale_hits = 0
        checked = 0
        with open(csv_p, encoding="utf-8") as f:
            header = f.readline()
        import csv as _csv
        rows = list(_csv.DictReader(open(csv_p, encoding="utf-8")))
        bak_rows = [json.loads(l) for l in bak_p.read_bytes().split(b"\n") if l]
        by_id = {r["id"]: r for r in bak_rows}
        for row in rows:
            b = by_id.get(row.get("id"))
            if not b or b["id"] not in target.get((t, split), set()):
                continue
            checked += 1
            order_col = row.get("order", "")
            # CSV encodes order/options with separators; just check reject position label
            if b["order"] == [REJECT] and REJECT in order_col and b["gold_capability_id"] is None:
                stale_hits += 1
        report["csv_staleness"][f"{t}/{split}"] = {"target_rows_in_csv": checked,
                                                   "matches_pre_repair": stale_hits}

print(json.dumps({"error_count": len(errors), "errors": errors[:20],
                  "report": report}, ensure_ascii=False, indent=2, default=str))
