"""Historical one-off verifier for the pre-migration K=0 -> K=1 repair.

Reads the archived 14-key bundle under ``data_v4_build/full_r1`` and the audit
produced there, so it is kept as executed. The four-field repair tool is
``v4_k0_to_k1_inplace.py``; this script is not part of the active pipeline.
"""
import json
import random
import hashlib
from collections import Counter
from pathlib import Path

BASE = Path(r"e:/Work Space/Git Repositories/DeepDive/scripts/laya_finetune/data_v4_build")
SRC = BASE / "full_r1"
OUT = BASE / "full_r1_k0_to_k1_inplace_audit"
REF = BASE / "full_r1_repaired"
THRESHOLDS = (0.82, 0.85, 0.88)
TAGS = {0.82: "threshold_082", 0.85: "threshold_085", 0.88: "threshold_088"}
SPLITS = ("train", "test", "val")
REJECT = "REJECT"
ROW_KEYS = [
    "id", "source_query_id", "split", "part", "lang", "state", "order", "options",
    "gold_index", "target_kind", "gold_capability_id", "reject_index", "shuffle", "src_sha",
]

CAPSET = set(json.loads((SRC / "data" / "capability_ids.json").read_text(encoding="utf-8")))

# detail by threshold/split
detail = {}
with open(OUT / "k0_to_k1_repair_detail.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        detail.setdefault((r["threshold"], r["split"]), {})[r["id"]] = r

errors = []
summary = {}
for t in THRESHOLDS:
    ks = Counter()
    for split in SPLITS:
        cur = (SRC / "data" / TAGS[t] / f"v2_{split}.jsonl").read_bytes().split(b"\n")
        bak = (OUT / "backups" / TAGS[t] / f"v2_{split}.jsonl").read_bytes().split(b"\n")
        if len(cur) != len(bak):
            errors.append(f"{t}/{split}: line count {len(cur)} != {len(bak)}")
            continue
        changed = [i for i in range(len(cur)) if cur[i] != bak[i]]
        d = detail.get((t, split), {})
        if len(changed) != len(d):
            errors.append(f"{t}/{split}: changed {len(changed)} != detail {len(d)}")
        changed_ids = set()
        pos = Counter()
        for i in changed:
            row = json.loads(cur[i].decode("utf-8"))
            old = json.loads(bak[i].decode("utf-8"))
            rid = row["id"]
            changed_ids.add(rid)
            if rid not in d:
                errors.append(f"{t}/{split}: changed id not in detail {rid}")
                continue
            if list(row.keys()) != ROW_KEYS:
                errors.append(f"{t}/{split}/{rid}: key order")
            if len(row["order"]) != 2 or len(row["options"]) != 2:
                errors.append(f"{t}/{split}/{rid}: option count")
            if row["order"].count(REJECT) != 1:
                errors.append(f"{t}/{split}/{rid}: reject count")
            gold = row["gold_capability_id"]
            if gold not in CAPSET:
                errors.append(f"{t}/{split}/{rid}: invalid gold")
            if row["target_kind"] != "capability":
                errors.append(f"{t}/{split}/{rid}: target_kind")
            if row["order"][row["gold_index"]] != gold:
                errors.append(f"{t}/{split}/{rid}: gold index")
            if row["order"][row["reject_index"]] != REJECT:
                errors.append(f"{t}/{split}/{rid}: reject index")
            if set(row["order"]) != {gold, REJECT}:
                errors.append(f"{t}/{split}/{rid}: option set")
            for k in ("id", "source_query_id", "split", "part", "lang", "state", "src_sha"):
                if row[k] != old[k]:
                    errors.append(f"{t}/{split}/{rid}: changed {k}")
            for cid, text in zip(row["order"], row["options"]):
                if d[rid]["final_options"][row["order"].index(cid)] != text:
                    errors.append(f"{t}/{split}/{rid}: option text")
            prefix = f"k0-to-k1-inplace-v1:{t}:{split}"
            seed = f"{prefix}:{rid}"
            perm = [0, 1]
            random.Random(seed).shuffle(perm)
            base = [gold, REJECT]
            if row["shuffle"] != {"seed_prefix": prefix, "permutation": perm}:
                errors.append(f"{t}/{split}/{rid}: shuffle")
            if [base[x] for x in perm] != row["order"]:
                errors.append(f"{t}/{split}/{rid}: permutation order")
            if d[rid]["random_seed"] != seed:
                errors.append(f"{t}/{split}/{rid}: detail seed")
            pos[row["order"].index(REJECT)] += 1
        if set(d) != changed_ids:
            errors.append(f"{t}/{split}: detail id set mismatch")
        summary[f"{t}/{split}"] = {
            "changed": len(changed),
            "detail": len(d),
            "reject_positions": dict(sorted(pos.items())),
            "before_sha": hashlib.sha256((OUT / "backups" / TAGS[t] / f"v2_{split}.jsonl").read_bytes()).hexdigest()[:16],
            "after_sha": hashlib.sha256((SRC / "data" / TAGS[t] / f"v2_{split}.jsonl").read_bytes()).hexdigest()[:16],
        }
        for i in range(len(cur)):
            if cur[i] != bak[i]:
                continue
            # non-target rows must parse identically; bytes already proven equal
        # K distribution after repair
        for line in cur:
            if not line:
                continue
            r = json.loads(line.decode("utf-8"))
            ks[len(r["order"]) - 1] += 1
    summary[f"{t}/K"] = dict(sorted(ks.items()))

print(json.dumps({"errors": errors[:20], "error_count": len(errors),
                  "summary": summary}, ensure_ascii=False, indent=2))

# Deliverables and reference mtimes
print("deliverables:")
for name in ("k0_to_k1_repair_report.md", "k0_to_k1_repair_detail.jsonl",
             "k0_to_k1_diff_audit.json", "k0_to_k1_reject_position_distribution.csv",
             "k0_to_k1_pending_review.jsonl"):
    p = OUT / name
    print(f"  {name}: exists={p.exists()} size={p.stat().st_size if p.exists() else None}")

data_mtime = max((SRC / "data" / TAGS[t] / f"v2_{s}.jsonl").stat().st_mtime
                 for t in THRESHOLDS for s in SPLITS)
ref_mtime = max((REF / "data" / TAGS[t] / f"v2_{s}.jsonl").stat().st_mtime
                for t in THRESHOLDS for s in SPLITS)
print(f"latest full_r1 mtime={data_mtime:.0f}, latest full_r1_repaired mtime={ref_mtime:.0f}, "
      f"repaired_older={ref_mtime < data_mtime}")
