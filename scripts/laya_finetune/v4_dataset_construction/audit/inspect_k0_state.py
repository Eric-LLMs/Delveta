import json
import pathlib
import collections

BASE = pathlib.Path(r"e:/Work Space/Git Repositories/DeepDive/scripts/laya_finetune/data_v4_build")

for th in ("082", "085", "088"):
    print(f"== threshold {th} ==")
    for split in ("train", "test", "val"):
        for ver in ("full_r1", "full_r1_repaired"):
            p = BASE / ver / "data" / f"threshold_{th}" / f"v2_{split}.jsonl"
            if not p.exists():
                print(f"  {ver} {split}: MISSING {p}")
                continue
            ks = collections.Counter()
            cap0 = 0
            rej0 = 0
            n = 0
            for line in p.open(encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                n += 1
                k = len(r["order"]) - 1
                ks[k] += 1
                if k == 0:
                    if r.get("target_kind") == "capability":
                        cap0 += 1
                    else:
                        rej0 += 1
            print(f"  {ver} {split}: n={n} K={dict(sorted(ks.items()))} K0cap={cap0} K0rej={rej0}")

print()
pd = BASE / "full_r1_repaired" / "data" / "threshold_082" / "repair_detail.jsonl"
print("detail exists:", pd.exists(), pd)
if pd.exists():
    first = json.loads(pd.open(encoding="utf-8").readline())
    print("detail keys:", sorted(first.keys()))
    print("detail sample:", json.dumps(first, ensure_ascii=False)[:1200])
    rid = first.get("id") or first.get("sample_id") or first.get("row_id")
    print("rid:", rid)
    for ver in ("full_r1", "full_r1_repaired"):
        for split in ("train", "test", "val"):
            p = BASE / ver / "data" / "threshold_082" / f"v2_{split}.jsonl"
            if not p.exists():
                continue
            for ln, line in enumerate(p.open(encoding="utf-8"), 1):
                r = json.loads(line)
                if rid is not None and r.get("id") == rid:
                    print(f"{ver} {split} line {ln}: {json.dumps(r, ensure_ascii=False)[:800]}")
