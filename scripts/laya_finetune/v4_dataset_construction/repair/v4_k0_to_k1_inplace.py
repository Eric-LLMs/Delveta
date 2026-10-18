"""In-place K=0 -> K=1 repair for the full_r1 dataset.

Only rows whose audit record is an action with zero candidates and a valid
source gold capability are touched.  Non-target lines are preserved byte for
byte.  full_r1_repaired is read-only and used only to cross-check targets.

Usage:
    python v4_k0_to_k1_inplace.py --dry-run
    python v4_k0_to_k1_inplace.py --apply
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(r"e:/Work Space/Git Repositories/DeepDive/scripts/laya_finetune/data_v4_build")
SRC = BASE / "full_r1"
REF = BASE / "full_r1_repaired"
OUT = BASE / "full_r1_k0_to_k1_inplace_audit"

THRESHOLDS = (0.82, 0.85, 0.88)
TAGS = {0.82: "threshold_082", 0.85: "threshold_085", 0.88: "threshold_088"}
SPLITS = ("train", "test", "val")
REJECT = "REJECT"
SEED_PREFIX_TMPL = "k0-to-k1-inplace-v1:{threshold}:{split}"
ROW_KEYS = [
    "id", "source_query_id", "split", "part", "lang", "state", "order", "options",
    "gold_index", "target_kind", "gold_capability_id", "reject_index", "shuffle", "src_sha",
]
UNCHANGED_KEYS = (
    "id", "source_query_id", "split", "part", "lang", "state", "src_sha",
)

CAPS = json.loads((SRC / "data" / "capability_ids.json").read_text(encoding="utf-8"))
CAPSET = set(CAPS)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def serialize(row: dict) -> bytes:
    return json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def split_raw(data: bytes):
    parts = data.split(b"\n")
    items = [(i + 1, raw) for i, raw in enumerate(parts) if raw != b""]
    return parts, items


def load_cards() -> tuple[dict[str, str], list[dict]]:
    cards: dict[str, str] = {}
    conflicts: list[dict] = []
    for t in THRESHOLDS:
        for split in SPLITS:
            p = SRC / "data" / TAGS[t] / f"v2_{split}.jsonl"
            for row in load_jsonl(p):
                if len(row["order"]) != len(row["options"]):
                    conflicts.append({"id": row["id"], "reason": "order_options_len"})
                    continue
                for cid, text in zip(row["order"], row["options"]):
                    if cid in cards and cards[cid] != text:
                        conflicts.append({"id": row["id"], "capability": cid,
                                          "reason": "card_text_conflict"})
                    else:
                        cards[cid] = text
    missing = [c for c in CAPS + [REJECT] if c not in cards]
    if missing:
        raise SystemExit(f"missing card text: {missing}")
    return cards, conflicts


def ref_check(row: dict, gold: str, old: dict, cards: dict) -> str | None:
    if len(row.get("order", [])) != 2 or len(row.get("options", [])) != 2:
        return "reference_option_count"
    if sorted(row["order"]) != sorted([gold, REJECT]):
        return "reference_candidate_set"
    if row.get("target_kind") != "capability":
        return "reference_target_kind"
    if row.get("gold_capability_id") != gold:
        return "reference_gold_id"
    if row["order"][row.get("gold_index", -1)] != gold:
        return "reference_gold_index"
    if row["order"][row.get("reject_index", -1)] != REJECT:
        return "reference_reject_index"
    for cid, text in zip(row["order"], row["options"]):
        if cards.get(cid) != text:
            return "reference_card_text"
    for key in UNCHANGED_KEYS:
        if row.get(key) != old.get(key):
            return f"reference_{key}"
    return None


def build_plan(apply: bool):
    cards, card_conflicts = load_cards()
    if card_conflicts:
        raise SystemExit(f"card conflicts: {card_conflicts[:10]}")

    # Per-threshold audit maps and reference target id sets.
    audits_by_t: dict[float, dict[str, dict]] = {}
    ref_ids_by_t: dict[float, set[str]] = {}
    ref_detail_counts: dict[tuple[float, str], int] = Counter()
    for t in THRESHOLDS:
        tag = TAGS[t]
        audits: dict[str, dict] = {}
        for a in load_jsonl(SRC / "data" / tag / "construction_audit.jsonl"):
            if a["id"] in audits:
                raise SystemExit(f"{tag}: duplicate audit id {a['id']}")
            audits[a["id"]] = a
        audits_by_t[t] = audits

        ref_ids: set[str] = set()
        for d in load_jsonl(REF / "data" / tag / "repair_detail.jsonl"):
            if d["threshold"] != t:
                raise SystemExit(f"{tag}: repair_detail threshold mismatch {d['threshold']}")
            ref_ids.add(d["id"])
            ref_detail_counts[(t, d["split"])] += 1
        ref_ids_by_t[t] = ref_ids

    plans: dict[tuple[float, str], dict] = {}
    pending: list[dict] = []
    our_ids_by_t: dict[float, set[str]] = {t: set() for t in THRESHOLDS}
    k_before: dict[float, Counter] = {t: Counter() for t in THRESHOLDS}
    k_after: dict[float, Counter] = {t: Counter() for t in THRESHOLDS}
    position_counts: dict[tuple[float, str], Counter] = defaultdict(Counter)
    all_seen: dict[float, set[str]] = {t: set() for t in THRESHOLDS}

    for t in THRESHOLDS:
        tag = TAGS[t]
        audits = audits_by_t[t]
        for split in SPLITS:
            path = SRC / "data" / tag / f"v2_{split}.jsonl"
            ref_path = REF / "data" / tag / f"v2_{split}.jsonl"
            data = path.read_bytes()
            parts, items = split_raw(data)

            ref_rows: dict[str, dict] = {}
            for r in load_jsonl(ref_path):
                if r["id"] in ref_rows:
                    raise SystemExit(f"{ref_path}: duplicate ref id {r['id']}")
                ref_rows[r["id"]] = r

            seen_ids: set[str] = set()
            targets: list[dict] = []
            for lineno, raw in items:
                row = json.loads(raw.decode("utf-8"))
                rid = row["id"]
                if rid in seen_ids:
                    raise SystemExit(f"{path}: duplicate row id {rid}")
                seen_ids.add(rid)
                if rid in all_seen[t]:
                    raise SystemExit(f"{tag}: id appears in multiple split files: {rid}")
                all_seen[t].add(rid)

                k = len(row["order"]) - 1
                k_before[t][k] += 1
                after_k = k
                a = audits.get(rid)

                if k == 0 and a is None:
                    pending.append({
                        "threshold": t, "split": split, "id": rid,
                        "reason": "audit_missing",
                        "detail": "no construction_audit record",
                    })
                elif k == 0 and a is not None:
                    gold = a.get("source_gold_capability_id")
                    cc = a.get("candidate_count")
                    if not a.get("is_action"):
                        pass  # K=0 no-action, out of scope
                    elif a.get("threshold") != t:
                        pending.append({
                            "threshold": t, "split": split, "id": rid,
                            "reason": "audit_threshold_conflict",
                            "detail": f"audit threshold={a.get('threshold')!r}",
                        })
                    elif a.get("split") != split:
                        pending.append({
                            "threshold": t, "split": split, "id": rid,
                            "reason": "audit_split_conflict",
                            "detail": f"audit split={a.get('split')!r}",
                        })
                    elif gold not in CAPSET:
                        pending.append({
                            "threshold": t, "split": split, "id": rid,
                            "reason": "invalid_source_gold",
                            "detail": f"source_gold_capability_id={gold!r}",
                        })
                    elif cc != 0:
                        pending.append({
                            "threshold": t, "split": split, "id": rid,
                            "reason": "audit_k_conflict",
                            "detail": f"audit candidate_count={cc!r} row K=0",
                        })
                    elif a.get("label_reason") != "no_candidates":
                        pending.append({
                            "threshold": t, "split": split, "id": rid,
                            "reason": "unexpected_label_reason",
                            "detail": f"label_reason={a.get('label_reason')!r}",
                        })
                    elif row.get("split") != split:
                        pending.append({
                            "threshold": t, "split": split, "id": rid,
                            "reason": "row_split_conflict",
                            "detail": f"row split={row.get('split')!r}",
                        })
                    elif (row["order"] != [REJECT]
                          or row["target_kind"] != "reject"
                          or row.get("gold_index") != 0
                          or row.get("reject_index") != 0):
                        pending.append({
                            "threshold": t, "split": split, "id": rid,
                            "reason": "row_shape_conflict",
                            "detail": (f"order={row['order']} "
                                       f"target_kind={row['target_kind']} "
                                       f"gold_index={row.get('gold_index')} "
                                       f"reject_index={row.get('reject_index')}"),
                        })
                    elif row.get("gold_capability_id") not in (None, gold):
                        pending.append({
                            "threshold": t, "split": split, "id": rid,
                            "reason": "row_gold_conflict",
                            "detail": f"row gold={row.get('gold_capability_id')!r} audit gold={gold!r}",
                        })
                    else:
                        ref = ref_rows.get(rid)
                        if ref is None:
                            pending.append({
                                "threshold": t, "split": split, "id": rid,
                                "reason": "reference_row_missing",
                                "detail": "id absent from full_r1_repaired",
                            })
                        else:
                            why = ref_check(ref, gold, row, cards)
                            if why is not None:
                                pending.append({
                                    "threshold": t, "split": split, "id": rid,
                                    "reason": why,
                                    "detail": "full_r1_repaired disagrees with audit",
                                })
                            else:
                                prefix = SEED_PREFIX_TMPL.format(threshold=t, split=split)
                                seed = f"{prefix}:{rid}"
                                perm = [0, 1]
                                random.Random(seed).shuffle(perm)
                                base_order = [gold, REJECT]
                                order = [base_order[i] for i in perm]
                                options = [cards[c] for c in order]
                                new_row = dict(row)
                                new_row["order"] = order
                                new_row["options"] = options
                                new_row["gold_index"] = order.index(gold)
                                new_row["target_kind"] = "capability"
                                new_row["gold_capability_id"] = gold
                                new_row["reject_index"] = order.index(REJECT)
                                new_row["shuffle"] = {
                                    "seed_prefix": prefix,
                                    "permutation": perm,
                                }
                                new_row = {key: new_row[key] for key in ROW_KEYS}
                                validate_new_row(new_row, old=row, gold=gold,
                                                 perm=perm, prefix=prefix, cards=cards)
                                new_line = serialize(new_row)
                                targets.append({
                                    "lineno": lineno,
                                    "raw": raw,
                                    "old": row,
                                    "new": new_row,
                                    "new_line": new_line,
                                    "gold": gold,
                                    "audit": a,
                                    "seed": seed,
                                    "seed_prefix": prefix,
                                    "permutation": perm,
                                    "reject_position": order.index(REJECT),
                                })
                                after_k = 1
                                our_ids_by_t[t].add(rid)
                                position_counts[(t, split)][order.index(REJECT)] += 1

                k_after[t][after_k] += 1

            # Target ids must be unique and present exactly once.
            target_ids = [x["old"]["id"] for x in targets]
            if len(target_ids) != len(set(target_ids)):
                raise SystemExit(f"{path}: duplicate target ids")

            plans[(t, split)] = {
                "threshold": t,
                "split": split,
                "path": path,
                "ref_path": ref_path,
                "data": data,
                "parts": parts,
                "items": items,
                "targets": targets,
                "before_sha": sha256_file(path),
            }

    # Reference / audit target sets must agree with the previous repair.
    for t in THRESHOLDS:
        ours = our_ids_by_t[t]
        refs = ref_ids_by_t[t]
        if ours != refs:
            missing = sorted(refs - ours)
            extra = sorted(ours - refs)
            raise SystemExit(
                f"target set mismatch at {t}: missing_from_new={len(missing)} "
                f"extra_in_new={len(extra)} samples={missing[:5] + extra[:5]}"
            )
        for split in SPLITS:
            n = sum(1 for p in plans.values()
                    if p["threshold"] == t and p["split"] == split
                    for _ in p["targets"])
            if n != ref_detail_counts[(t, split)]:
                raise SystemExit(
                    f"{t} {split}: target count {n} != reference {ref_detail_counts[(t, split)]}"
                )

    return {
        "cards": cards,
        "plans": plans,
        "pending": pending,
        "k_before": k_before,
        "k_after": k_after,
        "position_counts": position_counts,
        "our_ids_by_t": our_ids_by_t,
        "ref_ids_by_t": ref_ids_by_t,
    }


def validate_new_row(row: dict, old: dict, gold: str, perm: list[int],
                     prefix: str, cards: dict[str, str]) -> None:
    if list(row.keys()) != ROW_KEYS:
        raise AssertionError(f"schema keys: {list(row.keys())}")
    if len(row["order"]) != 2 or len(row["options"]) != 2:
        raise AssertionError("option count")
    if row["order"].count(REJECT) != 1 or row["order"].count(gold) != 1:
        raise AssertionError(f"order={row['order']}")
    if len(set(row["order"])) != 2:
        raise AssertionError("duplicate option")
    if row["target_kind"] != "capability":
        raise AssertionError("target_kind")
    if row["gold_capability_id"] != gold:
        raise AssertionError("gold_capability_id")
    if row["order"][row["gold_index"]] != gold:
        raise AssertionError("gold_index")
    if row["order"][row["reject_index"]] != REJECT:
        raise AssertionError("reject_index")
    if row["shuffle"] != {"seed_prefix": prefix, "permutation": perm}:
        raise AssertionError("shuffle")
    base = [gold, REJECT]
    if [base[i] for i in perm] != row["order"]:
        raise AssertionError("permutation/order")
    for cid, text in zip(row["order"], row["options"]):
        if cards.get(cid) != text:
            raise AssertionError(f"card text {cid}")
    for key in UNCHANGED_KEYS:
        if row[key] != old[key]:
            raise AssertionError(f"changed {key}")


def write_backup(plan: dict, out_dir: Path) -> dict:
    tag = TAGS[plan["threshold"]]
    bdir = out_dir / "backups" / tag
    bdir.mkdir(parents=True, exist_ok=True)
    bpath = bdir / f"v2_{plan['split']}.jsonl"
    shutil.copy2(plan["path"], bpath)
    if sha256_file(bpath) != plan["before_sha"]:
        raise SystemExit(f"backup hash mismatch: {bpath}")
    return {
        "backup_path": str(bpath),
        "backup_sha256": sha256_file(bpath),
    }


def apply_plan(plan: dict, out_dir: Path, cards: dict[str, str]) -> dict:
    path = plan["path"]
    target_by_line = {x["lineno"]: x for x in plan["targets"]}
    if len(target_by_line) != len(plan["targets"]):
        raise SystemExit(f"{path}: duplicate target line numbers")
    original_by_line = {ln: raw for ln, raw in plan["items"]}

    # Re-read and make sure nothing changed since preflight.
    data = path.read_bytes()
    if sha256_bytes(data) != plan["before_sha"]:
        raise SystemExit(f"{path}: file changed during preflight")
    parts, items = split_raw(data)
    if len(items) != len(plan["items"]):
        raise SystemExit(f"{path}: line count changed")
    for lineno, raw in items:
        if lineno in target_by_line:
            if sha256_bytes(raw) != sha256_bytes(target_by_line[lineno]["raw"]):
                raise SystemExit(f"{path}:{lineno}: source line changed")
        elif raw != original_by_line[lineno]:
            raise SystemExit(f"{path}:{lineno}: non-target line changed")

    new_parts = list(parts)
    for lineno, item in target_by_line.items():
        new_line = item["new_line"]
        if parts[lineno - 1].endswith(b"\r"):
            new_line += b"\r"
        new_parts[lineno - 1] = new_line
    new_data = b"\n".join(new_parts)

    backup = write_backup(plan, out_dir)
    try:
        path.write_bytes(new_data)
        return verify_written(plan, parts, target_by_line, backup, cards)
    except Exception:
        shutil.copy2(backup["backup_path"], path)
        if sha256_file(path) != plan["before_sha"]:
            raise SystemExit(f"{path}: restore failed")
        raise


def verify_written(plan: dict, parts: list[bytes], target_by_line: dict,
                   backup: dict, cards: dict[str, str]) -> dict:
    path = plan["path"]
    items = plan["items"]
    # Byte-level post-write verification.
    after = path.read_bytes()
    after_parts = after.split(b"\n")
    if len(after_parts) != len(parts):
        raise SystemExit(f"{path}: line count changed after write")
    for idx, raw in enumerate(parts):
        expected = target_by_line.get(idx + 1)
        if expected is not None:
            exp = expected["new_line"]
            if raw.endswith(b"\r"):
                exp += b"\r"
            if after_parts[idx] != exp:
                raise SystemExit(f"{path}:{idx + 1}: target line mismatch")
        elif after_parts[idx] != raw:
            raise SystemExit(f"{path}:{idx + 1}: non-target line changed")

    # Re-parse and validate every target row + file invariants.
    after_items = [(i + 1, raw) for i, raw in enumerate(after_parts) if raw != b""]
    if len(after_items) != len(items):
        raise SystemExit(f"{path}: row count changed")
    if [json.loads(raw)["id"] for _, raw in after_items] != \
       [json.loads(raw)["id"] for _, raw in items]:
        raise SystemExit(f"{path}: row order changed")

    for lineno, item in target_by_line.items():
        row = json.loads(after_parts[lineno - 1].decode("utf-8"))
        validate_new_row(row, old=item["old"], gold=item["gold"],
                         perm=item["permutation"], prefix=item["seed_prefix"],
                         cards=cards)
        if row["id"] != item["old"]["id"]:
            raise SystemExit(f"{path}:{lineno}: id changed")

    return {
        "threshold": plan["threshold"],
        "split": plan["split"],
        "file": str(path.relative_to(BASE)),
        "before_sha256": plan["before_sha"],
        "after_sha256": sha256_file(path),
        "backup_path": backup["backup_path"],
        "backup_sha256": backup["backup_sha256"],
        "total_lines": len(items),
        "target_rows": len(plan["targets"]),
        "modified_rows": len(plan["targets"]),
        "non_target_changed": 0,
        "line_order_changed": False,
        "total_rows_after": len(after_items),
        "threshold_unchanged": True,
        "split_ownership_unchanged": True,
        "schema_valid": True,
        "gold_index_valid": True,
        "status": "PASS",
    }


def write_outputs(result: dict, file_audits: list[dict]) -> None:
    # Detail records.
    detail_path = OUT / "k0_to_k1_repair_detail.jsonl"
    with open(detail_path, "w", encoding="utf-8", newline="\n") as f:
        for t in THRESHOLDS:
            for split in SPLITS:
                plan = result["plans"][(t, split)]
                for item in plan["targets"]:
                    old = item["old"]
                    audit = item["audit"]
                    rec = {
                        "id": old["id"],
                        "source_query_id": old["source_query_id"],
                        "threshold": t,
                        "split": split,
                        "file_path": str(plan["path"]),
                        "file_path_relative": str(plan["path"].relative_to(BASE)),
                        "original_line_number": item["lineno"],
                        "original_k": 0,
                        "repaired_k": 1,
                        "original_target_kind": old["target_kind"],
                        "original_label": REJECT,
                        "original_gold_capability_id": item["gold"],
                        "original_gold_score": audit.get("gold_score"),
                        "original_gold_retrieval_rank": audit.get("gold_retrieval_rank"),
                        "original_label_reason": audit.get("label_reason"),
                        "original_decision_state": audit.get("decision_state"),
                        "final_gold_capability_id": item["gold"],
                        "final_label": item["gold"],
                        "injected_capability_id": item["gold"],
                        "injected_card_text": result["cards"][item["gold"]],
                        "candidate_source": "source_gold_injected",
                        "retrieval_score_available": False,
                        "repair_reason": "k0_gold_supervision_injection",
                        "final_order": item["new"]["order"],
                        "final_options": item["new"]["options"],
                        "final_gold_index": item["new"]["gold_index"],
                        "final_reject_index": item["new"]["reject_index"],
                        "reject_position": item["reject_position"],
                        "random_seed": item["seed"],
                        "seed_prefix": item["seed_prefix"],
                        "permutation": item["permutation"],
                        "original_line": item["raw"].decode("utf-8"),
                        "repaired_line": item["new_line"].decode("utf-8"),
                        "original_line_sha256": sha256_bytes(item["raw"]),
                        "repaired_line_sha256": sha256_bytes(item["new_line"]),
                        "verification": "PASS",
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Pending review.
    with open(OUT / "k0_to_k1_pending_review.jsonl", "w",
              encoding="utf-8", newline="\n") as f:
        for rec in result["pending"]:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # REJECT position distribution.
    rows = []
    for t in THRESHOLDS:
        for split in SPLITS:
            c = result["position_counts"][(t, split)]
            total = c[0] + c[1]
            rows.append([
                t, split, c[0], f"{(100.0 * c[0] / total):.2f}" if total else "0.00",
                c[1], f"{(100.0 * c[1] / total):.2f}" if total else "0.00", total,
            ])
        c = Counter()
        for split in SPLITS:
            c += result["position_counts"][(t, split)]
        total = c[0] + c[1]
        rows.append([t, "ALL", c[0], f"{(100.0 * c[0] / total):.2f}" if total else "0.00",
                     c[1], f"{(100.0 * c[1] / total):.2f}" if total else "0.00", total])
    all_c = Counter()
    for k, c in result["position_counts"].items():
        all_c += c
    total = all_c[0] + all_c[1]
    rows.append(["ALL", "ALL", all_c[0],
                 f"{(100.0 * all_c[0] / total):.2f}" if total else "0.00",
                 all_c[1], f"{(100.0 * all_c[1] / total):.2f}" if total else "0.00",
                 total])
    with open(OUT / "k0_to_k1_reject_position_distribution.csv", "w",
              encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "split", "position_0_count", "position_0_pct",
                    "position_1_count", "position_1_pct", "total"])
        w.writerows(rows)

    # Diff audit.
    audit_obj = {
        "source_dir": str(SRC),
        "reference_dir": str(REF),
        "audit_dir": str(OUT),
        "random_seed_scheme": (
            "k0-to-k1-inplace-v1:<threshold>:<split>:<row id>; "
            "base order = [gold capability, REJECT]"
        ),
        "files": file_audits,
        "summary": {
            "files_modified": len(file_audits),
            "target_rows": sum(x["target_rows"] for x in file_audits),
            "modified_rows": sum(x["modified_rows"] for x in file_audits),
            "non_target_changed": sum(x["non_target_changed"] for x in file_audits),
            "line_order_changed": any(x["line_order_changed"] for x in file_audits),
            "pending_review": len(result["pending"]),
        },
        "status": "PASS" if not result["pending"] else "PASS_WITH_PENDING",
    }
    with open(OUT / "k0_to_k1_diff_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_obj, f, ensure_ascii=False, indent=2)
        f.write("\n")

    # Markdown report.
    lines: list[str] = []
    lines.append("# K=0 → K=1 原位修复报告\n")
    lines.append(f"- 数据目录：`{SRC}`")
    lines.append(f"- 只读对照：`{REF}`")
    lines.append(f"- 审计输出：`{OUT}`")
    lines.append("- 修复对象：原始 K=0、`is_action=True`、`candidate_count=0`、"
                 "`source_gold_capability_id` 属于 18 个有效 capability 的行。")
    lines.append("- 修复方式：补入唯一 Gold capability 卡 + 保留 REJECT，K=1，总选项 2。")
    lines.append("- 注入候选不是 ANN 召回，`retrieval_score_available=false`、"
                 "`candidate_source=source_gold_injected`，不伪造检索分数。")
    lines.append("- 原始 `construction_audit.jsonl` 未修改；修复溯源见本目录明细。\n")

    lines.append("## 1. 修改数量\n")
    lines.append("| threshold | split | 修改行数 |")
    lines.append("|---|---|---:|")
    total = 0
    for t in THRESHOLDS:
        for split in SPLITS:
            n = len(result["plans"][(t, split)]["targets"])
            total += n
            lines.append(f"| {t} | {split} | {n} |")
        n = sum(len(result["plans"][(t, s)]["targets"]) for s in SPLITS)
        lines.append(f"| {t} | ALL | {n} |")
    lines.append(f"| ALL | ALL | {total} |\n")

    lines.append("## 2. 修复前后 K 分布\n")
    lines.append("| threshold | K | 修复前 | 修复后 |")
    lines.append("|---|---:|---:|---:|")
    for t in THRESHOLDS:
        for k in (0, 1, 2, 3):
            lines.append(f"| {t} | {k} | {result['k_before'][t][k]} | {result['k_after'][t][k]} |")
    lines.append("")

    lines.append("## 3. REJECT 位置分布\n")
    lines.append("| threshold | split | 位置 0 | 位置 0 占比 | 位置 1 | 位置 1 占比 | 合计 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]}% | {row[4]} | {row[5]}% | {row[6]} |")
    lines.append("")
    lines.append("随机种子规则：`k0-to-k1-inplace-v1:<threshold>:<split>:<row id>`，"
                 "基础顺序 `[gold capability, REJECT]`，用 `random.Random(seed).shuffle([0,1])` "
                 "生成排列；每条的种子、排列和 REJECT 位置记录在明细 JSONL 中。\n")

    lines.append("## 4. 验证结果\n")
    lines.append("- 目标行：K=1、恰好 1 张 capability 卡 + 1 张 REJECT、总选项 2、"
                 "Gold 索引指向正确 capability：PASS")
    lines.append("- 非目标行逐字节一致、行顺序不变、文件行数不变：PASS（0 行变更）")
    lines.append("- 文件归属、split、query、样本 ID、`src_sha` 未改变：PASS")
    lines.append("- 与 `full_r1_repaired` 交叉核对：目标集合、Gold capability、卡片文本一致：PASS")
    lines.append(f"- 待确认样本：{len(result['pending'])}")
    lines.append("- 未启动模型训练或部署。\n")

    lines.append("## 5. 文件与备份\n")
    lines.append("| 文件 | 修改前 SHA-256 | 修改后 SHA-256 | 目标行 | 非目标行变更 |")
    lines.append("|---|---|---|---:|---:|")
    for fa in file_audits:
        lines.append(f"| `{fa['file']}` | `{fa['before_sha256'][:16]}…` | "
                     f"`{fa['after_sha256'][:16]}…` | {fa['target_rows']} | 0 |")
    lines.append("")
    lines.append("完整哈希见 `k0_to_k1_diff_audit.json`；原始文件备份位于 "
                 f"`{OUT / 'backups'}`。\n")

    if result["pending"]:
        lines.append("## 6. 待确认样本\n")
        lines.append("| threshold | split | id | 原因 |")
        lines.append("|---|---|---|---|")
        for rec in result["pending"]:
            lines.append(f"| {rec['threshold']} | {rec['split']} | `{rec['id']}` | "
                         f"{rec['reason']} |")
        lines.append("")
    else:
        lines.append("## 6. 待确认样本\n")
        lines.append("无。所有目标样本均通过 Gold、引用行和唯一定位校验。\n")

    lines.append("## 7. 结论\n")
    lines.append("PASS。目标行已原位从 K=0 改为 K=1，其他数据行未改动；"
                 "未启动训练或部署。")
    lines.append("")

    (OUT / "k0_to_k1_repair_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.apply == args.dry_run:
        raise SystemExit("choose exactly one of --dry-run / --apply")

    result = build_plan(apply=args.apply)

    if args.dry_run:
        print(json.dumps({
            "mode": "dry-run",
            "targets": {str(t): sum(len(p["targets"]) for p in result["plans"].values()
                                    if p["threshold"] == t) for t in THRESHOLDS},
            "pending": len(result["pending"]),
            "k_before": {str(t): dict(sorted(result["k_before"][t].items()))
                         for t in THRESHOLDS},
            "k_after": {str(t): dict(sorted(result["k_after"][t].items()))
                        for t in THRESHOLDS},
            "reject_positions": {
                f"{t}/{s}": dict(sorted(result["position_counts"][(t, s)].items()))
                for t in THRESHOLDS for s in SPLITS
            },
        }, ensure_ascii=False, indent=2))
        return

    if OUT.exists():
        raise SystemExit(f"audit output already exists: {OUT}")

    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "backups").mkdir()

    file_audits: list[dict] = []
    try:
        for t in THRESHOLDS:
            for split in SPLITS:
                plan = result["plans"][(t, split)]
                file_audits.append(apply_plan(plan, OUT, result["cards"]))
    except Exception:
        # Restore every file that has a backup, then re-raise.
        for fa in file_audits:
            p = BASE / fa["file"]
            shutil.copy2(fa["backup_path"], p)
        raise

    write_outputs(result, file_audits)
    print(json.dumps({
        "mode": "apply",
        "audit_dir": str(OUT),
        "files_modified": len(file_audits),
        "modified_rows": sum(x["modified_rows"] for x in file_audits),
        "pending": len(result["pending"]),
        "status": "PASS",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
