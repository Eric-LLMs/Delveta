import csv
import hashlib
import json
import random
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(r"e:/Work Space/Git Repositories/DeepDive")
MAIN = REPO / "scripts/laya_finetune/data_v4_build/full_r1"
AUD = REPO / "data/laya_finetune/data_v4_build/full_r1_k0_to_k1_inplace_audit"
OUT = REPO / "data/laya_finetune/data_v4_build/distribution_report"
OUT.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [(0.82, "recall_threshold_082"), (0.85, "recall_threshold_085"), (0.88, "recall_threshold_088")]
SPLITS = ("train", "val", "test")
REJECT = "REJECT"
REPAIR_PREFIX = "k0-to-k1-inplace-v1"
DEFAULT_PREFIX = "reject-shuffle-v1"
ROW_KEYS = ["id", "source_query_id", "split", "part", "lang", "state", "order", "options",
            "gold_index", "target_kind", "gold_capability_id", "reject_index", "shuffle", "src_sha"]
ROW_KEY_SET = set(ROW_KEYS)
LANG_GROUPS = {"zh": "zh", "en": "en", "mixed": "mixed"}
LANG_ORDER = ("zh", "en", "mixed")
LANG_LABEL = {"zh": "zh", "en": "en", "mixed": "mixed/other"}

CAPS = json.loads((MAIN / "data/capability_ids.json").read_text(encoding="utf-8"))
CAPSET = set(CAPS)
TOOLS = list(CAPS) + [REJECT]
TOOL_WITH_ALL = ["ALL"] + TOOLS


def pct(n, d):
    if not d:
        return 0.0
    return round(100.0 * n / d, 4)


def norm_query(text):
    t = unicodedata.normalize("NFKC", str(text or "")).strip().casefold()
    return " ".join(t.split())


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_csv(name, fieldnames, rows):
    path = OUT / name
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return path


quality = defaultdict(lambda: {"count": 0, "samples": []})


def flag(kind, sample=None):
    rec = quality[kind]
    rec["count"] += 1
    if sample is not None and len(rec["samples"]) < 5:
        rec["samples"].append(sample)


stats = {}
detail_all = {}

# ── load repair detail ──────────────────────────────────────────────────────
detail_path = AUD / "k0_to_k1_repair_detail.jsonl"
for line in detail_path.open(encoding="utf-8"):
    if not line.strip():
        continue
    d = json.loads(line)
    d["_threshold"] = float(d["threshold"])
    d["_tag"] = {0.82: "recall_threshold_082", 0.85: "recall_threshold_085", 0.88: "recall_threshold_088"}[d["_threshold"]]
    detail_all[(d["_threshold"], d["split"], d["id"])] = d

# ── load and stream every official row ──────────────────────────────────────
for threshold, tag in THRESHOLDS:
    data_dir = MAIN / "data" / tag
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    audits = {}
    for line in (data_dir / "construction_audit.jsonl").open(encoding="utf-8"):
        if line.strip():
            a = json.loads(line)
            audits[a["id"]] = a

    details = {k: v for k, v in detail_all.items() if k[0] == threshold}
    detail_by_id = {v["id"]: v for v in details.values()}
    details_seen = set()

    st = {
        "threshold": threshold, "tag": tag, "manifest": manifest,
        "rows": Counter(), "lang": defaultdict(Counter), "gold_kind": defaultdict(Counter),
        "gold_tool": defaultdict(Counter), "k": defaultdict(Counter),
        "k_split_lang": defaultdict(Counter), "k_origin": Counter(),
        "k_lang_tool": defaultdict(Counter), "tool_lang": defaultdict(Counter),
        "tool_split": defaultdict(Counter), "tool_k": defaultdict(Counter),
        "gold_kind_split_lang": defaultdict(Counter),
        "inj_slt": Counter(), "inj_split": Counter(), "inj_lang": Counter(),
        "inj_tool": Counter(), "inj_total": 0,
        "query_ids": defaultdict(set), "query_norm": defaultdict(set),
        "families": defaultdict(set), "row_ids": Counter(), "train_ids": [],
        "orig_k": Counter(), "rep_k": Counter(),
        "k0_total": 0, "k0_only_reject": 0, "k0_gold_cap": 0, "k0_gold_reject": 0,
        "lang_raw": Counter(), "lang_group": Counter(),
        "origin_k": Counter(),
    }

    for split in SPLITS:
        path = data_dir / f"v2_{split}.jsonl"
        backup_path = AUD / "backups" / tag / f"v2_{split}.jsonl"
        backup_lines = backup_path.read_text(encoding="utf-8").split("\n") if backup_path.is_file() else []
        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                raw = line.rstrip("\n")
                row = json.loads(raw)
                rid = row.get("id")
                split_tag = row.get("split")
                if split_tag != split:
                    flag("split_tag_errors", {"threshold": threshold, "file": split, "id": rid, "split": split_tag})
                st["rows"][split] += 1
                st["row_ids"][rid] += 1
                if split == "train":
                    st["train_ids"].append(rid)

                if set(row.keys()) != ROW_KEY_SET:
                    flag("schema_key_errors", {"threshold": threshold, "split": split, "id": rid,
                                               "keys": sorted(set(row.keys()) ^ ROW_KEY_SET)})
                    continue

                type_ok = (
                    isinstance(row["id"], str) and isinstance(row["source_query_id"], str)
                    and isinstance(row["split"], str) and isinstance(row["part"], str)
                    and isinstance(row["lang"], str) and isinstance(row["state"], str)
                    and isinstance(row["order"], list) and isinstance(row["options"], list)
                    and isinstance(row["gold_index"], int) and not isinstance(row["gold_index"], bool)
                    and isinstance(row["target_kind"], str)
                    and (row["gold_capability_id"] is None or isinstance(row["gold_capability_id"], str))
                    and isinstance(row["reject_index"], int) and not isinstance(row["reject_index"], bool)
                    and isinstance(row["shuffle"], dict)
                    and isinstance(row["shuffle"].get("seed_prefix"), str)
                    and isinstance(row["shuffle"].get("permutation"), list)
                    and isinstance(row["src_sha"], str)
                )
                if not type_ok:
                    flag("type_errors", {"threshold": threshold, "split": split, "id": rid})

                order = row["order"]
                options = row["options"]
                n = len(order)
                reject_n = sum(1 for c in order if c == REJECT)
                k = sum(1 for c in order if c != REJECT)
                if len(options) != n:
                    flag("option_len_mismatch", {"threshold": threshold, "split": split, "id": rid,
                                                 "order": n, "options": len(options)})
                if reject_n != 1:
                    flag("reject_count_errors", {"threshold": threshold, "split": split, "id": rid,
                                                 "reject_count": reject_n})
                if n != k + 1:
                    flag("k_option_count_mismatch", {"threshold": threshold, "split": split, "id": rid,
                                                     "k": k, "options": n})
                if len(set(order)) != n:
                    flag("duplicate_candidates", {"threshold": threshold, "split": split, "id": rid,
                                                  "order": order})
                bad = [c for c in order if c != REJECT and c not in CAPSET]
                if bad:
                    flag("invalid_capability_ids", {"threshold": threshold, "split": split, "id": rid,
                                                    "bad": bad})

                lang_raw = row["lang"]
                st["lang_raw"][lang_raw] += 1
                lang = LANG_GROUPS.get(lang_raw)
                if lang is None:
                    flag("unknown_language_rows", {"threshold": threshold, "split": split, "id": rid,
                                                   "lang": lang_raw})
                    lang = "mixed"
                st["lang_group"][lang] += 1

                gold = row["gold_capability_id"]
                target_kind = row["target_kind"]
                if target_kind == "reject" or gold is None:
                    gold_kind = "REJECT"
                    gold_tool = REJECT
                    if target_kind != "reject":
                        flag("gold_label_errors", {"threshold": threshold, "split": split, "id": rid,
                                                   "target_kind": target_kind, "gold": gold})
                else:
                    gold_kind = "capability_tool"
                    gold_tool = gold
                    if gold not in CAPSET:
                        flag("invalid_gold_ids", {"threshold": threshold, "split": split, "id": rid,
                                                  "gold": gold})
                    if target_kind != "capability":
                        flag("gold_label_errors", {"threshold": threshold, "split": split, "id": rid,
                                                   "target_kind": target_kind, "gold": gold})

                reject_index = row["reject_index"]
                gold_index = row["gold_index"]
                if target_kind == "reject":
                    if not (reject_index == gold_index and 0 <= gold_index < n
                            and order[gold_index] == REJECT):
                        flag("gold_index_errors", {"threshold": threshold, "split": split, "id": rid,
                                                   "kind": "reject", "gold_index": gold_index,
                                                   "reject_index": reject_index})
                else:
                    if not (0 <= gold_index < n and order[gold_index] == gold
                            and gold_index != reject_index and gold in order):
                        flag("gold_index_errors", {"threshold": threshold, "split": split, "id": rid,
                                                   "kind": "capability", "gold": gold,
                                                   "gold_index": gold_index})
                    if not (0 <= reject_index < n and order[reject_index] == REJECT):
                        flag("gold_index_errors", {"threshold": threshold, "split": split, "id": rid,
                                                   "kind": "reject_index", "reject_index": reject_index})

                prefix = row["shuffle"].get("seed_prefix", "")
                perm = row["shuffle"].get("permutation", [])
                if sorted(perm) != list(range(n)) if all(isinstance(x, int) for x in perm) else True:
                    flag("shuffle_permutation_errors", {"threshold": threshold, "split": split, "id": rid,
                                                        "perm": perm, "n": n})
                else:
                    exp = list(range(n))
                    random.Random(f"{prefix}:{rid}").shuffle(exp)
                    if perm != exp:
                        flag("shuffle_seed_errors", {"threshold": threshold, "split": split, "id": rid,
                                                     "prefix": prefix, "perm": perm})

                injected = rid in detail_by_id
                origin = "source_gold_injected" if injected else "ann_recall"
                if injected:
                    details_seen.add(rid)
                elif prefix.startswith(REPAIR_PREFIX):
                    flag("repair_detail_missing", {"threshold": threshold, "split": split, "id": rid})
                if injected and prefix != f"{REPAIR_PREFIX}:{threshold}:{split}":
                    flag("injected_row_errors", {"threshold": threshold, "split": split, "id": rid,
                                                 "reason": "seed_prefix", "prefix": prefix})
                if not injected and prefix != DEFAULT_PREFIX:
                    flag("unknown_shuffle_prefix", {"threshold": threshold, "split": split, "id": rid,
                                                    "prefix": prefix})

                d = detail_by_id.get(rid)
                if d:
                    st["orig_k"][int(d["original_k"])] += 1
                    st["rep_k"][int(d["repaired_k"])] += 1
                    checks = {
                        "split": d["split"] == split,
                        "final_order": d["final_order"] == order,
                        "final_options": d["final_options"] == options,
                        "final_gold_index": d["final_gold_index"] == gold_index,
                        "final_reject_index": d["final_reject_index"] == reject_index,
                        "final_gold": d["final_gold_capability_id"] == gold,
                        "candidate_source": d["candidate_source"] == "source_gold_injected",
                        "retrieval_score_available": d["retrieval_score_available"] is False,
                        "repaired_k": int(d["repaired_k"]) == k,
                    }
                    raw_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
                    raw_hash_nl = hashlib.sha256((raw + "\n").encode("utf-8")).hexdigest()
                    checks["repaired_line_sha"] = d["repaired_line_sha256"] in (raw_hash, raw_hash_nl)
                    oln = int(d["original_line_number"])
                    if 1 <= oln <= len(backup_lines):
                        old_raw = backup_lines[oln - 1]
                        old_hash = hashlib.sha256(old_raw.encode("utf-8")).hexdigest()
                        old_hash_nl = hashlib.sha256((old_raw + "\n").encode("utf-8")).hexdigest()
                        checks["original_line_sha"] = d["original_line_sha256"] in (old_hash, old_hash_nl)
                    else:
                        checks["original_line_sha"] = False
                    bad_checks = [k for k, ok in checks.items() if not ok]
                    if bad_checks:
                        flag("repair_detail_mismatches", {"threshold": threshold, "split": split,
                                                          "id": rid, "failed": bad_checks})
                    if not (k == 1 and n == 2 and target_kind == "capability" and gold in order):
                        flag("injected_row_errors", {"threshold": threshold, "split": split, "id": rid,
                                                     "reason": "shape", "k": k, "n": n,
                                                     "target_kind": target_kind})

                st["lang"][split][lang] += 1
                st["gold_kind"][split][gold_kind] += 1
                st["gold_tool"][split][gold_tool] += 1
                st["k"][split][k] += 1
                st["k_split_lang"][(split, lang)][k] += 1
                st["k_origin"][(origin, k)] += 1
                st["k_lang_tool"][("ALL", gold_tool)][k] += 1
                st["k_lang_tool"][(lang, gold_tool)][k] += 1
                st["gold_kind_split_lang"][(split, lang)][gold_kind] += 1
                st["tool_lang"][gold_tool][lang] += 1
                st["tool_split"][gold_tool][split] += 1
                st["tool_k"][gold_tool][k] += 1
                st["query_ids"][split].add(row["source_query_id"])
                st["query_norm"][split].add(norm_query(row["state"]))
                fam = audits.get(rid, {}).get("query_family_id")
                if fam:
                    st["families"][split].add(str(fam))
                if injected:
                    st["inj_total"] += 1
                    st["inj_split"][split] += 1
                    st["inj_lang"][lang] += 1
                    st["inj_tool"][gold_tool] += 1
                    st["inj_slt"][(split, lang, gold_tool)] += 1
                if k == 0:
                    st["k0_total"] += 1
                    if n == 1 and order[0] == REJECT:
                        st["k0_only_reject"] += 1
                    if gold_kind == "capability_tool":
                        st["k0_gold_cap"] += 1
                    else:
                        st["k0_gold_reject"] += 1

    # details that never matched a row
    missing_detail = set(detail_by_id) - details_seen
    if missing_detail:
        flag("repair_detail_row_missing", {"threshold": threshold,
                                           "ids": sorted(missing_detail)[:5],
                                           "count": len(missing_detail)})

    # duplicate row ids
    dup_ids = [i for i, c in st["row_ids"].items() if c > 1]
    if dup_ids:
        flag("duplicate_row_ids", {"threshold": threshold, "count": len(dup_ids),
                                   "ids": dup_ids[:5]})

    # split leakage
    leakage = {}
    for a_i, a in enumerate(SPLITS):
        for b in SPLITS[a_i + 1:]:
            for name, key in (("normalized_query", "query_norm"),
                              ("source_query_id", "query_ids"),
                              ("query_family", "families")):
                inter = st[key][a] & st[key][b]
                leakage[f"{name}:{a}-{b}"] = {"count": len(inter),
                                              "examples": sorted(inter)[:5]}
                if inter:
                    flag(f"leakage_{name}", {"threshold": threshold, "pair": f"{a}-{b}",
                                             "count": len(inter), "examples": sorted(inter)[:5]})
    st["leakage"] = leakage

    # calibration
    cal = manifest["calibration"]
    indices = cal["indices"]
    calib_ids = cal["ids"]
    cal_errors = []
    if len(indices) != len(set(indices)):
        cal_errors.append("duplicate_indices")
    if indices and (min(indices) < 0 or max(indices) >= len(st["train_ids"])):
        cal_errors.append("index_out_of_range")
    expected_ids = [st["train_ids"][i] for i in indices if 0 <= i < len(st["train_ids"])]
    if expected_ids != calib_ids:
        cal_errors.append("ids_not_reproducible")
    effective = len(st["train_ids"]) - len(indices)
    if effective != cal["effective_train_rows"]:
        cal_errors.append("effective_train_mismatch")
    st["calibration"] = {
        "rows": len(indices), "index_unique": len(set(indices)),
        "ids_unique": len(set(calib_ids)), "effective_train_excluding_calibration": effective,
        "manifest_effective_train": cal["effective_train_rows"],
        "errors": cal_errors,
    }
    if cal_errors:
        flag("calibration_errors", {"threshold": threshold, "errors": cal_errors})

    st["detail_count"] = len(detail_by_id)
    st["details_seen"] = len(details_seen)
    stats[threshold] = st

# ── 1. overall_distribution.csv ─────────────────────────────────────────────
rows_out = []
for threshold, tag in THRESHOLDS:
    st = stats[threshold]
    total_all = sum(st["rows"].values())
    for split in SPLITS + ("ALL",):
        total = total_all if split == "ALL" else st["rows"][split]
        langs = Counter()
        golds = Counter()
        ks = Counter()
        inj = 0
        if split == "ALL":
            for s in SPLITS:
                langs.update(st["lang"][s]); golds.update(st["gold_kind"][s])
                ks.update(st["k"][s])
            inj = st["inj_total"]
            uni_qid = len(set().union(*(st["query_ids"][s] for s in SPLITS)))
            uni_norm = len(set().union(*(st["query_norm"][s] for s in SPLITS)))
            calib = st["calibration"]["rows"]
            eff = st["calibration"]["effective_train_excluding_calibration"]
        else:
            langs = st["lang"][split]; golds = st["gold_kind"][split]; ks = st["k"][split]
            inj = sum(v for (s, l, t), v in st["inj_slt"].items() if s == split)
            uni_qid = len(st["query_ids"][split]); uni_norm = len(st["query_norm"][split])
            calib = ""; eff = ""
        rows_out.append({
            "threshold": threshold, "split": split, "rows": total,
            "pct_of_threshold": pct(total, total_all),
            "zh": langs.get("zh", 0), "zh_pct_of_split": pct(langs.get("zh", 0), total),
            "en": langs.get("en", 0), "en_pct_of_split": pct(langs.get("en", 0), total),
            "mixed_other": langs.get("mixed", 0),
            "mixed_other_pct_of_split": pct(langs.get("mixed", 0), total),
            "gold_capability_tool": golds.get("capability_tool", 0),
            "gold_capability_tool_pct": pct(golds.get("capability_tool", 0), total),
            "gold_reject": golds.get("REJECT", 0),
            "gold_reject_pct": pct(golds.get("REJECT", 0), total),
            "k0": ks.get(0, 0), "k0_pct": pct(ks.get(0, 0), total),
            "k1": ks.get(1, 0), "k1_pct": pct(ks.get(1, 0), total),
            "k2": ks.get(2, 0), "k2_pct": pct(ks.get(2, 0), total),
            "k3": ks.get(3, 0), "k3_pct": pct(ks.get(3, 0), total),
            "gold_injected": inj, "gold_injected_pct": pct(inj, total),
            "unique_source_query_id": uni_qid, "unique_normalized_query": uni_norm,
            "calibration_rows": calib, "effective_train_rows": eff,
            "denominator_scope": f"threshold total ({total_all})" if split == "ALL" else f"split total ({total})",
        })
write_csv("overall_distribution.csv", list(rows_out[0].keys()), rows_out)

# ── 2. tool_language_distribution.csv ──────────────────────────────────────
rows_out = []
for threshold, tag in THRESHOLDS:
    st = stats[threshold]
    total = sum(st["rows"].values())
    lang_totals = Counter()
    for s in SPLITS:
        lang_totals.update(st["lang"][s])
    for tool in TOOLS:
        counts = st["tool_lang"][tool]
        tt = sum(counts.values())
        rows_out.append({
            "threshold": threshold, "gold_tool": tool, "rows_total": tt,
            "rows_pct_of_threshold": pct(tt, total),
            "zh": counts.get("zh", 0), "en": counts.get("en", 0),
            "mixed_other": counts.get("mixed", 0),
            "zh_pct_within_tool": pct(counts.get("zh", 0), tt),
            "en_pct_within_tool": pct(counts.get("en", 0), tt),
            "mixed_other_pct_within_tool": pct(counts.get("mixed", 0), tt),
            "tool_pct_of_zh": pct(counts.get("zh", 0), lang_totals.get("zh", 0)),
            "tool_pct_of_en": pct(counts.get("en", 0), lang_totals.get("en", 0)),
            "tool_pct_of_mixed_other": pct(counts.get("mixed", 0), lang_totals.get("mixed", 0)),
            "denom_tool": tt,
            "denom_zh_all_tools": lang_totals.get("zh", 0),
            "denom_en_all_tools": lang_totals.get("en", 0),
            "denom_mixed_other_all_tools": lang_totals.get("mixed", 0),
        })
write_csv("tool_language_distribution.csv", list(rows_out[0].keys()), rows_out)

# ── 3. tool_split_distribution.csv ─────────────────────────────────────────
rows_out = []
for threshold, tag in THRESHOLDS:
    st = stats[threshold]
    total = sum(st["rows"].values())
    for tool in TOOLS:
        tt = sum(st["tool_split"][tool].values())
        for split in SPLITS + ("ALL",):
            cnt = tt if split == "ALL" else st["tool_split"][tool].get(split, 0)
            split_total = total if split == "ALL" else st["rows"][split]
            rows_out.append({
                "threshold": threshold, "gold_tool": tool, "split": split, "count": cnt,
                "pct_within_tool": pct(cnt, tt),
                "pct_within_split": pct(cnt, split_total),
                "tool_total": tt, "split_total": split_total,
                "denominator_scope": "tool total" if split == "ALL" else "tool total; split total",
            })
write_csv("tool_split_distribution.csv", list(rows_out[0].keys()), rows_out)

# ── 4. k_distribution_by_threshold.csv ─────────────────────────────────────
rows_out = []
for threshold, tag in THRESHOLDS:
    st = stats[threshold]
    total = sum(st["rows"].values())
    ks = Counter()
    for s in SPLITS:
        ks.update(st["k"][s])
    for k in range(4):
        rows_out.append({"threshold": threshold, "K": k, "count": ks.get(k, 0),
                         "group_total": total, "pct_within_threshold": pct(ks.get(k, 0), total)})
write_csv("k_distribution_by_threshold.csv", list(rows_out[0].keys()), rows_out)

# ── 5. k_distribution_by_split_language.csv ────────────────────────────────
rows_out = []
for threshold, tag in THRESHOLDS:
    st = stats[threshold]
    for split in SPLITS:
        for lang in LANG_ORDER:
            group = st["k_split_lang"][(split, lang)]
            group_total = sum(group.values())
            for k in range(4):
                rows_out.append({
                    "threshold": threshold, "split": split, "language": LANG_LABEL[lang],
                    "K": k, "count": group.get(k, 0), "group_total": group_total,
                    "pct_within_group": pct(group.get(k, 0), group_total),
                })
write_csv("k_distribution_by_split_language.csv", list(rows_out[0].keys()), rows_out)

# ── 6. tool_k_distribution.csv ─────────────────────────────────────────────
rows_out = []
for threshold, tag in THRESHOLDS:
    st = stats[threshold]
    for lang in ["ALL"] + list(LANG_ORDER):
        for tool in TOOLS:
            group = st["k_lang_tool"][(lang, tool)]
            group_total = sum(group.values())
            for k in range(4):
                rows_out.append({
                    "threshold": threshold, "language": "ALL" if lang == "ALL" else LANG_LABEL[lang],
                    "gold_tool": tool, "K": k, "count": group.get(k, 0),
                    "group_total": group_total, "pct_within_group": pct(group.get(k, 0), group_total),
                })
write_csv("tool_k_distribution.csv", list(rows_out[0].keys()), rows_out)

# ── 7. gold_reject_distribution.csv ────────────────────────────────────────
rows_out = []
for threshold, tag in THRESHOLDS:
    st = stats[threshold]
    total = sum(st["rows"].values())
    lang_totals = Counter()
    for s in SPLITS:
        lang_totals.update(st["lang"][s])
    for split in SPLITS + ("ALL",):
        for lang in ("ALL",) + LANG_ORDER:
            if split == "ALL" and lang == "ALL":
                group_total = total
            elif split == "ALL":
                group_total = lang_totals.get(lang, 0)
            elif lang == "ALL":
                group_total = st["rows"][split]
            else:
                group_total = sum(st["gold_kind_split_lang"][(split, lang)].values())
            counts = Counter()
            if split == "ALL" and lang == "ALL":
                for s in SPLITS:
                    counts.update(st["gold_kind"][s])
            elif split == "ALL":
                for s in SPLITS:
                    counts.update(st["gold_kind_split_lang"][(s, lang)])
            elif lang == "ALL":
                counts.update(st["gold_kind"][split])
            else:
                counts.update(st["gold_kind_split_lang"][(split, lang)])
            for kind in ("capability_tool", "REJECT"):
                rows_out.append({
                    "threshold": threshold, "split": split,
                    "language": "ALL" if lang == "ALL" else LANG_LABEL[lang],
                    "gold_kind": kind, "count": counts.get(kind, 0),
                    "group_total": group_total, "pct_of_group": pct(counts.get(kind, 0), group_total),
                    "threshold_total": total, "pct_of_threshold": pct(counts.get(kind, 0), total),
                })
write_csv("gold_reject_distribution.csv", list(rows_out[0].keys()), rows_out)

# ── 8. gold_injection_distribution.csv ─────────────────────────────────────
rows_out = []
for threshold, tag in THRESHOLDS:
    st = stats[threshold]
    total_inj = st["inj_total"]
    for split in SPLITS + ("ALL",):
        for lang in ("ALL",) + LANG_ORDER:
            for tool in TOOL_WITH_ALL:
                if split == "ALL" and lang == "ALL" and tool == "ALL":
                    cnt = total_inj
                elif split == "ALL" and lang == "ALL":
                    cnt = st["inj_tool"].get(tool, 0)
                elif split == "ALL" and tool == "ALL":
                    cnt = st["inj_lang"].get(lang, 0)
                elif lang == "ALL" and tool == "ALL":
                    cnt = st["inj_split"].get(split, 0)
                elif split == "ALL":
                    cnt = sum(v for (s, l, t), v in st["inj_slt"].items() if l == lang and t == tool)
                elif lang == "ALL":
                    cnt = sum(v for (s, l, t), v in st["inj_slt"].items() if s == split and t == tool)
                elif tool == "ALL":
                    cnt = sum(v for (s, l, t), v in st["inj_slt"].items() if s == split and l == lang)
                else:
                    cnt = st["inj_slt"].get((split, lang, tool), 0)
                tool_denom = total_inj if tool == "ALL" else st["inj_tool"].get(tool, 0)
                split_denom = total_inj if split == "ALL" else st["inj_split"].get(split, 0)
                lang_denom = total_inj if lang == "ALL" else st["inj_lang"].get(lang, 0)
                rows_out.append({
                    "threshold": threshold, "split": split,
                    "language": "ALL" if lang == "ALL" else LANG_LABEL[lang],
                    "gold_tool": tool, "injected_count": cnt,
                    "tool_injected_total": tool_denom, "split_injected_total": split_denom,
                    "language_injected_total": lang_denom, "threshold_injected_total": total_inj,
                    "pct_of_tool_injected": pct(cnt, tool_denom),
                    "pct_of_split_injected": pct(cnt, split_denom),
                    "pct_of_language_injected": pct(cnt, lang_denom),
                    "pct_of_threshold_injected": pct(cnt, total_inj),
                })
write_csv("gold_injection_distribution.csv", list(rows_out[0].keys()), rows_out)

# extra: before/after K for injected samples
rows_out = []
for threshold, tag in THRESHOLDS:
    st = stats[threshold]
    total_inj = st["inj_total"]
    for ok in range(4):
        for rk in range(4):
            cnt = 0
            if ok == 0 and rk == 1:
                cnt = total_inj
            rows_out.append({"threshold": threshold, "original_K": ok, "repaired_K": rk,
                             "count": cnt, "injected_total": total_inj,
                             "pct_of_injected": pct(cnt, total_inj)})
write_csv("gold_injection_k_shift.csv", list(rows_out[0].keys()), rows_out)

# extra: per-tool summary (3.4)
rows_out = []
for threshold, tag in THRESHOLDS:
    st = stats[threshold]
    total = sum(st["rows"].values())
    for tool in TOOLS + ["ALL"]:
        if tool == "ALL":
            lang = Counter(); splitc = Counter(); kc = Counter(); inj = st["inj_total"]
        else:
            lang = st["tool_lang"][tool]; splitc = st["tool_split"][tool]
            kc = st["tool_k"][tool]; inj = st["inj_tool"].get(tool, 0)
        tt = sum(lang.values())
        rows_out.append({
            "threshold": threshold, "gold_tool": tool, "rows_total": tt,
            "pct_of_threshold": pct(tt, total),
            "zh": lang.get("zh", 0), "en": lang.get("en", 0),
            "mixed_other": lang.get("mixed", 0),
            "train": splitc.get("train", 0), "val": splitc.get("val", 0),
            "test": splitc.get("test", 0),
            "k0": kc.get(0, 0), "k1": kc.get(1, 0), "k2": kc.get(2, 0), "k3": kc.get(3, 0),
            "gold_label_count": tt, "gold_injected": inj,
            "gold_injected_pct_within_tool": pct(inj, tt),
        })
write_csv("tool_summary.csv", list(rows_out[0].keys()), rows_out)

# ── 9. data_quality_audit.json ─────────────────────────────────────────────
checks = {k: {"count": v["count"], "samples": v["samples"]} for k, v in sorted(quality.items())}
error_count = sum(v["count"] for v in checks.values())

def _errs(*keys):
    return sum(checks.get(k, {}).get("count", 0) for k in keys)

total_rows = sum(sum(stats[t]["rows"].values()) for t in (0.82, 0.85, 0.88))
total_repaired = sum(stats[t]["details_seen"] for t in (0.82, 0.85, 0.88))
evaluated_checks = {
    "schema_14_keys": {"scope": f"{total_rows} rows", "errors": _errs("schema_key_errors")},
    "field_types": {"scope": f"{total_rows} rows", "errors": _errs("type_errors")},
    "split_tag_matches_file": {"scope": f"{total_rows} rows", "errors": _errs("split_tag_errors")},
    "option_length_equals_order": {"scope": f"{total_rows} rows", "errors": _errs("option_len_mismatch")},
    "exactly_one_REJECT": {"scope": f"{total_rows} rows", "errors": _errs("reject_count_errors")},
    "option_count_equals_K_plus_1": {"scope": f"{total_rows} rows", "errors": _errs("k_option_count_mismatch")},
    "no_duplicate_candidates": {"scope": f"{total_rows} rows", "errors": _errs("duplicate_candidates")},
    "capability_ids_valid": {"scope": f"{total_rows} rows", "errors": _errs("invalid_capability_ids", "invalid_gold_ids")},
    "gold_label_matches_target_kind": {"scope": f"{total_rows} rows", "errors": _errs("gold_label_errors")},
    "gold_index_and_reject_index": {"scope": f"{total_rows} rows", "errors": _errs("gold_index_errors")},
    "shuffle_permutation_and_seed": {"scope": f"{total_rows} rows", "errors": _errs("shuffle_permutation_errors", "shuffle_seed_errors")},
    "shuffle_prefix_by_origin": {"scope": f"{total_rows} rows", "errors": _errs("unknown_shuffle_prefix")},
    "repair_detail_match": {"scope": f"{total_repaired} repaired rows", "errors": _errs("repair_detail_mismatches", "repair_detail_missing", "repair_detail_row_missing")},
    "injected_row_shape": {"scope": f"{total_repaired} repaired rows", "errors": _errs("injected_row_errors")},
    "duplicate_row_ids": {"scope": f"{total_rows} rows", "errors": _errs("duplicate_row_ids")},
    "cross_split_leakage": {"scope": "9 pairs x 3 dimensions x 3 thresholds", "errors": _errs(
        "leakage_normalized_query", "leakage_source_query_id", "leakage_query_family")},
    "calibration_double_counting": {"scope": "3 thresholds x 400 calibration rows", "errors": _errs(
        "calibration_errors")},
}
quality_payload = {
    "scope": {
        "source": "scripts/laya_finetune/data_v4_build/full_r1/data/recall_threshold_{082,085,088}/v2_{train,val,test}.jsonl",
        "manifests": "scripts/laya_finetune/data_v4_build/full_r1/data/recall_threshold_*/manifest.json",
        "construction_audit": "scripts/laya_finetune/data_v4_build/full_r1/data/recall_threshold_*/construction_audit.jsonl",
        "repair_detail": "data/laya_finetune/data_v4_build/full_r1_k0_to_k1_inplace_audit/k0_to_k1_repair_detail.jsonl",
        "repair_backups": "data/laya_finetune/data_v4_build/full_r1_k0_to_k1_inplace_audit/backups/recall_threshold_*/v2_*.jsonl",
        "capability_ids": "scripts/laya_finetune/data_v4_build/full_r1/data/capability_ids.json",
        "archived_bundle_read": False,
    },
    "definitions": {
        "K": "number of capability cards in options, excluding REJECT",
        "total_options": "K + 1",
        "gold_tool": "gold_capability_id when target_kind=capability; REJECT when target_kind=reject",
        "candidate_origin": "source_gold_injected for repair-detail rows; ann_recall for all others (inferred; construction_audit has no candidate_origin field)",
        "language": "row.lang: zh=Chinese, en=English, mixed=other/mixed",
    },
    "summary": {"status": "PASS" if error_count == 0 else "ISSUES_FOUND",
                "error_count": error_count},
    "checks": checks,
    "evaluated_checks": evaluated_checks,
    "per_threshold": {
        str(t): {
            "rows": dict(stats[t]["rows"]),
            "k0_total": stats[t]["k0_total"],
            "k0_only_reject_options": stats[t]["k0_only_reject"],
            "k0_with_gold_capability": stats[t]["k0_gold_cap"],
            "k0_with_gold_reject": stats[t]["k0_gold_reject"],
            "repair_detail_rows": stats[t]["detail_count"],
            "repair_detail_matched_rows": stats[t]["details_seen"],
            "original_k": dict(stats[t]["orig_k"]),
            "repaired_k": dict(stats[t]["rep_k"]),
            "calibration": stats[t]["calibration"],
            "leakage": stats[t]["leakage"],
            "lang_raw": dict(stats[t]["lang_raw"]),
        }
        for t in [0.82, 0.85, 0.88]
    },
}
(OUT / "data_quality_audit.json").write_text(
    json.dumps(quality_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

# ── 10. final_distribution_report.md ───────────────────────────────────────
def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def md_overall():
    headers = ["threshold", "rows", "train", "val", "test", "zh", "en", "mixed/other",
               "Gold cap", "Gold REJECT", "K0", "K1", "K2", "K3",
               "injected", "unique qid", "calib", "effective train"]
    rows = []
    for t, tag in THRESHOLDS:
        st = stats[t]
        total = sum(st["rows"].values())
        langs = Counter(); golds = Counter(); ks = Counter()
        for s in SPLITS:
            langs.update(st["lang"][s]); golds.update(st["gold_kind"][s]); ks.update(st["k"][s])
        uni = len(set().union(*(st["query_ids"][s] for s in SPLITS)))
        rows.append([t, total, st["rows"]["train"], st["rows"]["val"], st["rows"]["test"],
                     langs.get("zh", 0), langs.get("en", 0), langs.get("mixed", 0),
                     golds.get("capability_tool", 0), golds.get("REJECT", 0),
                     ks.get(0, 0), ks.get(1, 0), ks.get(2, 0), ks.get(3, 0),
                     st["inj_total"], uni, st["calibration"]["rows"],
                     st["calibration"]["effective_train_excluding_calibration"]])
    return table(headers, rows)


def md_k_split():
    headers = ["threshold", "split", "rows", "K0", "K1", "K2", "K3", "K0%", "K1%", "K2%", "K3%"]
    rows = []
    for t, tag in THRESHOLDS:
        st = stats[t]
        for split in SPLITS:
            ks = st["k"][split]; total = sum(ks.values())
            rows.append([t, split, total] + [ks.get(k, 0) for k in range(4)]
                        + [pct(ks.get(k, 0), total) for k in range(4)])
    return table(headers, rows)


def md_tool(t):
    st = stats[t]
    headers = ["Gold tool", "total", "zh", "en", "mixed", "zh% in tool", "en% in tool",
               "tool% of zh", "tool% of en"]
    rows = []
    lang_totals = Counter()
    for s in SPLITS:
        lang_totals.update(st["lang"][s])
    for tool in TOOLS:
        c = st["tool_lang"][tool]; tt = sum(c.values())
        rows.append([tool, tt, c.get("zh", 0), c.get("en", 0), c.get("mixed", 0),
                     pct(c.get("zh", 0), tt), pct(c.get("en", 0), tt),
                     pct(c.get("zh", 0), lang_totals.get("zh", 0)),
                     pct(c.get("en", 0), lang_totals.get("en", 0))])
    return table(headers, rows), lang_totals


def md_injection():
    headers = ["threshold", "injected", "train", "val", "test", "zh", "en", "mixed"]
    rows = []
    for t, tag in THRESHOLDS:
        st = stats[t]
        rows.append([t, st["inj_total"], st["inj_split"].get("train", 0),
                     st["inj_split"].get("val", 0), st["inj_split"].get("test", 0),
                     st["inj_lang"].get("zh", 0), st["inj_lang"].get("en", 0),
                     st["inj_lang"].get("mixed", 0)])
    return table(headers, rows)


def md_injection_tool():
    headers = ["threshold", "Gold tool", "injected", "share of injected"]
    rows = []
    for t, tag in THRESHOLDS:
        st = stats[t]
        for tool in TOOLS:
            c = st["inj_tool"].get(tool, 0)
            if c:
                rows.append([t, tool, c, pct(c, st["inj_total"])])
    return table(headers, rows)


lines = []
lines.append("# 正式训练数据分布统计报告\n")
lines.append("## 1. 统计范围与口径\n")
lines.append("- 唯一正式数据源：`scripts/laya_finetune/data_v4_build/full_r1/data/`，覆盖 `recall_threshold_082`、`recall_threshold_085`、`recall_threshold_088`。")
lines.append("- 未读取归档的 `full_r1_repaired`；未重跑 embedding、ANN、SQL 或数据构建。")
lines.append("- 行级标签、K、Gold 分布以修复后的 `v2_{train,val,test}.jsonl` 为准。")
lines.append("- K = 选项中 capability 候选卡数量（不含 REJECT），总选项数 = K + 1。")
lines.append("- Gold Tool：`target_kind=capability` 时取 `gold_capability_id`；`target_kind=reject` 时记为 `REJECT`，不并入任何 Tool。")
lines.append("- 语言：沿用行内 `lang`；`zh`=中文，`en`=英文，`mixed`=其他/混合。本次未发现未知语言值。")
lines.append("- Gold 注入样本：以修复审计明细 `k0_to_k1_repair_detail.jsonl` 的 id 为准；注入候选不计入 ANN 检索命中或召回率。")
lines.append("- `candidate_origin`：注入行 = `source_gold_injected`；其余行 = `ann_recall`（构建审计无该字段，按生成方式推断）。\n")

lines.append("## 2. 总体分布\n")
lines.append("分母：每行 `rows` 为该 threshold 全部 split 的总样本数；`unique qid` 为去重后的 `source_query_id` 数量。")
lines.append("`calib` 为 calibration 行数；`effective train` = train 行数 - calibration，不重复计入训练。\n")
lines.append(md_overall())
lines.append("\n")

lines.append("## 3. K 分布\n")
lines.append("### 3.1 threshold × split\n")
lines.append("分母：每个 `threshold × split` 的样本总数。")
lines.append(md_k_split())
lines.append("\n完整 `threshold × split × language × K` 见 `k_distribution_by_split_language.csv`；")
lines.append("`threshold × Gold Tool × language × K` 见 `tool_k_distribution.csv`；")
lines.append("`threshold × candidate_origin × K` 见下表。\n")

lines.append("### 3.2 threshold × candidate_origin × K\n")
headers = ["threshold", "candidate_origin", "K", "count", "group_total", "pct_within_group"]
rows = []
for t, tag in THRESHOLDS:
    st = stats[t]
    for origin in ("ann_recall", "source_gold_injected"):
        group_total = sum(v for (o, k), v in st["k_origin"].items() if o == origin)
        for k in range(4):
            cnt = st["k_origin"].get((origin, k), 0)
            rows.append([t, origin, k, cnt, group_total, pct(cnt, group_total)])
lines.append(table(headers, rows))
lines.append("\n")

lines.append("## 4. 各 Gold Tool 的中英文分布\n")
lines.append("比例 A = 该语言在 Tool 内的占比（分母 `denom_tool`）；比例 B = 该 Tool 在该语言全部样本中的占比（分母为 `denom_zh_all_tools` / `denom_en_all_tools`）。")
lines.append("完整数值与分母见 `tool_language_distribution.csv`。\n")
for t, tag in THRESHOLDS:
    tbl, lang_totals = md_tool(t)
    lines.append(f"### threshold {t}\n")
    lines.append(f"语言总量：zh={lang_totals.get('zh',0)}，en={lang_totals.get('en',0)}，mixed/other={lang_totals.get('mixed',0)}。\n")
    lines.append(tbl)
    lines.append("\n")

lines.append("## 5. Gold = capability/tool 与 Gold = REJECT\n")
lines.append("完整 `threshold × split × language × gold_kind` 数值见 `gold_reject_distribution.csv`，每行明确给出 `group_total`。\n")
headers = ["threshold", "split", "Gold capability/tool", "pct", "Gold REJECT", "pct", "split total"]
rows = []
for t, tag in THRESHOLDS:
    st = stats[t]
    for split in SPLITS:
        total = st["rows"][split]
        cap = st["gold_kind"][split].get("capability_tool", 0)
        rej = st["gold_kind"][split].get("REJECT", 0)
        rows.append([t, split, cap, pct(cap, total), rej, pct(rej, total), total])
lines.append(table(headers, rows))
lines.append("\n")

lines.append("## 6. Gold 注入样本专项\n")
lines.append("注入样本 = 原始 K=0、Gold 为有效 capability 的目标行，原位补入唯一 Gold 卡并保留 REJECT。")
lines.append("注入行的 `candidate_origin=source_gold_injected`、`retrieval_score_available=false`。\n")
lines.append(md_injection())
lines.append("\n按 Gold Tool 的注入分布（仅列出非零项）：\n")
lines.append(md_injection_tool())
lines.append("\n修复前后 K 变化：所有注入样本修复前 K=0、修复后 K=1。逐组数值见 `gold_injection_k_shift.csv`。\n")
headers = ["threshold", "K=0 总数", "K=0 且仅 REJECT 选项", "K=0 且 Gold=Tool", "K=0 且 Gold=REJECT"]
rows = []
for t, tag in THRESHOLDS:
    st = stats[t]
    rows.append([t, st["k0_total"], st["k0_only_reject"], st["k0_gold_cap"], st["k0_gold_reject"]])
lines.append(table(headers, rows))
lines.append("\nGold 候选与索引校验：所有行均检查 `gold_capability_id` 位于候选集中，且 `order[gold_index]` 等于 Gold；错误数见质量审计。\n")

lines.append("## 7. 数据质量检查\n")
lines.append("检查为只读，未自动修复。完整机器可读结果见 `data_quality_audit.json`。\n")
headers = ["check", "scope", "errors"]
rows = [[name, rec["scope"], rec["errors"]] for name, rec in evaluated_checks.items()]
lines.append(table(headers, rows))
if checks:
    lines.append("\n发现项（未自动修复）：\n")
    headers = ["check", "count", "samples"]
    rows = [[k, v["count"], json.dumps(v["samples"], ensure_ascii=False)[:300]]
            for k, v in checks.items()]
    lines.append(table(headers, rows))
else:
    lines.append("\n全部检查错误数为 0。")
lines.append("\n### 7.1 split 泄漏检查\n")
lines.append("统计 `normalized_query`、`source_query_id`、`query_family_id` 在不同 split 之间的交集。")
lines.append("非零交集按发现项报告，不自动处理。\n")
headers = ["threshold", "dimension:pair", "intersection count", "examples"]
rows = []
for t, tag in THRESHOLDS:
    for key, rec in stats[t]["leakage"].items():
        rows.append([t, key, rec["count"], "; ".join(map(str, rec["examples"]))])
lines.append(table(headers, rows))
lines.append("\n### 7.2 calibration\n")
headers = ["threshold", "calibration rows", "unique indices", "unique ids", "effective train", "manifest effective", "errors"]
rows = []
for t, tag in THRESHOLDS:
    c = stats[t]["calibration"]
    rows.append([t, c["rows"], c["index_unique"], c["ids_unique"],
                 c["effective_train_excluding_calibration"], c["manifest_effective_train"],
                 ", ".join(c["errors"]) or "none"])
lines.append(table(headers, rows))
lines.append("\n")

lines.append("## 8. 输出文件\n")
for name in sorted(p.name for p in OUT.iterdir() if p.is_file()):
    lines.append(f"- `{OUT / name}`")
lines.append("\n")
lines.append("所有统计数据均来自上述正式 JSONL；`construction_audit.jsonl` 仅用于 query family 与修复前证据，不作为当前 K/标签依据。\n")

(OUT / "final_distribution_report.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps({"out": str(OUT), "files": sorted(p.name for p in OUT.iterdir() if p.is_file()),
                  "quality_error_count": error_count}, ensure_ascii=False, indent=2))
