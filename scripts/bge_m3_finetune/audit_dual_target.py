#!/usr/bin/env python3
"""Read-only audit of scripts/bge_m3_finetune/data/ against the TWO Delveta
retrieval chains (Agent = Query->Capability Description; Intent Funnel =
Query->Standard/Similar corpus).

Evidence sources (all read-only):
  data/bge_m3_{train,test}.jsonl                the {query,pos,neg} rows
  data/bge_m3_{train,test}_manifest.jsonl       per-row provenance
  logs/_registry_dump.json                      live capability registry snapshot
                                                (description en+zh, standard+similar
                                                queries per capability)

Emits a structured report; never writes any data file.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "scripts" / "bge_m3_finetune" / "data"
REG = ROOT / "logs" / "_registry_dump.json"

# curated confusable/hard-negative pairs (logs/_cap_only_eval.py DISTRACTORS,
# frozen 2026-09-22). Read here as a literal so this audit has no import side
# effects on the eval module.
CONFUSABLE = {
    "cap-add-term": ("cap-create-folder", "cap-edit-file"),
    "cap-artifact": ("cap-slides", "cap-summary"),
    "cap-bash": ("cap-edit-file", "cap-read-file"),
    "cap-create-folder": ("cap-add-term", "cap-bash"),
    "cap-edit-file": ("cap-bash", "cap-read-file"),
    "cap-mindmap": ("cap-slides", "cap-summary"),
    "cap-pdf-extract-text": ("cap-pdf-table-to-text", "cap-read-document"),
    "cap-pdf-table-to-text": ("cap-pdf-extract-text", "cap-summary"),
    "cap-rag-search": ("cap-web-search", "cap-social-search"),
    "cap-read-document": ("cap-pdf-extract-text", "cap-read-file"),
    "cap-read-file": ("cap-read-document", "cap-edit-file"),
    "cap-slides": ("cap-mindmap", "cap-artifact"),
    "cap-social-search": ("cap-web-search", "cap-rag-search"),
    "cap-summary": ("cap-mindmap", "cap-artifact"),
    "cap-translate": ("cap-summary", "cap-read-document"),
    "cap-vision": ("cap-read-document", "cap-pdf-extract-text"),
    "cap-web-search": ("cap-social-search", "cap-rag-search"),
}


def read_jsonl(p: Path) -> list[dict]:
    out = []
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def norm(text: str) -> str:
    s = unicodedata.normalize("NFKC", str(text or "")).lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\u4e00-\u9fff ]", "", s, flags=re.UNICODE)
    return s.strip()


def ngrams(text: str, n: int = 4) -> set[str]:
    s = norm(text)
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))} or {s}


def cosine(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / ((len(a) ** 0.5) * (len(b) ** 0.5))


def main() -> None:
    tr = read_jsonl(DATA / "bge_m3_train.jsonl")
    te = read_jsonl(DATA / "bge_m3_test.jsonl")
    mt = read_jsonl(DATA / "bge_m3_train_manifest.jsonl")
    me = read_jsonl(DATA / "bge_m3_test_manifest.jsonl")
    reg = {e["capability_id"]: e for e in json.load(open(REG, encoding="utf-8"))}

    # ---- corpus map from the registry (source of truth for chain B) --------
    cap2queries: dict[str, set[str]] = {}
    std_n = sim_n = 0
    for cid, e in reg.items():
        qs: set[str] = set()
        for st in e.get("standard", []):
            std_n += 1
            qs.add(st["query"].strip())
            for sim in st.get("similar", []):
                sim_n += 1
                qs.add(sim["query"].strip())
        cap2queries[cid] = qs
    q2caps: dict[str, set[str]] = defaultdict(set)
    for cid, qs in cap2queries.items():
        for q in qs:
            q2caps[q].add(cid)

    def desc_parts(cid: str) -> tuple[str, str]:
        d = reg[cid].get("description", "") or ""
        if "中文：" in d:
            en, zh = d.split("中文：", 1)
            return en.strip(), zh.strip()
        return d.strip(), ""

    print("=" * 78)
    print("A. DATA ASSETS")
    print("=" * 78)
    print(f"registry: {len(reg)} capabilities | standard={std_n} similar={sim_n} total={std_n+sim_n}")
    for name, rows, man in (("train", tr, mt), ("test", te, me)):
        print(f"{name}: rows={len(rows)} manifest={len(man)} fields={list(rows[0].keys())}")
        print(f"  sources: {dict(Counter(m.get('source') for m in man))}")
        print(f"  langs:   {dict(Counter(m.get('lang') for m in man))}")

    # ---- corpus integrity --------------------------------------------------
    print("\ncorpus dupes / cross-cap overlap:")
    allq = [q for qs in cap2queries.values() for q in qs]
    dups = {q: c for q, c in Counter(allq).items() if c > 1}
    xcap = {q: sorted(c) for q, c in q2caps.items() if len(c) > 1}
    print(f"  duplicate query texts within corpus: {len(dups)}")
    print(f"  queries claimed by >1 capability:   {len(xcap)}")
    for q, cs in list(xcap.items())[:6]:
        print(f"    {q!r} -> {cs}")

    # ---- task validation ---------------------------------------------------
    print("\n" + "=" * 78)
    print("B. TASK A (Description) vs TASK B (Query corpus) — is pos valid?")
    print("=" * 78)
    def analyze(rows, man, label):
        pos_in_gold_corpus = pos_is_desc = pos_in_other = 0
        desc_marker = ("Use it when", "does not apply", "中文：", "tool:", "does:")
        for r, m in zip(rows, man):
            gold = m["capability_id"]
            for p in r["pos"]:
                ps = str(p)
                if any(k in ps for k in desc_marker) or len(ps) > 120:
                    pos_is_desc += 1
                elif ps.strip() in cap2queries.get(gold, set()):
                    pos_in_gold_corpus += 1
                elif any(ps.strip() in cap2queries.get(c, set()) for c in cap2queries):
                    pos_in_other += 1
        print(f"[{label}] pos entries: desc-like={pos_is_desc} "
              f"in-gold-corpus={pos_in_gold_corpus} in-other-corpus={pos_in_other}")
    analyze(tr, mt, "train")
    analyze(te, me, "test")

    # ---- negative classification ------------------------------------------
    print("\n" + "=" * 78)
    print("C. NEGATIVE QUALITY (structural — semantic verdicts are candidates only)")
    print("=" * 78)
    cat = Counter()
    samples = defaultdict(list)
    per_cap_negcat = defaultdict(Counter)
    for split, rows, man in (("train", tr, mt), ("test", te, me)):
        for r, m in zip(rows, man):
            gold = m["capability_id"]
            conf = set(CONFUSABLE.get(gold, ()))
            for n in r["neg"]:
                ns = n.strip()
                owner = q2caps.get(ns, set())
                if gold in owner:
                    k = "4_wrong_negative(actually gold)"
                elif owner & conf:
                    k = "2_hard_but_valid(curated confusable)"
                elif owner and owner.isdisjoint({gold}):
                    k = "1_valid(other capability)"
                elif not owner:
                    k = "5_undeterminable(not in any corpus)"
                else:
                    k = "other"
                cat[k] += 1
                per_cap_negcat[gold][k] += 1
                if len(samples[k]) < 4:
                    samples[k].append((split, gold, ns[:60], sorted(owner)))
    tot_neg = sum(cat.values())
    for k, v in cat.most_common():
        print(f"  {k:42} {v:6}  ({v/tot_neg*100:5.1f}%)")
    for k, ex in samples.items():
        print(f"  e.g. {k}: {ex[:2]}")

    # cap-research negatives specifically (audit doc §11 claims random cross-cap)
    rs = [(m['capability_id'], r['neg']) for r, m in zip(tr, mt) if m['capability_id'] == 'cap-research']
    if rs:
        print(f"  cap-research train rows: {len(rs)}  neg[0] sample: {rs[0][1][:2]}")

    # ---- leakage -----------------------------------------------------------
    print("\n" + "=" * 78)
    print("D. LEAKAGE")
    print("=" * 78)
    trex = {r["query"] for r in tr}
    teex = {r["query"] for r in te}
    print(f"exact train∩test: {len(trex & teex)}")
    trn = {norm(r["query"]) for r in tr}
    ten = {norm(r["query"]) for r in te}
    print(f"normalized train∩test: {len(trn & ten)}")
    corpus_norm = {norm(q) for q in allq}
    te_in_corpus = ten & corpus_norm
    print(f"test queries also present in the 908 training corpus (index contamination): {len(te_in_corpus)}")
    for q in list(te_in_corpus)[:5]:
        print(f"    {q!r}")

    # near-dup candidates: train->test via 4-gram inverted index (candidates only)
    idx = defaultdict(list)
    for i, r in enumerate(tr):
        for g in ngrams(r["query"]):
            idx[g].append(i)
    hits = Counter()
    ex = []
    for r in te:
        cand = Counter()
        for g in set(ngrams(r["query"])):
            for i in idx.get(g, ()):  # each gram can be shared by many -> bounded below
                cand[i] += 1
        best = 0.0
        for i, shared in cand.items():
            if shared < 6:
                continue
            c = cosine(ngrams(r["query"]), ngrams(tr[i]["query"]))
            if c > best:
                best = c
        for th in (0.80, 0.90, 0.95):
            if best >= th:
                hits[th] += 1
        if best >= 0.90 and len(ex) < 6:
            ex.append((r["query"][:50], best))
    print(f"near-dup candidates (train x test, char 4-gram cosine): "
          f">=0.80:{hits[0.80]}  >=0.90:{hits[0.90]}  >=0.95:{hits[0.95]}  (of {len(te)})")
    for q, s in ex:
        print(f"    {s:.3f}  {q!r}")

    # ---- coverage matrix ---------------------------------------------------
    print("\n" + "=" * 78)
    print("E. COVERAGE MATRIX (per capability x lang): A-eligible = has usable description")
    print("=" * 78)
    trc = defaultdict(Counter)
    tec = defaultdict(Counter)
    for m in mt:
        trc[m["capability_id"]][m["lang"]] += 1
    for m in me:
        tec[m["capability_id"]][m["lang"]] += 1
    print(f"{'capability':<24}{'desc_en':>8}{'desc_zh':>8}{'A_tr_zh':>8}{'A_tr_en':>8}{'A_te_zh':>8}{'A_te_en':>8}{'B_tr':>7}{'B_te':>7}")
    for cid in sorted(reg):
        en, zh = desc_parts(cid)
        has_en = "Y" if en else "-"
        has_zh = "Y" if zh else "-"
        print(f"{cid:<24}{has_en:>8}{has_zh:>8}"
              f"{trc[cid]['zh']:>8}{trc[cid]['en']:>8}{tec[cid]['zh']:>8}{tec[cid]['en']:>8}"
              f"{sum(trc[cid].values()):>7}{sum(tec[cid].values()):>7}")


if __name__ == "__main__":
    main()
