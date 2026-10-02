#!/usr/bin/env python3
"""V2 dataset step 1: capture the NO-THRESHOLD production recall for the three
V1 query sets that V2 is derived from.

Query sources (V1 frozen bundles, read-only):
  train  scripts/laya_finetune/data/layachoice_v1_train_b_noprov.jsonl.gz   (856)
  val    scripts/laya_finetune/data/layachoice_v1_val_b_noprov.jsonl.gz     (150)
  test   scripts/laya_finetune/data/layachoice_v1_test_b_noprov.jsonl.gz    (900)

Same production seams as the funnel / the frozen T2 diagnostic:
  recall.load_index(SessionLocal)                          -> live index (admin view)
  recall.retriever.ann_recall(index, qvec, min_score=0.0)  -> full pre-gate pool
  (NO threshold filtering is applied afterwards - that is the V2 change)

READ-ONLY: no DB writes, no production code touched.
Output (this dir):
  v2_sql_audit.json    corpus provenance + index digest
  v2_raw_recall.jsonl  1906 x {source, sample_id, query, language, gold, raw:[...]}
"""
from __future__ import annotations

import asyncio
import hashlib
import gzip
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
OUT = Path(__file__).resolve().parent
BUNDLE = REPO / "scripts" / "laya_finetune" / "data"


def load_split(split: str):
    path = BUNDLE / f"layachoice_v1_{split}_b_noprov.jsonl.gz"
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    out = []
    for r in rows:
        out.append({
            "source": split,
            "sample_id": r["id"],
            "query": r["state"],
            "language": r["lang"],
            "gold": r["order"][r["gold"]],
        })
    return out


async def sql_audit(session):
    from sqlalchemy import text as sql_text

    async def scalar(sql, **p):
        return (await session.execute(sql_text(sql), p)).scalar()

    return {
        "queried_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "capabilities_enabled_active_nonresearch": await scalar(
            "SELECT count(*) FROM capabilities WHERE enabled AND status='active' "
            "AND intent_kind <> 'research'"),
        "standard_total": await scalar("SELECT count(*) FROM capability_standard_queries"),
        "similar_total": await scalar("SELECT count(*) FROM capability_similar_queries"),
    }


async def main():
    from core.infrastructure.db import SessionLocal
    from core.infrastructure.vector import TEIEmbedder
    from core.application.chat.intent_funnel.recall import load_index
    from core.application.chat.intent_funnel.recall.retriever import ann_recall

    async with SessionLocal() as s:
        audit = await sql_audit(s)

    index = await load_index(SessionLocal)
    if index is None:
        sys.exit("recall index EMPTY -> RECALL_UNAVAILABLE")
    corpus_caps = sorted({c.capability_id for c in index.corpus})
    digest = hashlib.sha256(
        "|".join(f"{c.kind}:{c.capability_id}:{c.language}:{c.query}"
                 for c in sorted(index.corpus, key=lambda c: c.query_id))
        .encode("utf-8")).hexdigest()[:12]
    audit["index"] = {
        "version": index.version, "recomputed_digest": digest,
        "corpus_rows_union": len(index.corpus),
        "corpus_rows_standard": sum(1 for c in index.corpus if c.kind == "standard"),
        "corpus_rows_similar": sum(1 for c in index.corpus if c.kind == "similar"),
        "distinct_capabilities": len(corpus_caps), "capabilities": corpus_caps,
        "ann_pool_per_path": 64, "min_score_used": 0.0,
    }

    queries = load_split("train") + load_split("val") + load_split("test")
    by_source = Counter(q["source"] for q in queries)
    audit["query_sets"] = dict(by_source)
    golds = {q["gold"] for q in queries}
    audit["gold_missing_from_corpus"] = sorted(golds - set(corpus_caps))

    emb = TEIEmbedder(timeout=60.0)
    t0 = time.monotonic()
    qvs = []
    try:
        for i, q in enumerate(queries):
            for attempt in range(6):
                try:
                    qvs.append((await emb.embed([q["query"]]))[0])
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt == 5:
                        raise
                    await asyncio.sleep(0.5 * (attempt + 1))
            if (i + 1) % 200 == 0:
                print(f"[v2-capture] embedded {i+1}/{len(queries)} "
                      f"({time.monotonic()-t0:.0f}s)", flush=True)
    finally:
        await emb._client.aclose()

    with (OUT / "v2_raw_recall.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for q, qv in zip(queries, qvs):
            res = await ann_recall(index, qv, min_score=0.0)
            raw = [{"capability_id": c.capability_id, "score": round(c.score, 6),
                    "query_kind": c.query_kind, "source_query_id": c.query_id,
                    "source_query": c.matched_example, "language": c.language,
                    "standard_query_id": c.standard_query_id}
                   for c in res.candidates]
            fh.write(json.dumps({
                "source": q["source"], "sample_id": q["sample_id"], "query": q["query"],
                "language": q["language"], "gold_capability_id": q["gold"],
                "raw": raw}, ensure_ascii=False) + "\n")

    audit["capture"] = {"n_queries": len(queries),
                        "elapsed_sec": round(time.monotonic() - t0, 1),
                        "query_embedding_dim": len(qvs[0])}
    (OUT / "v2_sql_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[v2-capture] index={index.version} corpus={len(index.corpus)} "
          f"caps={len(corpus_caps)} sets={dict(by_source)}")
    print(f"[v2-capture] wrote v2_raw_recall.jsonl ({len(queries)} rows, "
          f"{audit['capture']['elapsed_sec']}s)")


if __name__ == "__main__":
    asyncio.run(main())
