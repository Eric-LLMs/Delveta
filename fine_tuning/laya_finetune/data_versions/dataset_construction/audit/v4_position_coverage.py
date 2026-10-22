"""Independent position-coverage test: REJECT and capability gold at every
legal slot, using the real tokenizer and the frozen V2 renderer."""
import json, sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

REPO = Path(r"e:/Work Space/Git Repositories/DeepDive")
V2 = REPO / "scripts/laya_finetune/V2"
RUN = REPO / "scripts/laya_finetune/data_v4_build/smoke50_r1"
for p in (V2, REPO / "packages", REPO / "deploy/laya"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from core.application.chat.intent_funnel.cap_router.card_renderer import (
        INSTRUCTIONS, REJECT_CARD, REJECT_LABEL, render_choice_question)
except ModuleNotFoundError:
    import importlib
    from types import ModuleType
    CHAIN = ("core", "core.application", "core.application.chat",
             "core.application.chat.intent_funnel",
             "core.application.chat.intent_funnel.cap_router")
    for n in [m for m in sys.modules if m == "core" or m.startswith("core.")]:
        del sys.modules[n]
    for n in CHAIN:
        stub = ModuleType(n)
        stub.__path__ = [str(REPO.joinpath("packages", *n.split(".")))]
        stub.__package__ = n
        sys.modules[n] = stub
    _r = importlib.import_module(
        "core.application.chat.intent_funnel.cap_router.card_renderer")
    INSTRUCTIONS, REJECT_CARD, REJECT_LABEL, render_choice_question = (
        _r.INSTRUCTIONS, _r.REJECT_CARD, _r.REJECT_LABEL, _r.render_choice_question)

import layachoice_v2_dataset as D
import layachoice_v2_render as R
import layachoice_v2_spec as S
import layachoice_sequence as SEQ
import laya.agent as A
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

root = Path(snapshot_download(repo_id=S.BASE_REPO, revision=S.BASE_REVISION,
                              allow_patterns=[f"{S.BASE_SUBFOLDER}/tokenizer/*"],
                              local_files_only=True))
model_dir = root / S.BASE_SUBFOLDER
A._fix_tokenizer_config(str(model_dir))
tok = AutoTokenizer.from_pretrained(str(model_dir / "tokenizer"))

rows = []
for split in ("train", "val", "test"):
    for line in (RUN / "data/threshold_085" / f"v2_{split}.jsonl").read_text(
            encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))

# Real-data coverage: where does REJECT sit, where does capability gold sit?
real_reject_pos = Counter()
real_gold_pos = Counter()
for row in rows:
    n = len(row["order"])
    k = n - 1
    real_reject_pos[(k, row["reject_index"])] += 1
    if row["target_kind"] == "capability":
        real_gold_pos[(k, row["gold_index"])] += 1

by_k = {}
for row in rows:
    k = len(row["order"]) - 1
    if k not in by_k:
        by_k[k] = row
assert sorted(by_k) == [0, 1, 2, 3], sorted(by_k)


def rebuild(row, new_order):
    """Rebuild a row with `new_order`, keeping metadata consistent."""
    positions = {cid: i for i, cid in enumerate(row["order"])}
    new = dict(row)
    new["order"] = list(new_order)
    new["options"] = [row["options"][positions[c]] for c in new_order]
    new["gold_index"] = new_order.index(row["order"][row["gold_index"]])
    new["reject_index"] = new_order.index(REJECT_LABEL)
    base = [c for c in row["order"] if c != REJECT_LABEL] + [REJECT_LABEL]
    new["shuffle"] = {"seed_prefix": S.SHUFFLE_SEED_PREFIX,
                      "permutation": [base.index(c) for c in new_order]}
    return new


fails = []
checks = 0


def check(cond, msg):
    global checks
    checks += 1
    if not cond:
        fails.append(msg)


def verify_variant(row, tag):
    ex = D.adapt_row(row)
    item = R.build_item(tok, ex, INSTRUCTIONS, S.MAX_LEN, S.HEAD_MAX_LEN)
    n = len(row["order"])
    check(item["label"] == row["gold_index"], f"{tag}: label")
    check(len(item["target"]) == n, f"{tag}: target len")
    check(sum(item["target"]) == 1.0, f"{tag}: target mass")
    check(item["target"][row["gold_index"]] == 1.0, f"{tag}: target index")
    check(item["markers"] == item["markers"] and len(item["markers"]) == n,
          f"{tag}: markers")
    q = {"type": "choice", "instructions": INSTRUCTIONS,
         "criteria": dict(zip(row["order"], row["options"]))}
    internal = A.Agent._to_internal(q)
    check(R.question_of(ex, INSTRUCTIONS) == internal, f"{tag}: question shape")
    ids, markers = SEQ.build_sequence(
        tok, row["state"], internal, S.MAX_LEN, S.HEAD_MAX_LEN)
    check(ids == item["ids"], f"{tag}: training sequence drift")
    check(markers == item["markers"], f"{tag}: training markers drift")
    return ex


# Variant A: REJECT at every legal position, for every K present.
reject_variants = 0
for k in sorted(by_k):
    row = by_k[k]
    caps = [c for c in row["order"] if c != REJECT_LABEL]
    for p in range(k + 1):
        new_order = caps[:p] + [REJECT_LABEL] + caps[p:]
        verify_variant(rebuild(row, new_order), f"K{k}_reject@{p}")
        reject_variants += 1

# Variant B: capability gold at every legal position (K>=1 rows).
gold_variants = 0
for k in sorted(by_k):
    if k == 0:
        continue
    row = by_k[k]
    caps = [c for c in row["order"] if c != REJECT_LABEL]
    gold = row["order"][row["gold_index"]]
    others = [c for c in caps if c != gold]
    n = k + 1
    for g in range(n):
        r = (g + 1) % n
        slots = [None] * n
        slots[g] = gold
        slots[r] = REJECT_LABEL
        it = iter(others)
        for i in range(n):
            if slots[i] is None:
                slots[i] = next(it)
        verify_variant(rebuild(row, slots), f"K{k}_gold@{g}")
        gold_variants += 1

# Variant C: every real smoke row (all 150), end-to-end target check.
for row in rows:
    ex = D.adapt_row(row)
    item = R.build_item(tok, ex, INSTRUCTIONS, S.MAX_LEN, S.HEAD_MAX_LEN)
    check(item["label"] == row["gold_index"], f"{row['id']}: real label")
    check(len(item["target"]) == len(row["order"]), f"{row['id']}: real target len")
    check(item["target"][row["gold_index"]] == 1.0, f"{row['id']}: real target")
    check(ex.gold_id == row["order"][row["gold_index"]], f"{row['id']}: gold_id")
    if row["target_kind"] == "reject":
        check(ex.gold_id == REJECT_LABEL, f"{row['id']}: reject gold")
        check(row["gold_index"] == row["reject_index"], f"{row['id']}: reject index")
    else:
        check(ex.gold_id == row["gold_capability_id"], f"{row['id']}: cap gold")

print(json.dumps({
    "rows_loaded": len(rows),
    "variant_a_reject_positions": reject_variants,
    "variant_b_gold_positions": gold_variants,
    "real_reject_positions": {f"K{k}@{p}": c for (k, p), c in sorted(real_reject_pos.items())},
    "real_capability_gold_positions": {f"K{k}@{p}": c for (k, p), c in sorted(real_gold_pos.items())},
    "checks": checks,
    "failures": len(fails),
    "failure_samples": fails[:20],
}, ensure_ascii=False, indent=2))
raise SystemExit(0 if not fails else 2)
