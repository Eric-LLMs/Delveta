import json, sys
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
    INSTRUCTIONS, REJECT_CARD, REJECT_LABEL = _r.INSTRUCTIONS, _r.REJECT_CARD, _r.REJECT_LABEL
    render_choice_question = _r.render_choice_question

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

snap = json.loads((V2 / "data/aligned/v2_registry_snapshot.json").read_text(encoding="utf-8"))
entries = {cid: SimpleNamespace(capability_id=e["capability_id"],
                                tool_binding=e["tool_binding"],
                                description=e["description"],
                                negatives=e["negatives"],
                                parameters=e["parameters"])
           for cid, e in snap.items()}
manifest = {t: json.loads((RUN / f"data/{t}/manifest.json").read_text(encoding="utf-8"))
            for t in ("threshold_082", "threshold_085", "threshold_088")}

fails = []
summary = {"rows": 0, "per_k": {}, "prod_identical": 0,
           "prod_order_differs": 0, "prod_checks": 0}


def ck(cond, msg):
    if not cond:
        fails.append(msg)


for tag in manifest:
    for split in ("train", "val", "test"):
        for line in (RUN / f"data/{tag}/v2_{split}.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rid = row["id"]
            ex = D.adapt_row(row)
            k = len(row["order"]) - 1
            summary["rows"] += 1
            summary["per_k"][k] = summary["per_k"].get(k, 0) + 1

            item = R.build_item(tok, ex, INSTRUCTIONS, S.MAX_LEN, S.HEAD_MAX_LEN)
            ck(item["label"] == row["gold_index"], f"{rid}: label")
            ck(len(item["target"]) == len(row["order"]), f"{rid}: target len")
            ck(sum(item["target"]) == 1.0, f"{rid}: target mass")
            ck(item["target"][row["gold_index"]] == 1.0, f"{rid}: target index")
            ck(len(item["markers"]) == len(row["order"]), f"{rid}: markers")

            q_row = {"type": "choice", "instructions": INSTRUCTIONS,
                     "criteria": dict(zip(row["order"], row["options"]))}
            internal_row = A.Agent._to_internal(q_row)
            ck(R.question_of(ex, INSTRUCTIONS) == internal_row, f"{rid}: question shape")
            ids_row, markers_row = SEQ.build_sequence(
                tok, row["state"], internal_row, S.MAX_LEN, S.HEAD_MAX_LEN)
            ck(ids_row == item["ids"], f"{rid}: training sequence drift")
            ck(markers_row == item["markers"], f"{rid}: training markers drift")

            cand = [c for c in row["order"] if c != REJECT_LABEL]
            q_prod = render_choice_question(
                [SimpleNamespace(capability_id=c) for c in cand], entries)
            summary["prod_checks"] += 1
            ck(q_prod["instructions"] == INSTRUCTIONS, f"{rid}: prod instructions")
            ck(list(q_prod["criteria"]) == cand + [REJECT_LABEL],
               f"{rid}: prod criteria order")
            ck(len(q_prod["criteria"]) == len(row["order"]), f"{rid}: prod count")
            for cid in cand:
                ck(q_prod["criteria"][cid] == row["options"][row["order"].index(cid)],
                   f"{rid}: prod card {cid}")
            ck(q_prod["criteria"][REJECT_LABEL] == REJECT_CARD, f"{rid}: prod reject card")

            ids_prod, markers_prod = SEQ.build_sequence(
                tok, row["state"], A.Agent._to_internal(q_prod),
                S.MAX_LEN, S.HEAD_MAX_LEN)
            ck(len(markers_prod) == len(row["order"]), f"{rid}: prod markers")
            if row["order"][-1] == REJECT_LABEL:
                summary["prod_identical"] += 1
                ck(ids_prod == item["ids"], f"{rid}: production order drift")
                ck(markers_prod == item["markers"], f"{rid}: production markers drift")
            else:
                summary["prod_order_differs"] += 1

summary["failures"] = len(fails)
summary["failure_samples"] = fails[:20]
print(json.dumps(summary, ensure_ascii=False, indent=2))
raise SystemExit(0 if not fails else 2)
