#!/usr/bin/env python3
"""Validation scoring + the explicit CHECKPOINT-SELECTION rule.

Scoring is per-sample generation under the SAME frozen protocol the trainer uses
(:mod:`sft_data`), so a checkpoint is judged on exactly what it was trained to do:
emit ONE native ``tool_call`` named for the pinned capability whose ``arguments``
is a flat ``{slot: value}`` map over the model-owned slots.

Reported metrics (overall + per capability):

* ``parse_rate``            — the reply yields a ``<tool_call>`` with OBJECT args;
* ``exact_match``           — name ok AND predicted args == gold args (slot-wise);
* ``field_accuracy``        — micro accuracy over the union of gold/predicted slots;
* ``required_missing_rate`` — micro rate over REQUIRED model slots absent from the
                              prediction (a value that is wrong still counts as present);
* ``illegal_slot_rate``     — fraction of samples emitting >=1 unauthorized key.

Selection is on the TASK metric (``eval.select_metric``), never train/eval loss;
ties break on fewer required-missing, then higher field accuracy, then lower step.

Splits: ``validation`` by default (the only split that may select a checkpoint).
``test`` is FROZEN and needs BOTH ``eval.allow_test: true`` and ``--allow-test``.
Formal-500 is refused everywhere (via :mod:`sft_data`).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# The repo + packages roots, so the ONE deterministic slot rule (shared with the
# runtime handler and the offline scorer) is importable here too. Pure module.
_ROOT = HERE.parents[1]
for _p in (str(_ROOT), str(_ROOT / "packages")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pe_config as cfgmod  # noqa: E402
import pe_contract as C  # noqa: E402
import pe_runtime as R  # noqa: E402
import sft_data as D  # noqa: E402

from core.application.chat.intent_funnel.cap_handler.search_scope import (  # noqa: E402
    effective_model_slots,
)

# The user message renders the raw sentence inside this tag (see the shared
# ``_user_prompt``); the per-sample slot rule needs the raw sentence, so it is
# recovered from the SFT row rather than duplicated onto every row.
_USER_SENTENCE_RE = re.compile(r"<user_sentence>(.*?)</user_sentence>", re.DOTALL)


def _raw_query(sample: dict) -> str:
    """The raw user sentence of an SFT row (the text the slot rule reads)."""
    for msg in sample.get("messages") or []:
        if msg.get("role") == "user":
            hit = _USER_SENTENCE_RE.search(str(msg.get("content") or ""))
            if hit:
                return hit.group(1)
    return ""


def _sample_slots(sample: dict) -> tuple[str, ...]:
    """The per-sample MODEL slots: the contract slots minus any slot a
    deterministic execution scope pins for this row (e.g. ``r/<sub>`` → ``query``)."""
    cap = sample["capability_id"]
    return effective_model_slots(cap, _raw_query(sample), C.model_slots(cap))


# ── canonicalization + per-sample scoring (pure, no torch) ────────────────────────


def canon_value(slot: str, value) -> str:
    """Canonical comparison form. Integer slots compare numerically (3 == \"3\" == 3.0);
    everything else compares as a stripped string."""
    if slot in C.INT_SLOTS:
        try:
            return str(int(float(value)))
        except (TypeError, ValueError):
            return str(value).strip()
    return str(value).strip()


def score_sample(cap: str, gold_args: dict, pred_name: str | None,
                 pred_args: dict, *, model_slots=None) -> dict:
    """Score one prediction against the gold. ``pred_args`` is the parsed object
    (or ``{}`` on a parse failure).

    ``model_slots`` optionally narrows the scored slot set to the PER-SAMPLE
    effective slots (a deterministic execution scope drops a normally-model-owned
    slot for one turn); ``None`` keeps the static capability slots."""
    slots = set(model_slots) if model_slots is not None else set(C.model_slots(cap))
    gold = {s: canon_value(s, gold_args[s]) for s in slots if s in gold_args}
    pred = {s: canon_value(s, pred_args[s]) for s in slots if s in pred_args}
    illegal = sorted(set(pred_args) - slots)
    name_ok = pred_name == cap
    considered = sorted(set(gold) | set(pred))
    field_hits = sum(1 for s in considered if pred.get(s) == gold.get(s))
    required = C.required_model_slots(cap)
    return {
        "capability_id": cap, "name_ok": name_ok,
        "exact_match": bool(name_ok and pred == gold),
        "fields_considered": len(considered), "field_hits": field_hits,
        "required_total": len(required),
        "required_present": sum(1 for s in required if s in pred),
        "required_correct": sum(1 for s in required if pred.get(s) == gold.get(s)),
        "illegal_keys": illegal,
    }


def _ratio(num: int, den: int) -> float:
    return (num / den) if den else 0.0


def _bucket(records: list[dict]) -> dict:
    n = len(records)
    req_total = sum(r["required_total"] for r in records)
    req_present = sum(r["required_present"] for r in records)
    req_correct = sum(r["required_correct"] for r in records)
    return {
        "samples": n,
        "exact_match": round(_ratio(sum(r["exact_match"] for r in records), n), 4),
        "field_accuracy": round(
            _ratio(sum(r["field_hits"] for r in records),
                   sum(r["fields_considered"] for r in records)), 4),
        "required_missing_rate": round(1.0 - _ratio(req_present, req_total), 4),
        "required_accuracy": round(_ratio(req_correct, req_total), 4),
        "illegal_slot_rate": round(
            _ratio(sum(1 for r in records if r["illegal_keys"]), n), 4),
        "illegal_keys_total": sum(len(r["illegal_keys"]) for r in records),
    }


def aggregate(results: list[dict]) -> dict:
    """``results`` is a list of ``{"record": <score_sample>, "parse_ok": bool}``."""
    n = len(results)
    records = [r["record"] for r in results]
    overall = _bucket(records)
    overall["parse_rate"] = round(
        _ratio(sum(1 for r in results if r["parse_ok"]), n), 4)
    by_cap: dict[str, dict] = {}
    for cap in sorted({r["capability_id"] for r in records}):
        sub = [r for r in results if r["record"]["capability_id"] == cap]
        b = _bucket([r["record"] for r in sub])
        b["parse_rate"] = round(_ratio(sum(1 for r in sub if r["parse_ok"]), len(sub)), 4)
        by_cap[cap] = b
    return {"overall": overall, "by_capability": by_cap}


# ── the explicit selection rule (pure) ────────────────────────────────────────────


def select_best(candidates: list[dict], metric: str, mode: str) -> dict:
    """Pick the checkpoint to export. ``candidates`` are
    ``{"checkpoint", "step", "metrics": <aggregate()>}``. Primary = ``metric`` on the
    overall bucket (higher is better for ``max``); ties break on lower
    required-missing, then higher field accuracy, then lower step."""
    if not candidates:
        raise cfgmod.ConfigError("no candidates to select from")
    if metric not in {"exact_match", "field_accuracy", "required_accuracy"}:
        raise cfgmod.ConfigError(f"unknown select_metric {metric!r}")
    if mode not in {"max", "min"}:
        raise cfgmod.ConfigError(f"unknown select_mode {mode!r}")

    def primary(c: dict) -> float:
        return c["metrics"]["overall"].get(metric, 0.0)

    sign = -1.0 if mode == "max" else 1.0
    return sorted(candidates, key=lambda c: (
        sign * primary(c),
        c["metrics"]["overall"]["required_missing_rate"],
        -c["metrics"]["overall"]["field_accuracy"],
        c["step"],
    ))[0]


# ── torch-side generation ─────────────────────────────────────────────────────────


def _prompts(tokenizer, rows: list[dict]) -> list[str]:
    return [D.render_prompt(tokenizer, r["messages"][:2], r["tools"], gen=True)
            for r in rows]


def generate(model, tokenizer, rows: list[dict], *, torch, device: str,
             max_new_tokens: int, batch_size: int) -> list[str]:
    tokenizer.padding_side = "left"
    texts: list[str] = []
    for i in range(0, len(rows), batch_size):
        chunk = _prompts(tokenizer, rows[i:i + batch_size])
        enc = tokenizer(chunk, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model.generate(**enc, do_sample=False, num_beams=1,
                                 max_new_tokens=max_new_tokens,
                                 pad_token_id=tokenizer.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        texts.extend(tokenizer.batch_decode(gen, skip_special_tokens=False))
    return texts


def _score_split(rows: list[dict], texts: list[str]) -> list[dict]:
    out = []
    for sample, text in zip(rows, texts):
        try:
            name, args = D.parse_tool_call_text(text)
            parse_ok = True
        except D.SftDataError:
            name, args, parse_ok = None, {}, False
        out.append({"record": score_sample(sample["capability_id"],
                                           D.gold_target(sample), name, args,
                                           model_slots=_sample_slots(sample)),
                    "parse_ok": parse_ok})
    return out


def run_eval(cfg: dict, *, split: str, allow_test: bool, checkpoint: Path | None,
             base_only: bool, limit: int | None, write: bool) -> dict:
    if split == D.TEST_SPLIT and not (allow_test and cfg["eval"].get("allow_test")):
        raise cfgmod.ConfigError(
            "refusing to score the FROZEN test split: needs --allow-test AND "
            "eval.allow_test: true (test never trains or selects)")
    if split not in (D.VAL_SPLIT, D.TEST_SPLIT):
        raise cfgmod.ConfigError(f"only {D.VAL_SPLIT}/{D.TEST_SPLIT} may be scored")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    data_dir = cfgmod.local_path(cfg, "eval", "data_dir")
    rows = D.load_split(data_dir, split, limit=limit)
    device = R.pick_device(torch)
    dtype = R.resolve_dtype(torch, cfg, device)
    tokenizer = R.load_tokenizer(AutoTokenizer, cfg["model"]["base"],
                                 str(cfg["model"]["revision"]))
    model = R.load_base_model(AutoModelForCausalLM, cfg["model"]["base"],
                              str(cfg["model"]["revision"]), dtype)
    label = "base"
    if checkpoint is not None:
        model = PeftModel.from_pretrained(model, str(checkpoint))
        label = Path(checkpoint).name
    model.eval().to(device)

    texts = generate(model, tokenizer, rows, torch=torch, device=device,
                     max_new_tokens=int(cfg["eval"]["max_new_tokens"]),
                     batch_size=int(cfg["eval"]["batch_size"]))
    results = _score_split(rows, texts)
    metrics = aggregate(results)
    report = {
        "split": split, "checkpoint": label, "base_only": base_only,
        "model": cfg["model"]["base"], "revision": str(cfg["model"]["revision"]),
        "n": len(rows), "metrics": metrics,
        "selection_rule": {"metric": cfg["eval"]["select_metric"],
                           "mode": cfg["eval"]["select_mode"]},
    }
    print(f"[{label}] {split} n={len(rows)}  "
          f"exact_match={metrics['overall']['exact_match']}  "
          f"parse_rate={metrics['overall']['parse_rate']}  "
          f"required_missing={metrics['overall']['required_missing_rate']}  "
          f"illegal={metrics['overall']['illegal_slot_rate']}")
    if write:
        _write_report(cfg, report, label)
    return report


def _write_report(cfg: dict, report: dict, label: str) -> Path:
    report_dir = cfgmod.local_path(cfg, "eval", "report_dir")
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"eval_{report['split']}_{label}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(report, report_dir / f"eval_{report['split']}_{label}.md")
    return report_dir


def _write_markdown(report: dict, path: Path) -> None:
    m = report["metrics"]
    o = m["overall"]
    lines = [
        f"# Evaluation — {report['split']} / {report['checkpoint']}", "",
        f"- model: `{report['model']}@{report['revision']}`",
        f"- samples: {report['n']}",
        f"- selection rule: `{report['selection_rule']['metric']}` "
        f"({report['selection_rule']['mode']})", "",
        "| scope | samples | exact_match | field_acc | parse_rate | "
        "required_missing | illegal_slot |",
        "|---|---|---|---|---|---|---|",
        f"| overall | {o['samples']} | {o['exact_match']} | {o['field_accuracy']} | "
        f"{o['parse_rate']} | {o['required_missing_rate']} | {o['illegal_slot_rate']} |",
    ]
    for cap, b in m["by_capability"].items():
        lines.append(
            f"| {cap} | {b['samples']} | {b['exact_match']} | {b['field_accuracy']} | "
            f"{b['parse_rate']} | {b['required_missing_rate']} | {b['illegal_slot_rate']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_selection(cfg: dict, *, split: str, allow_test: bool, limit: int | None) -> dict:
    """Evaluate every checkpoint under the train output dir, then select the best."""
    cands = R.checkpoint_dirs(R.train_output_dir(cfg))
    if not cands:
        raise cfgmod.ConfigError("no checkpoints to select from (train a run first)")
    scored = []
    for ckpt in cands:
        rep = run_eval(cfg, split=split, allow_test=allow_test, checkpoint=ckpt,
                       base_only=False, limit=limit, write=True)
        scored.append({"checkpoint": str(ckpt), "step": R.checkpoint_step(ckpt),
                       "metrics": rep["metrics"]})
    best = select_best(scored, cfg["eval"]["select_metric"], cfg["eval"]["select_mode"])
    out = {
        "selection": {"checkpoint": best["checkpoint"], "step": best["step"],
                      "metric": cfg["eval"]["select_metric"],
                      "mode": cfg["eval"]["select_mode"],
                      "tiebreak": ["required_missing_rate", "field_accuracy", "step"]},
        "candidates": [{"checkpoint": c["checkpoint"], "step": c["step"],
                        "overall": c["metrics"]["overall"]} for c in scored],
    }
    report_dir = cfgmod.local_path(cfg, "eval", "report_dir")
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "selection.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSELECTED {best['checkpoint']} (step {best['step']})  -> "
          f"{report_dir / 'selection.json'}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Qwen PE SFT — validation scoring + selection.")
    ap.add_argument("--config", default=str(cfgmod.DEFAULT_CONFIG))
    ap.add_argument("--split", default=D.VAL_SPLIT, choices=list((D.VAL_SPLIT, D.TEST_SPLIT)))
    ap.add_argument("--allow-test", action="store_true",
                    help="required (with eval.allow_test) to score the frozen test split")
    ap.add_argument("--checkpoint", default=None, help="an adapter dir to score")
    ap.add_argument("--base-only", action="store_true",
                    help="score the base model with no adapter (pre-training sanity)")
    ap.add_argument("--all-checkpoints", action="store_true",
                    help="score every checkpoint and write selection.json")
    ap.add_argument("--limit", type=int, default=None, help="cap scored rows (debug)")
    ap.add_argument("--no-write", action="store_true", help="do not write a report")
    args = ap.parse_args(argv)

    cfg = cfgmod.load_config(args.config)
    if args.all_checkpoints:
        run_selection(cfg, split=args.split, allow_test=args.allow_test, limit=args.limit)
        return 0
    ckpt = Path(args.checkpoint) if args.checkpoint else None
    run_eval(cfg, split=args.split, allow_test=args.allow_test, checkpoint=ckpt,
             base_only=(ckpt is None), limit=args.limit, write=not args.no_write)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
