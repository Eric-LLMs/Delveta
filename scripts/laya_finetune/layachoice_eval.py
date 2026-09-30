#!/usr/bin/env python3
"""Stage 3 - zero-shot baseline, deterministic epoch selection, final benchmark.

Three arms, each writing the COMPLETE per-row result, never only a summary:

  --baseline-zero-shot  the pinned multilingual checkpoint, its own temperature
                        ([1,1,1]), the frozen Final Test v3 (900 rows), no calibration.
                        This is the main comparison arm; BGE-M3 is not a substitute.
  --select              every `epoch-N` checkpoint under the run directory, each scored
                        on the frozen Validation v3 (150 rows). Purely deterministic:
                        best_epoch = argmax(val_top1), ties resolved to the EARLIER
                        epoch. Train loss is recorded and never used to pick.
  --benchmark           the best checkpoint on the frozen Final Test v3 (900 rows),
                        with the fitted temperature. Always exits 0 - the verdict is in
                        the report, and the Final Test never gates anything upstream.

Top-1 accuracy is invariant under any positive scalar temperature (softmax is monotone
in logits/T), so `--select` is temperature-independent; the temperature only moves the
probabilities, the confidence and hence the ECE.

Usage:
    python layachoice_eval.py --baseline-zero-shot --out out/baseline
    python layachoice_eval.py --select --run out --out out/select
    python layachoice_eval.py --benchmark --ckpt out/epoch-3 --out out/benchmark \
        --baseline out/baseline/report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import laya.common as C

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import layachoice_finetune as F  # noqa: E402

SMOKE_ROWS = 8


# ── metrics ─────────────────────────────────────────────────────────────────
def _rate(n: int, d: int) -> float:
    return round(n / d, 6) if d else 0.0


def _group(rows: list[dict], key) -> dict:
    out = {}
    for name in sorted({key(r) for r in rows}):
        sel = [r for r in rows if key(r) == name]
        out[name] = {
            "n": len(sel), "top1": _rate(sum(r["hit"] for r in sel), len(sel)),
            "pred_d1": _rate(sum(r["pred_role"] == "d1" for r in sel), len(sel)),
            "pred_d2": _rate(sum(r["pred_role"] == "d2" for r in sel), len(sel)),
        }
    return out


def summarise(rows: list[dict], *, arm: str, split: str, temperature, card_view: str,
              extra: dict | None = None) -> dict:
    n = len(rows)
    hits = sum(r["hit"] for r in rows)
    top1 = _rate(hits, n)
    conf = np.array([r["confidence"] for r in rows], dtype=float)
    correct = np.array([1.0 if r["hit"] else 0.0 for r in rows], dtype=float)
    confusions = Counter((r["gold_cap"], r["pred_cap"]) for r in rows if not r["hit"])
    role = Counter(r["pred_role"] for r in rows)
    report = {
        "arm": arm, "split": split, "rows": n,
        "card_view": card_view, "option_max_tokens": F.OPTION_MAX_TOKENS,
        "temperature": list(temperature) if isinstance(temperature, (list, tuple))
        else float(temperature),
        "top1": top1,
        "hard_negative_rate": round(1.0 - top1, 6),
        "pred_role": {"gold": role.get("gold", 0), "d1": role.get("d1", 0),
                      "d2": role.get("d2", 0)},
        "gold_to_d1": _rate(role.get("d1", 0), n),
        "gold_to_d2": _rate(role.get("d2", 0), n),
        "ece": round(float(C.ece_score(conf, correct)), 6),
        "mean_confidence": round(float(conf.mean()), 6) if n else 0.0,
        "mean_gold_prob": round(float(np.mean([r["gold_prob"] for r in rows])), 6) if n else 0.0,
        "by_language": _group(rows, lambda r: r["lang"]),
        "by_capability": _group(rows, lambda r: r["gold_cap"]),
        "top_confusions": [[g, p, c] for (g, p), c in confusions.most_common(10)],
    }
    if extra:
        report.update(extra)
    return report


def write_arm(out_dir: Path, name: str, rows: list[dict], report: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{name}.rows.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out_dir / f"{name}.report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out_dir / f"{name}.report.json"


def print_summary(report: dict) -> None:
    print(f"  rows={report['rows']} top1={report['top1']:.4f} "
          f"hardset={report['hard_negative_rate']:.4f} ece={report['ece']:.4f} "
          f"gold->d1={report['gold_to_d1']:.4f} gold->d2={report['gold_to_d2']:.4f}")
    for lang, d in report["by_language"].items():
        print(f"  lang {lang}: n={d['n']} top1={d['top1']:.4f}")


# ── arms ────────────────────────────────────────────────────────────────────
def arm_baseline(args) -> None:
    manifest = F.read_manifest(Path(args.data))
    instructions = manifest.get("instructions", F.DEFAULT_INSTRUCTIONS)
    model_dir = F.base_model_dir()
    cfg = F.read_cfg(model_dir, train=False)
    F.install()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.smoke else "cpu")
    tok = F.load_tokenizer(model_dir)
    model, _ = F.model_from_checkpoint(model_dir, model_dir, device)
    temperature = list(cfg.get("temperature", [1.0, 1.0, 1.0]))
    rows = F.load_bundle("test", Path(args.data), limit=SMOKE_ROWS if args.smoke else None)
    decided = F.decide(model, tok, rows, cfg, instructions, temperature, device)
    report = summarise(decided, arm="baseline-zero-shot", split="test",
                       temperature=temperature, card_view=manifest["formal_card_view"],
                       extra={"base_checkpoint": {"repo_id": F.BASE_REPO,
                                                  "revision": F.BASE_REVISION,
                                                  "subfolder": F.BASE_SUBFOLDER},
                              "calibrated": False, "smoke": args.smoke})
    print("[baseline-zero-shot] zero-shot, temperature from the checkpoint, no calibration")
    print_summary(report)
    print("  ->", write_arm(Path(args.out), "baseline", decided, report))


def arm_select(args) -> None:
    manifest = F.read_manifest(Path(args.data))
    instructions = manifest.get("instructions", F.DEFAULT_INSTRUCTIONS)
    run = Path(args.run)
    epochs = sorted((p for p in run.glob("epoch-*") if p.is_dir()),
                    key=lambda p: int(p.name.split("-")[1]))
    if not epochs:
        raise SystemExit(f"no epoch-* checkpoint directories under {run}")
    model_dir = F.base_model_dir()
    cfg = F.read_cfg(model_dir, train=False)
    F.install()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.smoke else "cpu")
    tok = F.load_tokenizer(model_dir)
    rows = F.load_bundle("val", Path(args.data), limit=SMOKE_ROWS if args.smoke else None)
    out_dir = Path(args.out)
    per_epoch = []
    for ckpt in epochs:
        model, _ = F.model_from_checkpoint(ckpt, model_dir, device)
        decided = F.decide(model, tok, rows, cfg, instructions, [1.0, 1.0, 1.0], device)
        meta_file = ckpt / "train_meta.json"
        train_meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
        report = summarise(decided, arm=f"validation-{ckpt.name}", split="val",
                           temperature=[1.0, 1.0, 1.0], card_view=manifest["formal_card_view"],
                           extra={"checkpoint": str(ckpt),
                                  "train_loss": train_meta.get("mean_loss"),
                                  "smoke": args.smoke})
        write_arm(out_dir, f"{ckpt.name}.val", decided, report)
        per_epoch.append({"epoch": int(ckpt.name.split("-")[1]), "ckpt": str(ckpt),
                          "val_top1": report["top1"], "val_ece": report["ece"],
                          "val_rows": report["rows"],
                          "val_report": str(out_dir / f"{ckpt.name}.val.report.json"),
                          "train_loss": train_meta.get("mean_loss")})
        print(f"[select] {ckpt.name}: val_top1={report['top1']:.4f} "
              f"train_loss={train_meta.get('mean_loss')}")
    # Deterministic: highest val top-1, ties to the earlier epoch. Train loss is only recorded.
    best = min(per_epoch, key=lambda e: (-e["val_top1"], e["epoch"]))
    selection = {
        "rule": "best_epoch = argmax(validation_top1); ties resolved to the earlier epoch; "
                "train loss is recorded and never used to pick",
        "metric": "top1 accuracy on the frozen Validation v3, card view "
                  f"{manifest['formal_card_view']}",
        "temperature_note": "top-1 is invariant under the scalar temperature, so selection "
                            "does not depend on it",
        "per_epoch": per_epoch,
        "best_epoch": best["epoch"], "best_ckpt": best["ckpt"],
        "best_val_top1": best["val_top1"], "smoke": args.smoke,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[select] best_epoch={best['epoch']} (val_top1={best['val_top1']:.4f}) -> "
          f"{out_dir / 'selection.json'}")


def arm_benchmark(args) -> None:
    manifest = F.read_manifest(Path(args.data))
    instructions = manifest.get("instructions", F.DEFAULT_INSTRUCTIONS)
    ckpt = Path(args.ckpt)
    model_dir = F.base_model_dir()
    cfg = F.read_cfg(model_dir, train=False)
    F.install()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.smoke else "cpu")
    tok = F.load_tokenizer(model_dir)
    model, _ = F.model_from_checkpoint(ckpt, model_dir, device)

    temp_file = Path(args.temperature_file) if args.temperature_file \
        else ckpt.parent / "calibration" / "temperature.json"
    calibrated = temp_file.exists()
    if calibrated:
        temperature = list(json.loads(temp_file.read_text(encoding="utf-8"))["temperature"])
    else:
        temperature = [1.0, 1.0, 1.0]
        print(f"[benchmark] no fitted temperature at {temp_file}; using {temperature}")

    rows = F.load_bundle("test", Path(args.data), limit=SMOKE_ROWS if args.smoke else None)
    decided = F.decide(model, tok, rows, cfg, instructions, temperature, device)
    extra = {"checkpoint": str(ckpt), "calibrated": calibrated,
             "temperature_file": str(temp_file) if calibrated else None,
             "smoke": args.smoke}
    if args.baseline:
        base = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        top1 = _rate(sum(r["hit"] for r in decided), len(decided))
        comparison = {
            "baseline_report": str(args.baseline),
            "baseline_top1": base["top1"],
            "top1": top1,
            "delta": round(top1 - base["top1"], 6),
            "beats_baseline": top1 > base["top1"],
        }
        if base["rows"] != len(decided):
            comparison["row_count_mismatch"] = [base["rows"], len(decided)]
        extra["baseline_comparison"] = comparison
    report = summarise(decided, arm="final-test", split="test", temperature=temperature,
                       card_view=manifest["formal_card_view"], extra=extra)
    print("[benchmark] best checkpoint on the frozen Final Test v3")
    print_summary(report)
    if "baseline_comparison" in extra:
        bc = extra["baseline_comparison"]
        verdict = "BEATS" if bc["beats_baseline"] else "does NOT beat"
        print(f"  zero-shot baseline top1={bc['baseline_top1']:.4f} "
              f"delta={bc['delta']:+.4f} -> {verdict} the zero-shot objective")
    print("  ->", write_arm(Path(args.out), "final_test", decided, report))


# ── CLI ─────────────────────────────────────────────────────────────────────
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    arm = ap.add_mutually_exclusive_group(required=True)
    arm.add_argument("--baseline-zero-shot", action="store_true")
    arm.add_argument("--select", action="store_true")
    arm.add_argument("--benchmark", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="8 rows per split, CPU")
    ap.add_argument("--data", default=str(F.DATA))
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", default=None, help="--select: the training run directory")
    ap.add_argument("--ckpt", default=None, help="--benchmark: the checkpoint directory")
    ap.add_argument("--temperature-file", default=None)
    ap.add_argument("--baseline", default=None,
                    help="--benchmark: the baseline report.json to compare against")
    args = ap.parse_args(argv)
    if args.select and not args.run:
        ap.error("--select requires --run")
    if args.benchmark and not args.ckpt:
        ap.error("--benchmark requires --ckpt")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.baseline_zero_shot:
        arm_baseline(args)
    elif args.select:
        arm_select(args)
    else:
        arm_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
