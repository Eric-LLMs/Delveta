#!/usr/bin/env python3
"""Post-selection export + pre-deploy checks.

Three DISTINCT artifacts (never conflated):

1. ``adapter``  — the selected LoRA adapter, copied to ``exports/best_adapter/``
                  (the training-time artifact; small; the source for everything else).
2. ``merged``   — the adapter merged into the pinned base HF model, saved to
                  ``exports/merged_hf/`` (a standalone HF checkpoint).
3. ``gguf``     — the DEPLOYMENT target (``Q5_K_M``), produced LATER with llama.cpp
                  FROM the merged HF model. This stage only WRITES THE PLAN — it
                  never quantises and never touches the live Ollama model.

The selected checkpoint comes from ``reports/selection.json`` (written by
:mod:`evaluate`, i.e. the validation task metric), or an explicit ``--checkpoint``.
Formal-500 is refused everywhere; an existing non-empty output dir is not clobbered
without ``--force``.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pe_config as cfgmod  # noqa: E402
import pe_runtime as R  # noqa: E402
import sft_data as D  # noqa: E402

ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin",
                 "README.md")


# ── selection + guardrails ────────────────────────────────────────────────────────


def selected_checkpoint(cfg: dict, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_dir():
            raise cfgmod.ConfigError(f"checkpoint not found: {p}")
        return p
    sel = cfgmod.local_path(cfg, "eval", "report_dir") / "selection.json"
    if not sel.is_file():
        raise cfgmod.ConfigError(
            f"no {sel} — run evaluate.py --all-checkpoints first, or pass --checkpoint")
    data = json.loads(sel.read_text(encoding="utf-8"))
    p = Path(data["selection"]["checkpoint"])
    if not p.is_dir():
        raise cfgmod.ConfigError(f"selected checkpoint is gone: {p}")
    return p


def _guard_target(path: Path, *, force: bool) -> None:
    D.refuse_formal500(path)
    if path.exists() and any(path.iterdir()) and not force:
        raise cfgmod.ConfigError(f"refusing to overwrite non-empty {path} (pass --force)")


def _dtype_from_str(torch, name: str):
    return {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            "float16": torch.float16, "fp16": torch.float16,
            "float32": torch.float32, "fp32": torch.float32}.get(str(name).lower(),
                                                                  torch.float32)


# ── 1. adapter ────────────────────────────────────────────────────────────────────


def export_adapter(cfg: dict, ckpt: Path, *, force: bool) -> Path:
    dst = cfgmod.local_path(cfg, "export", "output_dir") / "best_adapter"
    _guard_target(dst, force=force)
    present = [f for f in ADAPTER_FILES if (ckpt / f).is_file()]
    if "adapter_config.json" not in present:
        raise cfgmod.ConfigError(f"{ckpt} has no adapter_config.json — not a LoRA adapter")
    dst.mkdir(parents=True, exist_ok=True)
    for f in present:
        shutil.copy2(ckpt / f, dst / f)
    (dst / "EXPORT.json").write_text(json.dumps({
        "kind": "lora_adapter", "source_checkpoint": str(ckpt),
        "step": R.checkpoint_step(ckpt), "copied": present,
    }, indent=2), encoding="utf-8")
    print(f"adapter  -> {dst}  ({present})")
    return dst


# ── 2. merged HF model ────────────────────────────────────────────────────────────


def export_merged(cfg: dict, adapter: Path, *, force: bool) -> Path:
    dst = cfgmod.local_path(cfg, "export", "output_dir") / "merged_hf"
    _guard_target(dst, force=force)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = _dtype_from_str(torch, cfg["export"]["merged_dtype"])
    tokenizer = R.load_tokenizer(AutoTokenizer, cfg["model"]["base"],
                                 str(cfg["model"]["revision"]))
    base = R.load_base_model(AutoModelForCausalLM, cfg["model"]["base"],
                             str(cfg["model"]["revision"]), dtype)
    model = PeftModel.from_pretrained(base, str(adapter))
    merged = model.merge_and_unload()
    dst.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(dst), safe_serialization=True)
    tokenizer.save_pretrained(str(dst))
    (dst / "EXPORT.json").write_text(json.dumps({
        "kind": "merged_hf", "base": cfg["model"]["base"],
        "revision": str(cfg["model"]["revision"]),
        "adapter": str(adapter), "dtype": str(dtype).replace("torch.", ""),
    }, indent=2), encoding="utf-8")
    print(f"merged   -> {dst}")
    return dst


# ── 3. GGUF plan (never executed here) ────────────────────────────────────────────


def gguf_plan(cfg: dict, merged: Path | None) -> dict:
    out_dir = cfgmod.local_path(cfg, "export", "output_dir")
    quant = str(cfg["export"]["gguf"]["quant"])
    src = merged or (out_dir / "merged_hf")
    f16 = out_dir / "gguf" / "model.f16.gguf"
    final = out_dir / "gguf" / f"model.{quant}.gguf"
    return {
        "kind": "gguf_plan", "executed": False,
        "reason": "deployment quantization runs later; this stage never quantizes or "
                  "touches the live Ollama model",
        "merged_input": str(src),
        "steps": [
            f"python llama.cpp/convert_hf_to_gguf.py {src} --outfile {f16} --outtype f16",
            f"llama-quantize {f16} {final} {quant}",
        ],
        "note": f"the Ollama model serving the online extractor is NOT overwritten; "
                f"deploy {final} as a new model and update the endpoint URL explicitly.",
    }


def write_gguf_plan(cfg: dict, merged: Path | None) -> Path:
    plan = gguf_plan(cfg, merged)
    out_dir = cfgmod.local_path(cfg, "export", "output_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "gguf_plan.json"
    path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"gguf plan-> {path}  (NOT executed)")
    print("  " + "\n  ".join(plan["steps"]))
    return path


# ── pre-deploy checks ─────────────────────────────────────────────────────────────


def preflight(cfg: dict, adapter: Path) -> list[dict]:
    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    cfgfile = adapter / "adapter_config.json"
    acfg = json.loads(cfgfile.read_text(encoding="utf-8")) if cfgfile.is_file() else {}
    add("adapter_config.json present", cfgfile.is_file())
    add("adapter base matches the pinned model",
        str(acfg.get("base_model_name_or_path", "")) == cfg["model"]["base"],
        f"{acfg.get('base_model_name_or_path')!r} vs {cfg['model']['base']!r}")

    tpl_ok = False
    detail = "tokenizer not loaded (merged model not built)"
    merged = cfgmod.local_path(cfg, "export", "output_dir") / "merged_hf"
    if (merged / "tokenizer_config.json").is_file():
        try:
            import json as _json
            tk = _json.loads((merged / "tokenizer_config.json").read_text(encoding="utf-8"))
            tpl_ok = bool(tk.get("chat_template"))
            detail = "chat_template present in merged tokenizer" if tpl_ok \
                else "merged tokenizer has NO chat_template"
        except Exception as exc:  # noqa: BLE001 — report, never crash the check
            detail = f"could not read merged tokenizer: {exc!r}"
    else:
        detail = "merged_hf/ not built yet (run --which merged)"
    add("function-calling chat template survives", tpl_ok, detail)

    add("revision pinned", len(str(cfg["model"]["revision"])) >= 7,
        str(cfg["model"]["revision"]))
    return checks


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Qwen PE SFT — export + pre-deploy checks.")
    ap.add_argument("--config", default=str(cfgmod.DEFAULT_CONFIG))
    ap.add_argument("--which", default="adapter",
                    choices=["adapter", "merged", "gguf", "all"])
    ap.add_argument("--checkpoint", default=None, help="override the selected checkpoint")
    ap.add_argument("--force", action="store_true", help="overwrite a non-empty output dir")
    args = ap.parse_args(argv)

    cfg = cfgmod.load_config(args.config)
    ckpt = selected_checkpoint(cfg, args.checkpoint)
    print(f"selected checkpoint: {ckpt}\n")

    adapter = None
    if args.which in ("adapter", "all"):
        adapter = export_adapter(cfg, ckpt, force=args.force)
    else:
        adapter = cfgmod.local_path(cfg, "export", "output_dir") / "best_adapter"
        if not adapter.is_dir():
            adapter = ckpt

    merged = None
    if args.which in ("merged", "all"):
        merged = export_merged(cfg, adapter, force=args.force)

    if args.which in ("gguf", "all"):
        write_gguf_plan(cfg, merged)

    print("\n── pre-deploy checks ──")
    checks = preflight(cfg, adapter)
    for c in checks:
        print(f"  [{'PASS' if c['ok'] else 'FAIL'}] {c['check']}"
              + (f"  — {c['detail']}" if c["detail"] else ""))
    if args.which == "gguf":
        print("\nGGUF: plan written only; quantization/deployment is a LATER stage.")
        return 0
    return 0 if all(c["ok"] for c in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
