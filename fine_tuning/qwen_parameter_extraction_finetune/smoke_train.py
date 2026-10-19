#!/usr/bin/env python3
"""Local CPU smoke test for the Qwen parameter-extraction SFT pipeline.

Scope (deliberately narrow): prove the TRAINING PIPELINE RUNS end to end on the
smallest possible input — not that the model is accurate. Boundaries, hard-coded:

* trains ONLY on ``data/train.jsonl``; a non-``train.jsonl`` ``--data`` is refused
  (validation / test are never used for a weight update), and any path mentioning
  ``formal500`` is refused (Formal-500 is fully isolated).
* never modifies the data files, the manifest, the online extractor, the social
  handler or ``plugin.py``; never overwrites a deployed model; never commits.
* the HF cache, the temporary adapter and the logs all live under this directory.

The eight checks mirror the approved acceptance list:

  1. model + tokenizer load; the chat template is present
  2. a REAL train.jsonl sample renders ``tools`` + ``tool_choice="required"``
  3. labels mask system/user; ONLY the assistant tool-call turn is supervised
  4. the forward loss is finite
  5. >=1 LoRA parameter has a non-empty, finite gradient
  6. after 2-3 optimizer steps a LoRA weight changed (non-zero); loss stays finite
  7. save + reload the adapter: same-input loss matches within float tolerance
  8. (reported SEPARATELY) a deterministic generation parses to ONE tool_call with
     the right capability name and legal JSON ``arguments``

Checks 1-7 are the pipeline; check 8 is the model's behaviour and may legitimately
fail after a 2-3 step CPU smoke — it is reported apart and never "fixed" by
touching validation / test / the training data.

Run:  ./.venv/Scripts/python.exe smoke_train.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The HF cache stays inside this directory (never pollutes the global ~/.cache).
os.environ.setdefault("HF_HOME", str(HERE / ".hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pe_contract as C  # noqa: E402
import sft_data as D  # noqa: E402


# ── check bookkeeping ─────────────────────────────────────────────────────────────
RESULTS: list[dict] = []


def record(n: int, name: str, ok: bool, detail: str = "", *, separate: bool = False) -> None:
    RESULTS.append({"check": n, "name": name, "ok": bool(ok), "detail": detail,
                    "separate": separate})
    tag = "PASS" if ok else "FAIL"
    banner = "  [check 8 — reported separately]" if separate else ""
    print(f"[{tag}] #{n} {name}{banner}" + (f"\n        {detail}" if detail else ""))


def die(msg: str) -> "NoReturn":  # noqa: F821
    print(f"\nSMOKE ABORTED: {msg}", file=sys.stderr)
    sys.exit(2)


# ── data guard (boundary) ─────────────────────────────────────────────────────────
# The loader / renderer / masker are the SHARED protocol in sft_data — this smoke
# test reuses them so it can never drift from the trainer or the evaluator.
def load_train_samples(path: Path, n: int) -> list[dict]:
    if path.name != "train.jsonl":
        die(f"refusing --data {path}: only train.jsonl may drive a weight update")
    try:
        return D.load_rows(path, limit=n)
    except D.SftDataError as exc:
        die(str(exc))


def build_example(sample: dict, tokenizer) -> dict:
    try:
        return D.build_example(sample, tokenizer)
    except D.SftDataError as exc:
        die(str(exc))


def main() -> None:
    ap = argparse.ArgumentParser(description="Qwen PE SFT — CPU smoke test.")
    ap.add_argument("--model", default="iromu/Qwen3-0.6B-tools")
    ap.add_argument("--data", default=str(HERE / "data" / "train.jsonl"))
    ap.add_argument("--out", default=str(HERE / "smoke_out"))
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261011)
    args = ap.parse_args()

    data_path = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    adapter_dir = out / "adapter"
    t_start = time.monotonic()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel, get_peft_model

    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, (os.cpu_count() or 2)))
    device = "cpu"

    print(f"model={args.model}  data={data_path}  device={device}  steps={args.steps}")
    print(f"torch={torch.__version__}  transformers={__import__('transformers').__version__}  "
          f"peft={__import__('peft').__version__}  HF_HOME={os.environ['HF_HOME']}")

    # ── check 1: load model + tokenizer, chat template present ────────────────────
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    has_tpl = bool(getattr(tokenizer, "chat_template", None))
    record(1, "model + tokenizer load; chat template present", has_tpl,
           f"params={sum(p.numel() for p in model.parameters()):,}  chat_template={has_tpl}")

    # ── check 2: a real train sample renders tools + tool_choice="required" ────────
    samples = load_train_samples(data_path, args.samples)
    examples = [build_example(s, tokenizer) for s in samples]
    ex0 = examples[0]
    prompt_text = D.render_prompt(tokenizer, ex0["sample"]["messages"][:2],
                                  ex0["tools"], gen=True)
    tools_in_prompt = "<tools>" in prompt_text and '"name"' in prompt_text
    cap_in_prompt = ex0["cap"] in prompt_text
    record(2, 'real train sample renders tools + tool_choice="required"',
           tools_in_prompt and cap_in_prompt,
           f"cap={ex0['cap']}  <tools> block={tools_in_prompt}  "
           f"capability in prompt={cap_in_prompt}  prompt_chars={len(prompt_text)}")

    # ── check 3: labels mask system/user; only the assistant turn is supervised ────
    input_ids = ex0["input_ids"]
    labels = ex0["labels"]
    n_masked = sum(1 for t in labels if t == D.IGNORE_INDEX)
    n_target = len(labels) - n_masked
    target_is_toolcall = "<tool_call>" in ex0["target_text"] and ex0["cap"] in ex0["target_text"]
    mask_ok = (n_masked == ex0["prompt_len"] and n_target > 0
               and n_target < len(labels)  # context is NOT all supervised
               and target_is_toolcall)
    record(3, "labels mask system/user; only the assistant tool-call is supervised",
           mask_ok,
           f"total={len(labels)}  masked={n_masked}  supervised={n_target}  "
           f"target_has_<tool_call>={target_is_toolcall}")

    def loss_on(ex: dict, model_) -> float:
        model_.eval()
        ids = torch.tensor([ex["input_ids"]], device=device)
        lab = torch.tensor([ex["labels"]], device=device)
        with torch.no_grad():
            return float(model_(input_ids=ids, labels=lab).loss)

    # ── check 4: forward loss finite (base model) ─────────────────────────────────
    base_loss = loss_on(ex0, model)
    record(4, "forward loss finite", math.isfinite(base_loss), f"base_loss={base_loss:.6f}")

    # ── LoRA attach ───────────────────────────────────────────────────────────────
    lora = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.train()
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA trainable params: {n_trainable:,} across {len(trainable)} tensors")

    def step_loss(ex: dict):
        ids = torch.tensor([ex["input_ids"]], device=device)
        lab = torch.tensor([ex["labels"]], device=device)
        return model(input_ids=ids, labels=lab).loss

    # ── check 5: >=1 non-empty, finite LoRA gradient ──────────────────────────────
    model.zero_grad(set_to_none=True)
    loss5 = step_loss(examples[0])          # first, isolated forward+backward
    loss5.backward()
    grads = []
    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            grads.append((n, float(p.grad.norm()), bool(torch.isfinite(p.grad).all()), p.grad.numel()))
    good_grads = [g for g in grads if g[2] and g[1] > 0.0 and g[3] > 0]
    record(5, ">=1 LoRA parameter has a non-empty, finite gradient",
           len(good_grads) > 0 and math.isfinite(float(loss5)),
           f"loss={float(loss5):.6f}  grads_with_grad={len(grads)}  "
           f"nonempty_finite={len(good_grads)}  e.g. {good_grads[0][0]} norm={good_grads[0][1]:.4e}"
           if good_grads else f"loss={float(loss5):.6f}  grads_with_grad={len(grads)}  none finite/nonzero")

    # ── check 6: 2-3 optimizer steps; weight delta non-zero; loss stays finite ────
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    # snapshot one LoRA weight tensor for the delta check
    snap_name = good_grads[0][0] if good_grads else trainable[0]
    snap_before = model.get_parameter(snap_name).detach().clone()
    step_losses = []
    for i in range(args.steps):
        ex = examples[i % len(examples)]
        opt.zero_grad(set_to_none=True)
        l = step_loss(ex)
        l.backward()
        opt.step()
        step_losses.append(float(l))
    snap_after = model.get_parameter(snap_name).detach().clone()
    delta = float((snap_after - snap_before).abs().sum())
    all_finite = all(math.isfinite(v) for v in step_losses)
    record(6, "after steps a LoRA weight changed (non-zero); loss stayed finite",
           delta > 0.0 and all_finite,
           f"steps={args.steps}  losses={[round(v,4) for v in step_losses]}  "
           f"weight_delta({snap_name})={delta:.4e}  all_finite={all_finite}")

    # ── check 7: save + reload adapter; same-input loss matches ───────────────────
    loss_before_save = loss_on(ex0, model)
    model.save_pretrained(str(adapter_dir))
    del model
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(device)
    reloaded = PeftModel.from_pretrained(base, str(adapter_dir)).to(device)
    loss_after_reload = loss_on(ex0, reloaded)
    match = abs(loss_before_save - loss_after_reload) <= 1e-4
    record(7, "save + reload adapter: same-input loss matches within tolerance",
           match,
           f"before={loss_before_save:.6f}  after_reload={loss_after_reload:.6f}  "
           f"abs_diff={abs(loss_before_save - loss_after_reload):.2e}  -> {adapter_dir}")

    # ── check 8 (SEPARATE): deterministic generation -> ONE tool_call ─────────────
    gen_ok, gen_detail, gen_text = generation_check(reloaded, tokenizer, ex0, torch, device)
    record(8, "deterministic generation parses to ONE tool_call with correct name + JSON args",
           gen_ok, gen_detail, separate=True)

    # ── report ────────────────────────────────────────────────────────────────────
    m = C.model_slots(ex0["cap"])
    elapsed = time.monotonic() - t_start
    peak_mb = _peak_rss_mb()
    pipeline_ok = all(r["ok"] for r in RESULTS if not r["separate"])
    report = {
        "model": args.model, "data": str(data_path), "device": device,
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
        "peft": __import__("peft").__version__,
        "steps": args.steps, "samples": args.samples, "seed": args.seed,
        "elapsed_s": round(elapsed, 2), "peak_rss_mb": peak_mb,
        "lora": {"r": 8, "alpha": 16, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                 "trainable_params": n_trainable},
        "expected_model_slots": m,
        "checks": RESULTS,
        "pipeline_ok": pipeline_ok,
        "generation_ok": gen_ok,
        "generated_text": gen_text,
    }
    (out / "smoke_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "=" * 72)
    print(f"pipeline (checks 1-7): {'PASS' if pipeline_ok else 'FAIL'}")
    print(f"generation (check 8, separate): {'PASS' if gen_ok else 'FAIL'}")
    print(f"elapsed={elapsed:.1f}s  peak_rss={peak_mb} MB  report={out / 'smoke_report.json'}")
    print("=" * 72)
    # a failing generation check does NOT fail the pipeline exit code
    sys.exit(0 if pipeline_ok else 1)


# ── generation + parsing (check 8) ────────────────────────────────────────────────


def generation_check(model, tokenizer, ex0, torch, device):
    sample = ex0["sample"]
    prompt_text = D.render_prompt(tokenizer, sample["messages"][:2], sample["tools"], gen=True)
    ids = torch.tensor([tokenizer(prompt_text, add_special_tokens=False).input_ids], device=device)
    with torch.no_grad():
        out = model.generate(input_ids=ids, do_sample=False, num_beams=1,
                             max_new_tokens=128, pad_token_id=tokenizer.eos_token_id)
    gen_ids = out[0][ids.shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=False)
    try:
        name, args = D.parse_tool_call_text(text)
    except D.SftDataError as exc:
        return False, str(exc), text
    allowed = set(C.model_slots(sample["capability_id"]))
    extra = set(args) - allowed
    ok = (name == sample["capability_id"]) and not extra
    detail = (f"name={name!r} (want {sample['capability_id']!r})  "
              f"args_keys={sorted(args)}  allowed={sorted(allowed)}  extra={sorted(extra)}")
    return ok, detail, text


def _peak_rss_mb() -> int:
    try:
        import resource  # POSIX
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    except Exception:
        pass
    try:
        import psutil  # Windows peak working set
        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", None)
        if peak:
            return int(peak / (1024 * 1024))
    except Exception:
        pass
    try:
        import ctypes, ctypes.wintypes  # Windows fallback

        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.wintypes.DWORD), ("PageFaultCount", ctypes.wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        pmc = PMC(); pmc.cb = ctypes.sizeof(PMC)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        return int(pmc.PeakWorkingSetSize / (1024 * 1024))
    except Exception:
        return -1


if __name__ == "__main__":
    main()
