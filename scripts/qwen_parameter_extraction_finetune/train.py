#!/usr/bin/env python3
"""Formal LoRA SFT entry for the Qwen3-0.6B parameter-extraction model.

Task (unchanged across build/train/eval/export): fill the argument slots for ONE
already-chosen capability. Capability selection happens upstream; this model never
chooses, re-selects or questions the tool.

The training protocol lives in :mod:`sft_data` (the ONE loader / renderer /
masker), so this entry cannot drift from the evaluator or the CPU smoke test:

* examples come from ``data/train.jsonl`` ONLY (a non-``train.jsonl`` ``--data``
  and any path naming ``formal500`` are refused);
* the system + user prompt is masked; ONLY the assistant tool-call turn is
  supervised;
* the base checkpoint is the pinned HF revision; LoRA is attached with the config
  in ``train.lora``.

Modes (all three run WITHOUT starting a real training loop unless you omit them):

    --check     stdlib-only: config + data contract + frozen-hash gate. No torch.
    --dry-run   full wiring (tokenizer + model + LoRA + Trainer + dataset), then
                STOP before ``trainer.train()`` — proves the run is launchable.
    (default)   the real run; writes checkpoints under ``train.output_dir``.

Checkpoint SELECTION is NOT here and is NOT train loss — it is the validation task
metric computed by :mod:`evaluate`. This entry only saves checkpoints and supports
full-state resume (model/adapter, optimizer, scheduler, global step, RNG).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
# ``search_scope`` (the ONE slot-narrowing rule) is a stdlib-only production
# module; the trainer reuses it so the legal model face cannot drift from the
# builder / handler. It pulls no torch, so ``--check`` stays torch-free.
for _p in (str(REPO), str(REPO / "packages")):
    if _p not in sys.path:
        sys.path.append(_p)

# pe_runtime sets HF_HOME (the local cache) and TOKENIZERS_PARALLELISM on import.
import pe_config as cfgmod  # noqa: E402
import pe_contract as C  # noqa: E402
import pe_runtime as R  # noqa: E402
import sft_data as D  # noqa: E402

TRAIN_FILE = "train.jsonl"
VAL_FILE = "validation.jsonl"


# ── frozen-bundle + contract gate (stdlib only) ───────────────────────────────────


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_contract(cfg: dict, *, verify_hash: bool = True) -> dict:
    """The trainer's own pre-flight gate: the data it will read exists, matches the
    frozen manifest hash, and every row obeys the shared contract. This is a cheap
    structural check, NOT the authority — :mod:`validate_dataset` re-renders and is
    the authoritative verifier."""
    data_dir = cfgmod.local_path(cfg, "train", "data_dir")
    train_path = data_dir / TRAIN_FILE
    val_path = data_dir / VAL_FILE
    D.refuse_formal500(train_path)
    D.refuse_formal500(val_path)
    if not train_path.is_file():
        raise D.SftDataError(f"missing training file: {train_path}")

    manifest_path = cfgmod.local_path(cfg, "output", "manifests_dir") / "manifest.json"
    # FAIL-CLOSED: when the hash gate is on, a missing / unreadable / malformed
    # manifest, a missing recorded hash, or a mismatch is a hard error. A missing
    # manifest is NEVER silently treated as ``{}`` (that is what made the gate a
    # no-op). Only the explicit ``--no-hash-check`` escape hatch skips this.
    manifest: dict = {}
    if verify_hash:
        if not manifest_path.is_file():
            raise D.SftDataError(
                f"missing manifest {manifest_path} (hash gate is fail-closed; "
                f"pass --no-hash-check only to override)")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise D.SftDataError(
                f"unreadable/invalid manifest {manifest_path}: {exc!r}") from exc
        if not isinstance(manifest, dict):
            raise D.SftDataError(f"manifest {manifest_path} is not a JSON object")
    file_sha = (manifest.get("file_sha256") or {}) if verify_hash else {}
    got = _sha256(train_path)
    recorded = file_sha.get(TRAIN_FILE)
    if verify_hash:
        if not recorded:
            raise D.SftDataError(f"manifest {manifest_path} has no file_sha256[{TRAIN_FILE!r}]")
        if recorded != got:
            raise D.SftDataError(
                f"{TRAIN_FILE} hash {got} != frozen manifest {recorded}; refusing to train "
                f"on a mutated bundle")

    # The training entry also READS validation.jsonl (loss monitoring), so its
    # hash is gated too — a mutated validation file must not slip through.
    val_got = _sha256(val_path) if val_path.is_file() else None
    val_recorded = file_sha.get(VAL_FILE)
    if verify_hash and val_got is not None:
        if not val_recorded:
            raise D.SftDataError(f"manifest {manifest_path} has no file_sha256[{VAL_FILE!r}]")
        if val_recorded != val_got:
            raise D.SftDataError(
                f"{VAL_FILE} hash {val_got} != frozen manifest {val_recorded}")

    rows = D.load_rows(train_path)
    _validate_rows(rows)
    n_val = len(D.load_rows(val_path)) if val_path.is_file() else 0
    by_cap: dict[str, int] = {}
    for r in rows:
        by_cap[r["capability_id"]] = by_cap.get(r["capability_id"], 0) + 1
    val_ok = val_got is None or val_recorded == val_got
    return {
        "train_file": str(train_path),
        "train_sha256": got,
        "manifest_sha256": recorded,
        "validation_sha256": val_got,
        "hash_verified": bool(verify_hash and recorded == got and val_ok),
        "train_rows": len(rows),
        "validation_rows": n_val,
        "by_capability": dict(sorted(by_cap.items())),
    }


_QUERY_RE = re.compile(r"<user_sentence>(.*)</user_sentence>", re.DOTALL)


def _extract_query(user_content: str) -> str | None:
    """The verbatim user sentence embedded in the serialized user message."""
    m = _QUERY_RE.search(user_content or "")
    return m.group(1) if m else None


_NARROWING_RULE = None


def _narrowing_rule():
    """The ONE slot-narrowing rule (:func:`search_scope.effective_model_slots`),
    reused — never copied, and resolved even when the heavy ``core.…`` package chain
    is not importable (the isolated training venv lacks a parent-``__init__`` dep).
    ``search_scope`` is stdlib-only, so it can be loaded straight from its file."""
    global _NARROWING_RULE
    if _NARROWING_RULE is None:
        try:
            from core.application.chat.intent_funnel.cap_handler.search_scope import (
                effective_model_slots,
            )
        except Exception:
            import importlib.util
            path = (REPO / "packages" / "core" / "application" / "chat"
                    / "intent_funnel" / "cap_handler" / "search_scope.py")
            spec = importlib.util.spec_from_file_location("pe_search_scope", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            effective_model_slots = mod.effective_model_slots
        _NARROWING_RULE = effective_model_slots
    return _NARROWING_RULE


def _expected_slots(cap: str, query: str) -> list[str]:
    """The per-sample MODEL-owned slots for this turn — the SAME rule the builder
    and the runtime handler use (:func:`search_scope.effective_model_slots`). A
    deterministic ``r/<name>`` scope drops the pinned ``platform``; every other
    turn keeps the static capability slots."""
    return list(_narrowing_rule()(cap, query, C.model_slots(cap)))


def _check_meta(i: int, cap: str, args: dict, meta: dict) -> None:
    """The DECLARED-INTENT gate plus metadata-consistency checks.

    A required model slot may be absent from the target IFF it is declared in
    ``meta.missing_required`` or ``meta.expected_outcome == "clarify"``. Declared
    metadata must be coherent: every name is a real required slot, no duplicates,
    and each is actually absent. ``execute`` requires a COMPLETE target and may
    not declare any missing slot (a contradiction)."""
    missing = [s for s in C.required_model_slots(cap) if s not in args]
    req = set(C.required_model_slots(cap))
    outcome = meta.get("expected_outcome")
    declared = list(meta.get("missing_required") or [])
    if outcome not in (None, "execute", "clarify"):
        raise D.SftDataError(f"row {i} ({cap}): unknown expected_outcome {outcome!r}")
    if len(declared) != len(set(declared)):
        raise D.SftDataError(f"row {i} ({cap}): meta.missing_required has duplicates {declared}")
    for s in declared:
        if s not in req:
            raise D.SftDataError(
                f"row {i} ({cap}): missing_required names non-required slot {s!r}")
        if s not in missing:
            raise D.SftDataError(
                f"row {i} ({cap}): missing_required declares {s!r} but it is present")
    if outcome != "clarify" and declared:
        raise D.SftDataError(
            f"row {i} ({cap}): missing_required declared but expected_outcome={outcome!r}")
    declared_set = set(declared)
    if outcome == "clarify":
        declared_set |= set(missing)
    undeclared = [s for s in missing if s not in declared_set]
    if undeclared:
        raise D.SftDataError(
            f"row {i} ({cap}): target missing required slot(s) {undeclared} (undeclared)")


def _validate_rows(rows: list[dict]) -> None:
    """Every row must be an assistant tool-call turn for a known capability, whose
    ``function.parameters`` is the SAME model face the builder produced (static, or
    the one legal ``cap-social-search`` narrowing), whose ``function.arguments`` is
    a JSON object over that sample's model slots with type/length-valid values, and
    whose required slots are present unless correctly DECLARED missing. Shape only —
    no model call, no value repair."""
    for i, r in enumerate(rows):
        cap = r.get("capability_id")
        if cap not in C.SCHEMA:
            raise D.SftDataError(f"row {i}: unknown capability {cap!r} (model caps only)")
        msgs = r.get("messages") or []
        if len(msgs) != 3 or [m.get("role") for m in msgs] != ["system", "user", "assistant"]:
            raise D.SftDataError(f"row {i} ({cap}): messages must be system/user/assistant")
        query = _extract_query(msgs[1].get("content"))
        if query is None or not query.strip():
            raise D.SftDataError(f"row {i} ({cap}): user message has no <user_sentence>")
        tools = r.get("tools") or []
        if len(tools) != 1 or tools[0]["function"]["name"] != cap:
            raise D.SftDataError(f"row {i} ({cap}): tools must pin exactly one function named {cap}")
        expected_face = C.model_schema(cap, _expected_slots(cap, query))
        face = tools[0]["function"].get("parameters")
        if face != expected_face:
            raise D.SftDataError(
                f"row {i} ({cap}): function.parameters != the expected model face "
                f"(static or the one legal social narrowing)")
        calls = msgs[-1].get("tool_calls") or []
        if len(calls) != 1 or calls[0]["function"]["name"] != cap:
            raise D.SftDataError(f"row {i} ({cap}): assistant must carry one tool_call named {cap}")
        args = D.gold_target(r)
        if not isinstance(args, dict):
            raise D.SftDataError(f"row {i} ({cap}): target arguments must be a JSON object")
        allowed = set(expected_face["properties"])
        extra = set(args) - allowed
        if extra:
            raise D.SftDataError(f"row {i} ({cap}): target has unauthorized slot(s) {sorted(extra)}")
        if "arguments" in args and "arguments" not in allowed:
            raise D.SftDataError(f"row {i} ({cap}): nested {{'arguments': ...}} wrapper is forbidden")
        params = C.SCHEMA[cap]["params"]
        for slot, val in args.items():
            spec = params[slot]
            if spec["type"] == "integer":
                if not isinstance(val, int) or isinstance(val, bool):
                    raise D.SftDataError(f"row {i} ({cap}): {slot} must be an integer, got {val!r}")
            else:
                if not isinstance(val, str) or not val.strip():
                    raise D.SftDataError(
                        f"row {i} ({cap}): {slot} must be a non-empty string, got {val!r}")
                if spec.get("max_len") and len(val) > spec["max_len"]:
                    raise D.SftDataError(
                        f"row {i} ({cap}): {slot} exceeds max_len {spec['max_len']}")
        _check_meta(i, cap, args, r.get("meta") or {})


# ── pure planning helpers (unit-testable, no torch) ───────────────────────────────


def plan_steps(cfg: dict, n_train: int) -> dict:
    """Estimate the optimizer schedule. ``transformers`` no longer accepts
    ``warmup_ratio``, so the ratio is converted to ``warmup_steps`` HERE."""
    t = cfg["train"]
    eff = max(1, int(t["per_device_train_batch_size"]) * int(t["gradient_accumulation_steps"]))
    epochs = float(t["num_train_epochs"])
    per_epoch = math.ceil(n_train / eff)
    total = math.ceil(per_epoch * epochs)
    warmup = math.ceil(float(t["warmup_ratio"]) * total)
    return {
        "n_train": int(n_train), "effective_batch": eff, "epochs": epochs,
        "steps_per_epoch": per_epoch, "total_optimizer_steps": total,
        "warmup_steps": warmup,
    }


def resolve_resume(cfg: dict, value: str | None) -> tuple[Path | None, str]:
    """Map ``train.resume_from_checkpoint`` (or the CLI) to a checkpoint path.

    ``None`` -> a fresh run; ``"auto"`` -> the highest-step ``checkpoint-*`` under
    the output dir; anything else -> that path (must exist)."""
    out_dir = R.train_output_dir(cfg)
    if value is None or str(value).strip() == "":
        return None, "fresh"
    value = str(value).strip()
    if value == "auto":
        cands = R.checkpoint_dirs(out_dir)
        if not cands:
            raise cfgmod.ConfigError(f"resume=auto but no checkpoint-* under {out_dir}")
        return cands[-1], "auto(latest)"
    p = Path(value)
    if not p.is_dir():
        raise cfgmod.ConfigError(f"resume checkpoint not found: {p}")
    return p, "explicit"


# ── torch-side wiring (lazy import; only reached by --dry-run / a real run) ────────


def _build(model_cls, tok_cls, lora_cls, get_peft, cfg: dict, device: str, torch):
    base = cfg["model"]["base"]
    revision = str(cfg["model"]["revision"])
    dtype = R.resolve_dtype(torch, cfg, device)
    tokenizer = R.load_tokenizer(tok_cls, base, revision)
    model = R.load_base_model(model_cls, base, revision, dtype)
    lora_cfg = cfg["train"]["lora"]
    lora = lora_cls(
        r=int(lora_cfg["r"]), lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]), bias=str(lora_cfg["bias"]),
        target_modules=list(lora_cfg["target_modules"]), task_type="CAUSAL_LM")
    model = get_peft(model, lora)
    return model, tokenizer, dtype


def _make_dataset_and_collator(torch, examples: list[dict], pad_token_id: int):
    class _Ds(torch.utils.data.Dataset):
        def __len__(self): return len(examples)
        def __getitem__(self, i): return examples[i]

    def _collate(features):
        batch = D.collate_batch(list(features), pad_token_id=pad_token_id)
        return {k: torch.tensor(v) for k, v in batch.items()}

    return _Ds(), _collate


def _trainer_args(args_cls, cfg: dict, out_dir: Path, plan: dict, *, has_eval: bool, use_bf16: bool):
    t = cfg["train"]
    kwargs = dict(
        output_dir=str(out_dir),
        seed=int(t["seed"]), data_seed=int(t["seed"]),
        per_device_train_batch_size=int(t["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(t["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(t["gradient_accumulation_steps"]),
        num_train_epochs=float(t["num_train_epochs"]),
        learning_rate=float(t["learning_rate"]),
        lr_scheduler_type=str(t["lr_scheduler_type"]),
        warmup_steps=int(plan["warmup_steps"]),
        weight_decay=float(t["weight_decay"]),
        max_grad_norm=float(t["max_grad_norm"]),
        optim=str(t["optim"]),
        bf16=bool(use_bf16), fp16=False,
        logging_strategy="steps", logging_steps=int(t["logging_steps"]),
        save_strategy=str(t["save_strategy"]), save_steps=int(t["save_steps"]),
        save_total_limit=int(t["save_total_limit"]),
        gradient_checkpointing=bool(t["gradient_checkpointing"]),
        dataloader_num_workers=int(t["dataloader_num_workers"]),
        remove_unused_columns=False, report_to=[],
    )
    if has_eval:
        kwargs["eval_strategy"] = str(t["eval_strategy"])
        kwargs["eval_steps"] = int(t["eval_steps"])
    return args_cls(**kwargs)


def run(cfg: dict, *, dry_run: bool, resume_value: str | None, limit: int | None,
        max_seq_len: int | None) -> int:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments)

    torch.manual_seed(int(cfg["train"]["seed"]))
    device = R.pick_device(torch)
    seq_len = int(max_seq_len or cfg["train"]["max_seq_len"])

    data_dir = cfgmod.local_path(cfg, "train", "data_dir")
    train_rows = D.load_rows(data_dir / TRAIN_FILE, limit=limit)
    val_path = data_dir / VAL_FILE
    val_rows = D.load_rows(val_path, limit=limit) if val_path.is_file() else []

    model, tokenizer, dtype = _build(AutoModelForCausalLM, AutoTokenizer, LoraConfig,
                                     get_peft_model, cfg, device, torch)
    model.to(device)
    pad_id = int(tokenizer.pad_token_id)

    def examples_of(rows):
        out = []
        for r in rows:
            out.append(D.build_example(r, tokenizer, max_seq_len=seq_len))
        return out

    train_ex = examples_of(train_rows)
    val_ex = examples_of(val_rows) if val_rows else []
    lengths = [len(e["input_ids"]) for e in train_ex]
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    plan = plan_steps(cfg, len(train_ex))
    plan["max_input_tokens"] = max(lengths)
    plan["max_seq_len"] = seq_len
    plan["headroom"] = seq_len - max(lengths)
    plan["trainable_params"] = n_trainable
    plan["device"] = device
    plan["dtype"] = str(dtype).replace("torch.", "")

    print("── training plan ──")
    print(json.dumps(plan, indent=2))
    print(f"LoRA trainable: {n_trainable:,} params / "
          f"{sum(1 for p in model.parameters() if p.requires_grad)} tensors")

    if plan["headroom"] < 0:
        raise D.SftDataError(
            f"max_seq_len {seq_len} < longest example {max(lengths)}; raise it "
            f"(never truncate away the supervised target)")
    if dry_run:
        print("\nDRY-RUN: wiring built and validated; trainer.train() NOT called. "
              "No checkpoint written.")
        return 0

    out_dir = cfgmod.local_path(cfg, "train", "output_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    train_ds, collate = _make_dataset_and_collator(torch, train_ex, pad_id)
    val_ds = _make_dataset_and_collator(torch, val_ex, pad_id)[0] if val_ex else None
    targs = _trainer_args(TrainingArguments, cfg, out_dir, plan,
                          has_eval=val_ds is not None, use_bf16=(dtype == torch.bfloat16))
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      eval_dataset=val_ds, data_collator=collate,
                      processing_class=tokenizer)

    resume, how = resolve_resume(cfg, resume_value)
    print(f"\nresume: {how}" + (f" -> {resume}" if resume else ""))
    # HF Trainer restores model/adapter, optimizer, scheduler, global step and the
    # RNG/dataloader state from the checkpoint directory.
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    trainer.save_model(str(out_dir / "final"))
    print(f"done; checkpoints under {out_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Qwen PE SFT — formal LoRA training entry.")
    ap.add_argument("--config", default=str(cfgmod.DEFAULT_CONFIG))
    ap.add_argument("--check", action="store_true",
                    help="config + data contract gate only (no torch, no training)")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the full wiring, then stop before train()")
    ap.add_argument("--resume-from-checkpoint", default=None,
                    help="a checkpoint dir, 'auto' (latest), or omit for a fresh run")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the number of rows (debug only)")
    ap.add_argument("--max-seq-len", type=int, default=None, help="override config")
    ap.add_argument("--no-hash-check", action="store_true",
                    help="skip the frozen-manifest hash gate (debug only)")
    args = ap.parse_args(argv)

    cfg = cfgmod.load_config(args.config)
    summary = check_contract(cfg, verify_hash=not args.no_hash_check)
    print("── data contract ──")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.check:
        print("\nCHECK OK (no torch imported, no training). "
              "Authoritative bundle verifier: validate_dataset.py")
        return 0

    if not args.dry_run:
        print("\nNOTE: checkpoint selection is NOT train loss — run evaluate.py "
              "over the saved checkpoints (validation task metric).")
    return run(cfg, dry_run=args.dry_run, resume_value=args.resume_from_checkpoint,
               limit=args.limit, max_seq_len=args.max_seq_len)


if __name__ == "__main__":
    raise SystemExit(main())
