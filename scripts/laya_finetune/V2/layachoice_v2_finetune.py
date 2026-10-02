#!/usr/bin/env python3
"""LayaChoice-v2 — fine-tune the official Laya base on the frozen 4-way task.

Trains from the official ORIGINAL base checkpoint
(convaiinnovations/laya@55cf4c4e/multilingual), never from a v1 checkpoint.
The recipe (token budget, base, seeds, hyper-parameters) matches the accepted
v1 run; V2 differs only in the dataset/task (4-way: 3 capability + REJECT).

This module owns the model plumbing (base load, decisions, temperature fit,
the train loop, export) and reuses the dataset contract
(``layachoice_v2_dataset``) and the input boundary (``layachoice_v2_render``).

Modes:
  --train            4 epochs, one full-state checkpoint per epoch (default)
  --fit-temperature  LBFGS temperature on the frozen calibration split
  --export           write an Agent-loadable deployment directory

Usage:
    python layachoice_v2_finetune.py --train
    python layachoice_v2_finetune.py --train --smoke          # CPU, 8 rows, 1 step
    python layachoice_v2_finetune.py --fit-temperature --ckpt out/epoch-3
    python layachoice_v2_finetune.py --export --ckpt out/epoch-3 --out out/final
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

import laya.common as C

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import layachoice_v2_spec as S                     # noqa: E402
import layachoice_v2_dataset as D                  # noqa: E402
import layachoice_v2_render as R                   # noqa: E402

DEFAULT_OUT = HERE / "out"


# ── base checkpoint ─────────────────────────────────────────────────────────
def base_model_dir(revision: str = S.BASE_REVISION, subfolder: str = S.BASE_SUBFOLDER) -> Path:
    """The pinned multilingual checkpoint, fetched only if this revision is not cached."""
    from huggingface_hub import snapshot_download
    root = snapshot_download(
        repo_id=S.BASE_REPO, revision=revision,
        allow_patterns=[f"{subfolder}/rl_agent_config.json", f"{subfolder}/model.safetensors",
                        f"{subfolder}/tokenizer/*", f"{subfolder}/encoder/*"])
    root = Path(root)
    if root.name != revision:
        raise RuntimeError(f"resolved revision {root.name!r} != pinned {revision!r}")
    model_dir = root / subfolder
    for f in ("rl_agent_config.json", "model.safetensors", "encoder/config.json",
              "tokenizer/tokenizer.json"):
        if not (model_dir / f).exists():
            raise FileNotFoundError(f"base checkpoint incomplete: {model_dir / f}")
    return model_dir


def read_cfg(model_dir: Path, *, train: bool) -> dict:
    cfg = json.loads((model_dir / "rl_agent_config.json").read_text(encoding="utf-8"))
    if train:
        cfg["gradient_checkpointing"] = True
        cfg["max_tokens_per_batch"] = 4096
    cfg["max_len"] = S.MAX_LEN
    cfg["head_max_len"] = S.HEAD_MAX_LEN
    return cfg


def load_tokenizer(model_dir: Path):
    from transformers import AutoTokenizer
    import laya.agent as A
    A._fix_tokenizer_config(str(model_dir))
    path = model_dir / "tokenizer"
    return AutoTokenizer.from_pretrained(str(path if path.exists() else model_dir))


def model_from_checkpoint(ckpt: Path, model_dir: Path, device) -> tuple[torch.nn.Module, dict]:
    """Base architecture + a checkpoint's weights. strict=True on both loads."""
    from safetensors.torch import load_file
    cfg = read_cfg(model_dir, train=False)
    model = C.build_model(cfg, encoder_dir=str(model_dir / "encoder"))
    model.load_state_dict(load_file(str(ckpt / "model.safetensors")), strict=True)
    return model.to(device), cfg


def build_train_model(model_dir: Path, device) -> tuple[torch.nn.Module, dict]:
    """The official notebook's load path, verbatim in effect."""
    from safetensors.torch import load_file
    cfg = read_cfg(model_dir, train=True)
    model = C.build_model(cfg, encoder_dir=str(model_dir / "encoder"))
    model.load_state_dict(load_file(str(model_dir / "model.safetensors")), strict=True)
    model.encoder.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    return model.to(device), cfg


# ── decisions ───────────────────────────────────────────────────────────────
@torch.no_grad()
def decide(model, tok, examples: list[D.V2Example], cfg: dict, instructions: str,
           temperature, device, chunk: int = S.DECISION_CHUNK) -> list[dict]:
    """Per-row decisions, replicating the Agent's own path: softmax(logits[:k]/T) -> argmax.

    Top-1 is invariant under any positive scalar temperature; the temperature only
    moves the probabilities and the confidence.
    """
    model.eval()
    t_scale = float(temperature[C.QTYPES["choice"]] if isinstance(temperature, (list, tuple))
                    else temperature)
    out = []
    for i in range(0, len(examples), chunk):
        part = examples[i:i + chunk]
        items = [R.build_item(tok, e, instructions, cfg.get("max_len", S.MAX_LEN),
                              cfg.get("head_max_len", S.HEAD_MAX_LEN)) for e in part]
        b = C.collate_items([items], tok.pad_token_id)
        logits, _act = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                             b["marker_pos"].to(device), b["marker_mask"].to(device),
                             b["qtype"].to(device))
        logits = logits.float().cpu().numpy()
        for j, ex in enumerate(part):
            k = len(items[j]["markers"])
            z = logits[j, :k] / t_scale
            p = np.exp(z - z.max())
            p = p / p.sum()
            pred = int(p.argmax())
            out.append({
                "id": ex.id, "split": ex.split, "lang": ex.lang, "part": ex.part,
                "state": ex.state, "order": list(ex.order),
                "gold_index": ex.gold_index, "gold_id": ex.gold_id,
                "target_kind": ex.target_kind, "reject_index": ex.reject_index,
                "pred": pred, "pred_id": ex.order[pred], "hit": bool(pred == ex.gold_index),
                "gold_is_reject": ex.is_reject,
                "pred_is_reject": ex.order[pred] == S.REJECT_LABEL,
                "probs": [round(float(x), 6) for x in p],
                "confidence": round(float(p[pred]), 6),
                "gold_prob": round(float(p[ex.gold_index]), 6),
                "temperature": t_scale,
            })
    return out


def collect_fit_pairs(model, tok, examples: list[D.V2Example], cfg: dict,
                      instructions: str, device):
    """(logits[:k], gold_index) for every calibration row, in the frozen row order."""
    pairs = []
    for i in range(0, len(examples), S.DECISION_CHUNK):
        part = examples[i:i + S.DECISION_CHUNK]
        items = [R.build_item(tok, e, instructions, cfg.get("max_len", S.MAX_LEN),
                              cfg.get("head_max_len", S.HEAD_MAX_LEN)) for e in part]
        b = C.collate_items([items], tok.pad_token_id)
        with torch.no_grad():
            logits, _ = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                              b["marker_pos"].to(device), b["marker_mask"].to(device),
                              b["qtype"].to(device))
        logits = logits.float().cpu().numpy()
        for j, ex in enumerate(part):
            k = len(items[j]["markers"])
            pairs.append(([float(x) for x in logits[j, :k]], ex.gold_index))
    return pairs


# ── temperature calibration (official fit, Agent clamp) ──────────────────────
def fit_one_temp(pairs: list[tuple[list[float], int]], k: int) -> tuple[float, float]:
    """LBFGS fit of one scalar temperature on (logits, gold) pairs.

    Returns (raw_fitted_value, final_written_value); the written value is what the
    Agent's own ``clamp_temperature`` enforces on reload.
    """
    if len(pairs) < 10:
        return 1.0, C.clamp_temperature(1.0)
    Z = torch.full((len(pairs), k), -1e4)
    T = torch.zeros((len(pairs), k))
    for i, (logits, gold) in enumerate(pairs):
        Z[i, :len(logits)] = torch.tensor(logits, dtype=torch.float32)
        T[i, gold] = 1.0
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    try:
        opt.step(closure)
        raw = float(log_t.exp().item())
    except Exception as exc:                                  # noqa: BLE001
        print(f"fit_one_temp failed ({exc}); falling back to {S.FALLBACK_TEMP}")
        raw = S.FALLBACK_TEMP
    return raw, C.clamp_temperature(raw)


# ── RNG bookkeeping ─────────────────────────────────────────────────────────
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


# ── checkpoints ─────────────────────────────────────────────────────────────
def save_state_dict(state_dict: dict, path: Path) -> None:
    from safetensors.torch import save_file
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().to("cpu").contiguous() for k, v in state_dict.items()}, str(path))


def save_checkpoint(epoch_dir: Path, model, optimizer, scheduler, scaler, meta: dict) -> None:
    save_state_dict(model.state_dict(), epoch_dir / "model.safetensors")
    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "grad_scaler": scaler.state_dict(),
        "rng": rng_state(),
        "meta": meta,
    }, epoch_dir / "train_state.pt")
    (epoch_dir / "train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# ── train ───────────────────────────────────────────────────────────────────
def mode_train(args) -> None:
    import torch.distributed as dist

    smoke = args.smoke
    out = Path(args.out)
    data_dir = Path(args.data)
    manifest = D.read_manifest(data_dir)
    instructions = manifest.get("instructions", S.DEFAULT_INSTRUCTIONS)

    train_rows = D.load_split("train", data_dir)
    val_rows = D.load_split("val", data_dir, limit=S.SMOKE_ROWS if smoke else None)
    calib_rows, fit_rows = D.frozen_calibration(train_rows, manifest)
    epochs = 1 if smoke else args.epochs
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cpu" if smoke else ("cuda" if torch.cuda.is_available() else "cpu"))
    if smoke:
        world_size, rank, local_rank = 1, 0, 0
        fit_rows, calib_rows = fit_rows[:S.SMOKE_ROWS], calib_rows[:S.SMOKE_CALIB_ROWS]

    amp_name = None if smoke else read_cfg(base_model_dir(), train=False).get("amp_dtype")
    amp_dtype = None if smoke else C.amp_dtype(amp_name)
    use_scaler = amp_dtype == torch.float16
    if rank == 0:
        print(f"[train] device={device} world_size={world_size} amp={amp_name or 'fp32'} "
              f"grad_scaler={use_scaler} smoke={smoke}")

    model_dir = base_model_dir()
    cfg = read_cfg(model_dir, train=True)
    tok = load_tokenizer(model_dir)
    seed_everything(S.SEED_BASE + rank)
    model, _ = build_train_model(model_dir, device)

    train_items_all = fit_rows[: len(fit_rows) // world_size * world_size]
    my_rows = train_items_all[rank::world_size]
    if smoke:
        my_rows = my_rows[:S.SMOKE_ROWS]
    if rank == 0:
        print(f"[train] source_train_rows={len(train_rows)} calibration_rows={len(calib_rows)} "
              f"effective_train_rows={len(fit_rows)} after_world_size={len(train_items_all)} "
              f"this_rank={len(my_rows)}")

    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True)

    enc_params = [p for n, p in model.named_parameters() if "encoder." in n]
    head_params = [p for n, p in model.named_parameters() if "encoder." not in n]
    optimizer = torch.optim.AdamW(
        [{"params": enc_params, "lr": S.LR_ENCODER}, {"params": head_params, "lr": S.LR_HEAD}],
        weight_decay=S.WEIGHT_DECAY)
    # One optimizer step per full GRAD_ACCUM micro-batch group, plus one on the final
    # (possibly short) micro-batch of every epoch -- i.e. ceil(micro_batches / GRAD_ACCUM),
    # which is identical to ceil(my_rows / (MICRO_BATCH*GRAD_ACCUM)). The cosine schedule
    # must span EXACTLY that many steps, so T_max == the number of optimizer.step() calls
    # the loop below will make (a plain floor() undercounts the trailing short group).
    updates_per_epoch = -(-len(my_rows) // (S.MICRO_BATCH * S.GRAD_ACCUM))
    total_updates = max(1, updates_per_epoch * epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_updates, eta_min=1e-6)
    opt_steps = sched_steps = 0
    scaler = torch.amp.GradScaler(device.type, enabled=use_scaler)

    model.train()
    run_start = time.time()
    for epoch in range(epochs):
        t0 = time.time()
        random.seed(S.SEED_BASE + epoch + rank)
        random.shuffle(my_rows)
        progress = epoch / max(1, epochs - 1)
        sigma = S.SIGMA_START + (S.SIGMA_END - S.SIGMA_START) * progress
        accum_step, loss_sum, loss_n, updates = 0, 0.0, 0, 0
        for b_idx in range(0, len(my_rows), S.MICRO_BATCH):
            chunk = my_rows[b_idx:b_idx + S.MICRO_BATCH]
            items = [R.build_item(tok, e, instructions, cfg.get("max_len", S.MAX_LEN),
                                  cfg.get("head_max_len", S.HEAD_MAX_LEN)) for e in chunk]
            b = C.collate_items([items], tok.pad_token_id)
            ids = b["input_ids"].to(device)
            att = b["attention_mask"].to(device)
            mpos = b["marker_pos"].to(device)
            mmask = b["marker_mask"].to(device)
            qtype = b["qtype"].to(device)
            target = b["target"].to(device)

            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits, act = model(ids, att, mpos, mmask, qtype)
            logits = logits.float()
            mask = mmask
            k = mask.sum(-1, keepdim=True).float()
            eps = torch.randn((S.GROUP_SIZE,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                r = C.proper_reward(q, target.unsqueeze(0), qtype, mask,
                                    w_sph=S.W_SPH, w_rps=S.W_RPS)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)) \
                .sum(-1).mean()
            # 0*act keeps the act head in the graph for find_unused_parameters
            loss = (loss_rl + S.CE_WEIGHT * loss_ce) / S.GRAD_ACCUM + 0.0 * act.sum()

            scaler.scale(loss).backward()
            accum_step += 1
            loss_sum += float(loss.detach()) * S.GRAD_ACCUM
            loss_n += 1
            if accum_step % S.GRAD_ACCUM == 0 or (b_idx + S.MICRO_BATCH) >= len(my_rows):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), S.CLIP)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                opt_steps += 1
                sched_steps += 1

        meta = {"epoch": epoch + 1, "epochs": epochs, "sigma": round(sigma, 4),
                "micro_batches": loss_n, "optimizer_updates": updates,
                "mean_loss": round(loss_sum / max(1, loss_n), 6),
                "lr_encoder": optimizer.param_groups[0]["lr"],
                "lr_head": optimizer.param_groups[1]["lr"],
                "rows_this_rank": len(my_rows), "world_size": world_size, "rank": rank,
                "seconds": round(time.time() - t0, 2), "seed": S.SEED_BASE + epoch + rank}
        if rank == 0:
            epoch_dir = out / f"epoch-{epoch + 1}"
            if epoch_dir.exists() and not args.overwrite:
                raise FileExistsError(f"{epoch_dir} exists; pass --overwrite to replace it")
            save_checkpoint(epoch_dir, model, optimizer, scheduler, scaler, meta)
            # Per-epoch validation, reported the moment the epoch ends. Forward-only
            # (no_grad + eval); the RNG is snapshotted and restored around it so the
            # training trajectory stays byte-identical to a run without validation.
            import layachoice_v2_eval as E          # lazy: eval imports this module
            rng_before = rng_state()
            core = model.module if hasattr(model, "module") else model
            t_val = time.time()
            decided = decide(core, tok, val_rows, cfg, instructions,
                             [1.0, 1.0, 1.0], device)
            restore_rng_state(rng_before)
            model.train()
            rep = E.summarise(decided, arm=f"validation-epoch-{epoch + 1}", split="val",
                              temperature=[1.0, 1.0, 1.0],
                              card_view=manifest["formal_card_view"])
            rj = rep["reject"]
            val_metrics = {
                "epoch": epoch + 1, "rows": rep["rows"], "top1": rep["top1"],
                "ece": rep["ece"], "reject_recall": rj["recall"],
                "reject_fpr": rj["false_positive_rate"], "gold_reject_n": rj["gold_reject_n"],
                "seconds": round(time.time() - t_val, 2),
            }
            (epoch_dir / "val_metrics.json").write_text(
                json.dumps(val_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"[epoch {epoch + 1}/{epochs}] train_loss={meta['mean_loss']:.6f} "
                  f"val_top1={val_metrics['top1']:.4f} val_ece={val_metrics['ece']:.4f} "
                  f"reject_recall={val_metrics['reject_recall']:.4f} "
                  f"reject_fpr={val_metrics['reject_fpr']:.4f}  "
                  f"[sigma={meta['sigma']} updates={updates} "
                  f"train {meta['seconds']}s, val {val_metrics['seconds']}s, "
                  f"elapsed {round(time.time() - run_start, 2)}s] -> {epoch_dir}", flush=True)

    # The LR schedule must have advanced exactly once per optimizer step, and exactly the
    # planned number of times -- otherwise the cosine never matched the training horizon.
    if sched_steps != opt_steps:
        raise RuntimeError(f"scheduler.step() called {sched_steps}x but optimizer.step() "
                           f"{opt_steps}x")
    if opt_steps != total_updates:
        raise RuntimeError(f"actual optimizer updates {opt_steps} != planned total_updates "
                           f"{total_updates}")

    if rank == 0:
        import transformers
        import laya
        run_meta = {
            "bench": manifest["bench"],
            "base_checkpoint": {"repo_id": S.BASE_REPO, "revision": S.BASE_REVISION,
                                "subfolder": S.BASE_SUBFOLDER},
            "formal_card_view": manifest["formal_card_view"],
            "option_slots": S.OPTION_SLOTS,
            "reject_label": S.REJECT_LABEL,
            "option_max_tokens": S.OPTION_MAX_TOKENS,
            "head_max_len": S.HEAD_MAX_LEN,
            "max_len": S.MAX_LEN,
            "rows": {"source_train_rows": len(train_rows),
                     "calibration_rows": len(calib_rows),
                     "effective_train_rows": len(fit_rows),
                     "after_world_size_alignment": len(train_items_all)},
            "calibration": {"seed": S.CALIB_SEED, "indices": manifest["calibration"]["indices"],
                            "ids": manifest["calibration"]["ids"]},
            "split_by_rank": {str(r): len(train_items_all[r::world_size])
                              for r in range(world_size)},
            "hyperparameters": {"epochs": epochs, "micro_batch": S.MICRO_BATCH,
                                "grad_accum": S.GRAD_ACCUM, "group_size": S.GROUP_SIZE,
                                "lr_encoder": S.LR_ENCODER, "lr_head": S.LR_HEAD,
                                "weight_decay": S.WEIGHT_DECAY, "sigma_start": S.SIGMA_START,
                                "sigma_end": S.SIGMA_END, "ce_weight": S.CE_WEIGHT,
                                "w_sph": S.W_SPH, "w_rps": S.W_RPS, "clip": S.CLIP,
                                "total_updates": total_updates,
                                "optimizer_updates_actual": opt_steps,
                                "scheduler_steps_actual": sched_steps},
            "runtime": {"device": str(device), "world_size": world_size,
                        "amp_dtype": amp_name, "autocast": str(amp_dtype) if amp_dtype else None,
                        "grad_scaler": use_scaler, "seed_base": S.SEED_BASE,
                        "smoke": smoke,
                        "torch": torch.__version__, "laya": laya.__version__,
                        "transformers": transformers.__version__},
            "bundle_source_shas": manifest["source_files"],
        }
        (out / "train_manifest.json").write_text(
            json.dumps(run_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[train] wrote {out / 'train_manifest.json'}")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


# ── temperature ─────────────────────────────────────────────────────────────
def mode_fit_temperature(args) -> None:
    ckpt = Path(args.ckpt)
    manifest = D.read_manifest(Path(args.data))
    instructions = manifest.get("instructions", S.DEFAULT_INSTRUCTIONS)
    train_rows = D.load_split("train", Path(args.data))
    calib_rows, _ = D.frozen_calibration(train_rows, manifest)
    if args.smoke:
        calib_rows = calib_rows[:S.SMOKE_CALIB_ROWS]

    device = torch.device("cpu" if not torch.cuda.is_available() or args.smoke else "cuda")
    model_dir = base_model_dir()
    cfg = read_cfg(model_dir, train=False)
    tok = load_tokenizer(model_dir)
    model, _ = model_from_checkpoint(ckpt, model_dir, device)

    pairs = collect_fit_pairs(model, tok, calib_rows, cfg, instructions, device)
    k = len(calib_rows[0].order)
    raw, written = fit_one_temp(pairs, k)
    base_cfg = json.loads((model_dir / "rl_agent_config.json").read_text(encoding="utf-8"))
    base_temp = list(base_cfg.get("temperature", [1.0, 1.0, 1.0]))
    temperature = list(base_temp)
    temperature[C.QTYPES["choice"]] = written
    payload = {
        "ckpt": str(ckpt), "qtype": "choice", "qt_index": C.QTYPES["choice"],
        "calibration_rows": len(calib_rows), "calibration_seed": S.CALIB_SEED,
        "raw_fitted_value": raw, "final_written_value": written,
        "clamp": [C.TEMP_MIN, C.TEMP_MAX],
        "base_temperature": base_temp, "temperature": temperature,
        "fallback": S.FALLBACK_TEMP, "smoke": args.smoke,
    }
    if C.clamp_temperature(raw) != written:
        raise RuntimeError(f"clamp round-trip broken: clamp({raw}) != {written}")
    out_dir = Path(args.out) if args.out else ckpt.parent / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "temperature.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[calibration] fit {raw:.4f} -> written {written:.4f} "
          f"(clamp [{C.TEMP_MIN}, {C.TEMP_MAX}]) on {len(calib_rows)} rows -> "
          f"{out_dir / 'temperature.json'}")


# ── export ─────────────────────────────────────────────────────────────────
def mode_export(args) -> None:
    ckpt = Path(args.ckpt)
    out = Path(args.out)
    manifest = D.read_manifest(Path(args.data))
    instructions = manifest.get("instructions", S.DEFAULT_INSTRUCTIONS)
    model_dir = base_model_dir()
    cfg = read_cfg(model_dir, train=False)
    tok = load_tokenizer(model_dir)
    device = torch.device("cpu")
    model, _ = model_from_checkpoint(ckpt, model_dir, device)

    temp_file = Path(args.temperature_file) if args.temperature_file \
        else ckpt.parent / "calibration" / "temperature.json"
    temperature = [1.0, 1.0, 1.0]
    if temp_file.exists():
        payload = json.loads(temp_file.read_text(encoding="utf-8"))
        temperature = list(payload["temperature"])
    elif not args.smoke:
        raise FileNotFoundError(
            f"no fitted temperature at {temp_file}; run --fit-temperature first "
            f"(or pass --smoke to export an uncalibrated artifact)")

    out.mkdir(parents=True, exist_ok=True)
    dtype = {"preserve": None, "fp16": torch.float16, "bf16": torch.bfloat16}[args.export_dtype]
    state = model.state_dict()
    if dtype is not None:
        state = {k: v.to(dtype) for k, v in state.items()}
    save_state_dict(state, out / "model.safetensors")
    (out / "encoder").mkdir(exist_ok=True)
    shutil.copy2(model_dir / "encoder" / "config.json", out / "encoder" / "config.json")
    if (out / "tokenizer").exists():
        shutil.rmtree(out / "tokenizer")
    shutil.copytree(model_dir / "tokenizer", out / "tokenizer")

    exported = dict(cfg)
    exported["fine_tuned"] = True
    exported["temperature"] = temperature
    exported.pop("temperature_by_options", None)
    exported["model_name"] = "layachoice-v2"
    exported["layachoice"] = {"checkpoint": str(ckpt), "option_slots": S.OPTION_SLOTS,
                              "reject_label": S.REJECT_LABEL,
                              "option_max_tokens": S.OPTION_MAX_TOKENS,
                              "head_max_len": S.HEAD_MAX_LEN, "max_len": S.MAX_LEN,
                              "calibration": str(temp_file) if temp_file.exists() else None,
                              "export_dtype": args.export_dtype}
    (out / "rl_agent_config.json").write_text(
        json.dumps(exported, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Reload through the public entry point to prove the directory is Agent-loadable
    # and that the temperature survives a real reload.
    R.install()
    import laya
    agent = laya.Agent(str(out.resolve()), device="cpu")
    reloaded = list(agent.temperature)
    if reloaded[C.QTYPES["choice"]] != C.clamp_temperature(temperature[C.QTYPES["choice"]]):
        raise RuntimeError(f"Agent reloaded temperature {reloaded} != exported {temperature}")
    rows = D.load_split("test", Path(args.data), limit=5)
    questions = {S.QID: {"type": "choice", "instructions": instructions,
                         "criteria": dict(zip(r.order, r.options))} for r in rows[:1]}
    answers = agent.predict_batch([r.state for r in rows], questions,
                                  batch_size=5, max_len=cfg.get("max_len", S.MAX_LEN),
                                  head_max_len=cfg.get("head_max_len", S.HEAD_MAX_LEN))
    smoke = {"export_dir": str(out), "export_dtype": args.export_dtype,
             "size_bytes": (out / "model.safetensors").stat().st_size,
             "agent_temperature": reloaded, "rows": []}
    for row, res in zip(rows, answers):
        ans = res["answers"][S.QID]
        smoke["rows"].append({"id": row.id, "gold_id": row.gold_id,
                              "agent_choice": ans["choice"], "hit": ans["choice"] == row.gold_id,
                              "probabilities": ans["probabilities"]})
    (out / "reload_smoke.json").write_text(
        json.dumps(smoke, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[export] {out} dtype={args.export_dtype} "
          f"temperature={temperature} ({smoke['size_bytes'] / 1e6:.1f} MB); "
          f"Agent reload ok, {sum(r['hit'] for r in smoke['rows'])}/{len(smoke['rows'])} smoke hits")


# ── CLI ─────────────────────────────────────────────────────────────────────
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--train", action="store_true", help="fine-tune (default)")
    mode.add_argument("--fit-temperature", action="store_true")
    mode.add_argument("--export", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="CPU float32, 8 rows, 1 step, no GradScaler, no DDP")
    ap.add_argument("--data", default=str(D.DATA))
    ap.add_argument("--out", default=None,
                    help="run directory (--train) or deployment directory (--export); "
                         "defaults to scripts/laya_finetune/V2/out, and to <ckpt>/../calibration "
                         "for --fit-temperature")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--epochs", type=int, default=S.EPOCHS)
    ap.add_argument("--export-dtype", choices=("preserve", "fp16", "bf16"), default="preserve")
    ap.add_argument("--temperature-file", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    if args.export and not args.ckpt:
        ap.error("--export requires --ckpt")
    if args.fit_temperature and not args.ckpt:
        ap.error("--fit-temperature requires --ckpt")
    if args.export and not args.out:
        ap.error("--export requires --out (the deployment directory)")
    return args


def main(argv=None) -> None:
    for stream in (sys.stdout, sys.stderr):          # live, line-buffered output
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass
    args = parse_args(argv)
    if args.export:
        mode_export(args)
    elif args.fit_temperature:
        mode_fit_temperature(args)          # --out defaults to <ckpt>/../calibration
    else:
        mode_train(args if args.out else _with(args, out=str(DEFAULT_OUT)))


def _with(args, **kw):
    return argparse.Namespace(**{**vars(args), **kw})


if __name__ == "__main__":
    sys.exit(main())
