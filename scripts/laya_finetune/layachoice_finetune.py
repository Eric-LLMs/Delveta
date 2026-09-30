#!/usr/bin/env python3
"""Stage 2 - fine-tune Laya (multilingual) on LayaChoice-v1, then package it.

Runs on the rented GPU box. Everything it needs is under scripts/laya_finetune/data/
(the frozen bundle written by layachoice_prepare.py) plus the pinned base checkpoint
convaiinnovations/laya@55cf4c4e/multilingual - no registry, no database, no repo
checkout beyond this directory.

The recipe is the official Laya fine-tuning notebook (item construction + the
`train_ddp.py` loop), reproduced in README.md and followed verbatim here. The six
explicit deviations from that notebook are frozen and listed in README.md.

Modes:
  --train            4 epochs, one full-state checkpoint per epoch (default)
  --fit-temperature  LBFGS temperature on the frozen 85-row calibration split
  --export           write an Agent-loadable deployment directory

Usage:
    python layachoice_finetune.py --train
    python layachoice_finetune.py --train --smoke          # CPU, 8 rows, 1 step
    python layachoice_finetune.py --fit-temperature --ckpt out/epoch-3
    python layachoice_finetune.py --export --ckpt out/epoch-3 --out out/final
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
from typing import Dict, List, Optional, Union

import numpy as np
import torch

import laya.common as C

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DEFAULT_OUT = HERE / "out"

# ── input-token budget (train and eval read these two; they must not diverge) ─
# OPTION_MAX_TOKENS - the per-option hard cap. `laya.common.build_sequence` hard-codes
# 48, which cuts *every* frozen option: all 2568 B_noprov options (856 rows x 3) are
# 136-230 tokens and none is below 136. 256 is the cap the dataset was audited under
# (logs/_laya_ds/coverage_audit.json -> tokens: over_library_cap_48=2568,
# over_patched_cap_256=0), so nothing is truncated at 256.
#
# HEAD_MAX_LEN - a BUDGET the question head and every option SHARE, not a length:
# `opt_budget = head_max_len - sum(options)`; below 16 the even-share fallback shrinks
# the options and `head_ids[:max(8, opt_budget)]` cuts the head. At the 256 cap
# sum(3 options) + 3 markers peaks at 599, so 768 leaves ~169 tokens of slack and both
# the options and the 13-token head survive whole. The base checkpoint's own value is
# 256, which cannot cover 599 - at 256 the even-share fallback cuts every option to 80
# tokens. Raising HEAD_MAX_LEN adds no tokens to the sequence: once nothing is being
# cut, the ids are identical for 640 and for 896.
#
# MAX_LEN - the whole-sequence ceiling. The longest complete sequence measures 640
# tokens (13 head + 599 options + up to 24 state + 4 cls/sep), so 1024 never binds.
OPTION_MAX_TOKENS = 256
HEAD_MAX_LEN = 768
MAX_LEN = 1024
EPOCHS = 4
MICRO_BATCH = 8
GRAD_ACCUM = 4
GROUP_SIZE = 4
LR_ENCODER = 2.5e-5
LR_HEAD = 1.0e-4
WEIGHT_DECAY = 0.01
SIGMA_START = 0.4
SIGMA_END = 0.1
CE_WEIGHT = 1.0
W_SPH = 0.75
W_RPS = 1.0
CLIP = 1.0
SEED_BASE = 42
CALIB_SEED = 20260922
CALIB_MAX = 400
SMOKE_ROWS = 8
SMOKE_CALIB_ROWS = 5
FALLBACK_TEMP = 1.2
BASE_REPO = "convaiinnovations/laya"
BASE_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
BASE_SUBFOLDER = "multilingual"
DEFAULT_INSTRUCTIONS = "Which capability should handle the user's request?"
QID = "capability"
DECISION_CHUNK = 16


# ── the option builder (drop-in patch for laya.agent) ────────────────────────
# A verbatim copy of `laya.common.build_sequence` with ONE change: the per-option
# hard cap `max_length=48` becomes `max_length=OPTION_MAX_TOKENS`. Everything else
# is byte-for-byte the stock body - the `opt_budget < 16` even-share fallback, the
# `head_ids[: max(8, opt_budget)]` floor, the `#538` note and the `return_stats`
# dict. Keeping the fallback is the point: it is what stops a budget too tight for
# the options from silently cutting the instruction head down to 8 tokens.
def build_sequence(
    tok,
    state: Union[str, dict, list],
    q: Dict,
    max_len: int = 512,
    head_max_len: int = 192,
    option_order: Optional[List[int]] = None,
    truncate_left: bool = False,
    state_ids: Optional[List[int]] = None,
    return_stats: bool = False,
):
    """Format: [CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP].

    `state_ids` lets a caller tokenize the shared state once and reuse it across every question,
    instead of re-serializing and re-tokenizing the same document per question.

    `return_stats` adds a third return value describing what the head budget did to the options:
    `options` (how many the question defines), `options_distinct` (how many still have a token
    span of their own) and `tokens_per_option` (the cap applied to each, or None when none was).
    """
    mask_tok = tok.mask_token
    opts = C.render_options(q)
    order = option_order if option_order is not None else list(range(len(opts)))
    ins = str(q["ins"]).replace(mask_tok, " ")
    head_ids = C._encode_question_text(tok, "%s question: %s" % (q["t"], ins), add_special_tokens=False)
    opt_ids = []
    for i in order:
        # Cap at the tokenizer, not after the fact: `[:48]` still makes the tokenizer process the
        # whole (possibly long) description. truncation=True, max_length=48 keeps the first 48
        # tokens, which is exactly what the previous slice produced.
        opt_tokens = C._encode_question_text(
            tok,
            " " + opts[i].replace(mask_tok, " "),
            add_special_tokens=False,
            truncation=True,
            max_length=OPTION_MAX_TOKENS,
        )
        opt_ids.append([tok.mask_token_id] + opt_tokens)
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    per_option = None
    if opt_budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        per_option = per
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    head_ids = head_ids[: max(8, opt_budget)]
    ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tok.sep_token_id)
    room = max(0, max_len - len(ids) - 1)
    if state_ids is None:
        state_ids = C.encode_text(tok, C.serialize_state(state).replace(mask_tok, " "),
                                add_special_tokens=False)["input_ids"]
    # not state_ids[-room:]: with no room left, state_ids[-0:] is the whole state rather than none of it
    st = state_ids[max(0, len(state_ids) - room):] if truncate_left else state_ids[:room]
    ids = ids + st + [tok.sep_token_id]
    ids, markers = ids[:max_len], [m for m in markers if m < max_len]
    if not return_stats:
        return ids, markers
    # Two options that share a prefix can come out of the cut as the same token span: the marker
    # count still matches the option count, so the guard in `Agent._encode_state` passes and
    # nothing downstream can tell that the question lost the ability to name them apart. Counted
    # on the capped option ids, before assembly: re-slicing the finished sequence cannot close
    # the last option's span -- it runs on into the serialized state, which differs per request,
    # so the last option always looks distinguishable however it collided (#538).
    return ids, markers, {
        "options": len(opt_ids),
        "options_distinct": len({tuple(o) for o in opt_ids}),
        "tokens_per_option": per_option,
    }


def install() -> None:
    """Point laya.agent's build_sequence call site at the 256-cap builder."""
    import laya.agent as A
    A.build_sequence = build_sequence


# ── frozen bundle ───────────────────────────────────────────────────────────
def read_manifest(data_dir: Path = DATA) -> dict:
    return json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))


def load_bundle(split: str, data_dir: Path = DATA, limit: int | None = None) -> list[dict]:
    """Read one split of the frozen bundle, asserting it is the frozen revision."""
    import gzip
    path = data_dir / f"layachoice_v1_{split}_b_noprov.jsonl.gz"
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if limit is not None:
        return rows[:limit]
    manifest = read_manifest(data_dir)
    expected = manifest["rows"][split]
    if len(rows) != expected:
        raise RuntimeError(f"{split}: {len(rows)} rows, manifest says {expected}")
    if split == "train":
        src = "LayaChoice_v1_train.jsonl"
    elif split == "val":
        src = "LayaChoice_v1_validation_raw_v3.jsonl"
    else:
        src = "LayaChoice_v1_test_v3.jsonl"
    want = manifest["source_files"][src]
    if {r["src_sha"] for r in rows} != {want}:
        raise RuntimeError(f"{split}: bundle rows do not carry the frozen source sha {want}")
    return rows


def calibration_split(n_train: int) -> tuple[list[int], list[int]]:
    """Recompute the frozen calibration holdout (seed 20260922) from the row count."""
    order = list(range(n_train))
    random.Random(CALIB_SEED).shuffle(order)
    n_calib = min(CALIB_MAX, n_train // 10)
    return sorted(order[:n_calib]), sorted(order[n_calib:])


def frozen_calibration(train_rows: list[dict], manifest: dict) -> tuple[list[dict], list[dict]]:
    """Split the train bundle into (calibration, effective-train) by the manifest's
    recorded indices, after proving the recorded indices are reproducible.

    Exclusion happens on the FULL 856 before any world_size alignment, so a
    multi-rank truncation can never move a calibration row into training.
    """
    calib_idx, train_idx = calibration_split(len(train_rows))
    recorded = manifest["calibration"]["indices"]
    if calib_idx != recorded:
        raise RuntimeError(
            f"calibration indices are not reproducible: recomputed {len(calib_idx)} rows "
            f"differ from the {len(recorded)} recorded in manifest.json")
    ids = [train_rows[i]["id"] for i in calib_idx]
    if ids != manifest["calibration"]["ids"]:
        raise RuntimeError("calibration ids are not reproducible from the recorded indices")
    return [train_rows[i] for i in calib_idx], [train_rows[i] for i in train_idx]


# ── base checkpoint ─────────────────────────────────────────────────────────
def base_model_dir(revision: str = BASE_REVISION, subfolder: str = BASE_SUBFOLDER) -> Path:
    """The pinned multilingual checkpoint, fetched only if this revision is not cached."""
    from huggingface_hub import snapshot_download
    root = snapshot_download(
        repo_id=BASE_REPO, revision=revision,
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
        # The official notebook sets these two before building the model.
        cfg["gradient_checkpointing"] = True
        cfg["max_tokens_per_batch"] = 4096
    # The token budget is NOT train-only: the base checkpoint ships max_len=1024 /
    # head_max_len=256, and the eval and export paths read the same constants, so
    # training and inference can never render the input differently.
    cfg["max_len"] = MAX_LEN
    cfg["head_max_len"] = HEAD_MAX_LEN
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


# ── items and decisions ─────────────────────────────────────────────────────
def question_of(row: dict, instructions: str) -> dict:
    """The frozen B_noprov card view, in the library's compact question shape."""
    return {"t": "choice", "ins": instructions,
            "crit": dict(zip(row["order"], row["options"]))}


def build_item(tok, row: dict, cfg: dict, instructions: str) -> dict:
    q = question_of(row, instructions)
    ids, markers = build_sequence(tok, row["state"], q,
                                  cfg.get("max_len", MAX_LEN), cfg.get("head_max_len", HEAD_MAX_LEN))
    if len(markers) != len(C.render_options(q)):
        raise RuntimeError(f"{row['id']}: options exceed head_max_len={cfg.get('head_max_len')}")
    target = [0.0] * len(markers)
    target[row["gold"]] = 1.0
    return {"ids": ids, "markers": markers, "qtype": C.QTYPES["choice"],
            "target": target, "label": int(np.argmax(target)), "id": row["id"]}


@torch.no_grad()
def decide(model, tok, rows: list[dict], cfg: dict, instructions: str, temperature,
           device, chunk: int = DECISION_CHUNK) -> list[dict]:
    """Per-row decisions, replicating the Agent's own path: softmax(logits[:k]/T) -> argmax.

    Top-1 is invariant under any positive scalar temperature (softmax is monotone in
    logits/T), so `pred` does not depend on which temperature is passed; the
    probabilities and the confidence do.
    """
    model.eval()
    t_scale = float(temperature[C.QTYPES["choice"]] if isinstance(temperature, (list, tuple))
                    else temperature)
    out = []
    for i in range(0, len(rows), chunk):
        part = rows[i:i + chunk]
        items = [build_item(tok, r, cfg, instructions) for r in part]
        b = C.collate_items([items], tok.pad_token_id)
        logits, _act = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                             b["marker_pos"].to(device), b["marker_mask"].to(device),
                             b["qtype"].to(device))
        logits = logits.float().cpu().numpy()
        for j, row in enumerate(part):
            k = len(items[j]["markers"])
            z = logits[j, :k] / t_scale
            p = np.exp(z - z.max())
            p = p / p.sum()
            pred = int(p.argmax())
            roles = {row["gold"]: "gold", row["d1"]: "d1", row["d2"]: "d2"}
            out.append({
                "id": row["id"], "split": row["split"], "lang": row["lang"],
                "state": row["state"],
                "order": list(row["order"]),
                "gold": row["gold"], "gold_cap": row["order"][row["gold"]],
                "pred": pred, "pred_cap": row["order"][pred], "hit": bool(pred == row["gold"]),
                "pred_role": roles.get(pred, "?"),
                "probs": [round(float(x), 6) for x in p],
                "confidence": round(float(p[pred]), 6),
                "gold_prob": round(float(p[row["gold"]]), 6),
                "temperature": t_scale,
            })
    return out


# ── temperature calibration (official fit, Agent clamp) ──────────────────────
def fit_one_temp(pairs: list[tuple[list[float], int]], k: int) -> tuple[float, float]:
    """LBFGS fit of one scalar temperature on (logits, gold) pairs.

    Returns (raw_fitted_value, final_written_value). The written value is
    `laya.common.clamp_temperature`'s output, i.e. the range the Agent itself
    enforces on reload - the notebook's own [0.1, 10.0] clamp is not used, or the
    fitted value would be silently rewritten at load time and the "fit result ==
    exported config == Agent's effective value" invariant would break.
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
        print(f"fit_one_temp failed ({exc}); falling back to {FALLBACK_TEMP}")
        raw = FALLBACK_TEMP
    return raw, C.clamp_temperature(raw)


def collect_fit_pairs(model, tok, rows: list[dict], cfg: dict, instructions: str, device):
    """(logits[:k], gold) for every calibration row, in the frozen row order."""
    pairs = []
    for i in range(0, len(rows), DECISION_CHUNK):
        part = rows[i:i + DECISION_CHUNK]
        items = [build_item(tok, r, cfg, instructions) for r in part]
        b = C.collate_items([items], tok.pad_token_id)
        with torch.no_grad():
            logits, _ = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                              b["marker_pos"].to(device), b["marker_mask"].to(device),
                              b["qtype"].to(device))
        logits = logits.float().cpu().numpy()
        for j, row in enumerate(part):
            k = len(items[j]["markers"])
            pairs.append(([float(x) for x in logits[j, :k]], row["gold"]))
    return pairs


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
    manifest = read_manifest(data_dir)
    instructions = manifest.get("instructions", DEFAULT_INSTRUCTIONS)

    train_rows = load_bundle("train", data_dir)
    calib_rows, fit_rows = frozen_calibration(train_rows, manifest)
    epochs = 1 if smoke else args.epochs
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cpu" if smoke else ("cuda" if torch.cuda.is_available() else "cpu"))
    if smoke:
        world_size, rank, local_rank = 1, 0, 0
        fit_rows, calib_rows = fit_rows[:SMOKE_ROWS], calib_rows[:SMOKE_CALIB_ROWS]

    amp_name = None if smoke else read_cfg(base_model_dir(), train=False).get("amp_dtype")
    amp_dtype = None if smoke else C.amp_dtype(amp_name)
    use_scaler = amp_dtype == torch.float16
    if rank == 0:
        print(f"[train] device={device} world_size={world_size} amp={amp_name or 'fp32'} "
              f"grad_scaler={use_scaler} smoke={smoke}")

    model_dir = base_model_dir()
    cfg = read_cfg(model_dir, train=True)
    tok = load_tokenizer(model_dir)
    seed_everything(SEED_BASE + rank)
    model, _ = build_train_model(model_dir, device)

    train_items_all = fit_rows[: len(fit_rows) // world_size * world_size]
    my_rows = train_items_all[rank::world_size]
    if smoke:
        my_rows = my_rows[:SMOKE_ROWS]
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
        [{"params": enc_params, "lr": LR_ENCODER}, {"params": head_params, "lr": LR_HEAD}],
        weight_decay=WEIGHT_DECAY)
    total_updates = max(1, (len(my_rows) // (MICRO_BATCH * GRAD_ACCUM)) * epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_updates, eta_min=1e-6)
    scaler = torch.amp.GradScaler(device.type, enabled=use_scaler)

    model.train()
    for epoch in range(epochs):
        t0 = time.time()
        random.seed(SEED_BASE + epoch + rank)
        random.shuffle(my_rows)
        progress = epoch / max(1, epochs - 1)
        sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * progress
        accum_step, loss_sum, loss_n, updates = 0, 0.0, 0, 0
        for b_idx in range(0, len(my_rows), MICRO_BATCH):
            chunk = my_rows[b_idx:b_idx + MICRO_BATCH]
            items = [build_item(tok, r, cfg, instructions) for r in chunk]
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
            eps = torch.randn((GROUP_SIZE,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                r = C.proper_reward(q, target.unsqueeze(0), qtype, mask, w_sph=W_SPH, w_rps=W_RPS)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)) \
                .sum(-1).mean()
            # 0*act keeps the act head in the graph for find_unused_parameters
            loss = (loss_rl + CE_WEIGHT * loss_ce) / GRAD_ACCUM + 0.0 * act.sum()

            scaler.scale(loss).backward()
            accum_step += 1
            loss_sum += float(loss.detach()) * GRAD_ACCUM
            loss_n += 1
            if accum_step % GRAD_ACCUM == 0 or (b_idx + MICRO_BATCH) >= len(my_rows):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1

        meta = {"epoch": epoch + 1, "epochs": epochs, "sigma": round(sigma, 4),
                "micro_batches": loss_n, "optimizer_updates": updates,
                "mean_loss": round(loss_sum / max(1, loss_n), 6),
                "lr_encoder": optimizer.param_groups[0]["lr"],
                "lr_head": optimizer.param_groups[1]["lr"],
                "rows_this_rank": len(my_rows), "world_size": world_size, "rank": rank,
                "seconds": round(time.time() - t0, 2), "seed": SEED_BASE + epoch + rank}
        if rank == 0:
            epoch_dir = out / f"epoch-{epoch + 1}"
            if epoch_dir.exists() and not args.overwrite:
                raise FileExistsError(f"{epoch_dir} exists; pass --overwrite to replace it")
            save_checkpoint(epoch_dir, model, optimizer, scheduler, scaler, meta)
            print(f"[epoch {epoch + 1}/{epochs}] loss={meta['mean_loss']} sigma={meta['sigma']} "
                  f"updates={updates} {meta['seconds']}s -> {epoch_dir}")

    if rank == 0:
        import transformers
        import laya
        run_meta = {
            "bench": manifest["bench"],
            "base_checkpoint": {"repo_id": BASE_REPO, "revision": BASE_REVISION,
                                "subfolder": BASE_SUBFOLDER},
            "formal_card_view": manifest["formal_card_view"],
            "option_max_tokens": OPTION_MAX_TOKENS,
            "head_max_len": HEAD_MAX_LEN,
            "max_len": MAX_LEN,
            "option_cap_note": "library default is 48; 256 is the cap the dataset was audited "
                               "under (logs/_laya_ds/coverage_audit.json -> tokens: "
                               "over_library_cap_48=2568, over_patched_cap_256=0). The stock "
                               "even-share fallback is kept verbatim; only max_length changed.",
            "rows": {"source_train_rows": len(train_rows),
                     "calibration_rows": len(calib_rows),
                     "effective_train_rows": len(fit_rows),
                     "after_world_size_alignment": len(train_items_all)},
            "calibration": {"seed": CALIB_SEED, "indices": manifest["calibration"]["indices"],
                            "ids": manifest["calibration"]["ids"]},
            "split_by_rank": {str(r): len(train_items_all[r::world_size])
                              for r in range(world_size)},
            "hyperparameters": {"epochs": epochs, "micro_batch": MICRO_BATCH,
                                "grad_accum": GRAD_ACCUM, "group_size": GROUP_SIZE,
                                "lr_encoder": LR_ENCODER, "lr_head": LR_HEAD,
                                "weight_decay": WEIGHT_DECAY, "sigma_start": SIGMA_START,
                                "sigma_end": SIGMA_END, "ce_weight": CE_WEIGHT,
                                "w_sph": W_SPH, "w_rps": W_RPS, "clip": CLIP,
                                "total_updates": total_updates},
            "runtime": {"device": str(device), "world_size": world_size,
                        "amp_dtype": amp_name, "autocast": str(amp_dtype) if amp_dtype else None,
                        "grad_scaler": use_scaler, "seed_base": SEED_BASE,
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
    manifest = read_manifest(Path(args.data))
    instructions = manifest.get("instructions", DEFAULT_INSTRUCTIONS)
    train_rows = load_bundle("train", Path(args.data))
    calib_rows, _ = frozen_calibration(train_rows, manifest)
    if args.smoke:
        calib_rows = calib_rows[:SMOKE_CALIB_ROWS]

    device = torch.device("cpu" if not torch.cuda.is_available() or args.smoke else "cuda")
    model_dir = base_model_dir()
    cfg = read_cfg(model_dir, train=False)
    tok = load_tokenizer(model_dir)
    model, _ = model_from_checkpoint(ckpt, model_dir, device)

    pairs = collect_fit_pairs(model, tok, calib_rows, cfg, instructions, device)
    k = len(calib_rows[0]["order"])
    raw, written = fit_one_temp(pairs, k)
    base_cfg = json.loads((model_dir / "rl_agent_config.json").read_text(encoding="utf-8"))
    base_temp = list(base_cfg.get("temperature", [1.0, 1.0, 1.0]))
    temperature = list(base_temp)
    temperature[C.QTYPES["choice"]] = written
    payload = {
        "ckpt": str(ckpt), "qtype": "choice", "qt_index": C.QTYPES["choice"],
        "calibration_rows": len(calib_rows), "calibration_seed": CALIB_SEED,
        "raw_fitted_value": raw, "final_written_value": written,
        "clamp": [C.TEMP_MIN, C.TEMP_MAX],
        "base_temperature": base_temp, "temperature": temperature,
        "fallback": FALLBACK_TEMP, "smoke": args.smoke,
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
    manifest = read_manifest(Path(args.data))
    instructions = manifest.get("instructions", DEFAULT_INSTRUCTIONS)
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
    exported["model_name"] = "layachoice-v1"
    exported["layachoice"] = {"checkpoint": str(ckpt), "option_max_tokens": OPTION_MAX_TOKENS,
                              "head_max_len": HEAD_MAX_LEN, "max_len": MAX_LEN,
                              "calibration": str(temp_file) if temp_file.exists() else None,
                              "export_dtype": args.export_dtype}
    (out / "rl_agent_config.json").write_text(
        json.dumps(exported, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Reload through the public entry point. This is the only check that proves the
    # directory is Agent-loadable and that the temperature survives a real reload:
    # the fitted value, the exported config and the Agent's effective value are
    # read back from three different places and compared.
    install()
    import laya
    agent = laya.Agent(str(out.resolve()), device="cpu")
    reloaded = list(agent.temperature)
    if reloaded[C.QTYPES["choice"]] != C.clamp_temperature(temperature[C.QTYPES["choice"]]):
        raise RuntimeError(f"Agent reloaded temperature {reloaded} != exported {temperature}")
    rows = load_bundle("test", Path(args.data), limit=5)
    questions = {QID: {"type": "choice", "instructions": instructions,
                       "criteria": dict(zip(r["order"], r["options"]))} for r in rows[:1]}
    answers = agent.predict_batch([r["state"] for r in rows], questions,
                                  batch_size=5, max_len=cfg.get("max_len", MAX_LEN),
                                  head_max_len=cfg.get("head_max_len", HEAD_MAX_LEN))
    smoke = {"export_dir": str(out), "export_dtype": args.export_dtype,
             "size_bytes": (out / "model.safetensors").stat().st_size,
             "agent_temperature": reloaded, "rows": []}
    for row, res in zip(rows, answers):
        ans = res["answers"][QID]
        smoke["rows"].append({"id": row["id"], "gold_cap": row["order"][row["gold"]],
                              "agent_choice": ans["choice"], "hit": ans["choice"] == row["order"][row["gold"]],
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
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--out", default=None,
                    help="run directory (--train) or deployment directory (--export); "
                         "defaults to scripts/laya_finetune/out, and to <ckpt>/../calibration "
                         "for --fit-temperature")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
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
