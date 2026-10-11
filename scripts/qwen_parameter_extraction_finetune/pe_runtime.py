#!/usr/bin/env python3
"""Torch-side runtime helpers shared by the PE train / evaluate / export stages.

Device choice, compute dtype, base-checkpoint loading and checkpoint discovery live
HERE so the three stages cannot disagree about what "the base model at the pinned
revision" or "the latest checkpoint" means. ``torch`` / ``transformers`` / ``peft``
are imported INSIDE the functions, so this module imports cleanly in a bare
environment and the pure helpers stay unit-testable without a GPU stack.
"""
from __future__ import annotations

import os
from pathlib import Path

import pe_config as cfgmod

# The HF cache stays inside this directory for EVERY stage that loads a model
# (train / evaluate / export), so they share one cache and never touch ~/.cache.
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent / ".hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def pick_device(torch) -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(torch, cfg: dict, device: str):
    """bf16 only when the device can actually do it; CPU falls back to fp32."""
    want = str(cfg["model"].get("torch_dtype", "bfloat16")).lower()
    if want in {"bfloat16", "bf16"} and device == "cuda":
        return torch.bfloat16
    if want in {"float16", "fp16", "half"} and device == "cuda":
        return torch.float16
    return torch.float32


def _is_cached(base: str, revision: str) -> bool:
    """True when the pinned revision's snapshot is already in the local HF cache.
    Probed with ``config.json`` — present in every full repo snapshot."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:  # noqa: BLE001 — no hub client -> treat as uncached
        return False
    try:
        hit = try_to_load_from_cache(base, "config.json", revision=revision)
        return isinstance(hit, str)
    except Exception:  # noqa: BLE001
        return False


def from_pretrained_cached(cls, base: str, revision: str, **kwargs):
    """Load from the local cache when the pinned revision is present (no network);
    otherwise fall through to a normal (downloading) load."""
    if _is_cached(base, revision):
        return cls.from_pretrained(base, revision=revision, local_files_only=True, **kwargs)
    return cls.from_pretrained(base, revision=revision, **kwargs)


def load_base_model(model_cls, base: str, revision: str, dtype):
    """Pass the compute dtype under the name this transformers version expects."""
    try:
        return from_pretrained_cached(model_cls, base, revision, dtype=dtype)
    except TypeError:
        return from_pretrained_cached(model_cls, base, revision, torch_dtype=dtype)


def load_tokenizer(tok_cls, base: str, revision: str):
    tk = from_pretrained_cached(tok_cls, base, revision)
    if getattr(tk, "pad_token_id", None) is None:
        tk.pad_token = tk.eos_token
    return tk


def checkpoint_dirs(out_dir: Path) -> list[Path]:
    """``checkpoint-<step>`` dirs under ``out_dir``, ordered by step ascending."""
    if not out_dir.is_dir():
        return []

    def step(p: Path) -> int:
        try:
            return int(p.name.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            return -1

    dirs = [p for p in out_dir.glob("checkpoint-*") if p.is_dir() and step(p) >= 0]
    return sorted(dirs, key=step)


def checkpoint_step(path: Path) -> int:
    try:
        return int(Path(path).name.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return -1


def train_output_dir(cfg: dict) -> Path:
    return cfgmod.local_path(cfg, "train", "output_dir")
