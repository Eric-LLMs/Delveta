#!/usr/bin/env python3
"""The ONE SFT data + rendering + masking protocol for the PE fine-tune.

The formal trainer (:mod:`train`), the evaluator (:mod:`evaluate`) and the CPU
smoke test (:mod:`smoke_train`) all build their examples HERE, so there is a
single training protocol and it cannot drift between stages.

The recipe is the smoke-verified one, unchanged:

* render the prompt as the chat template with ``tools`` + ``tool_choice="required"``
  and ``add_generation_prompt=True``, and the full turn with
  ``add_generation_prompt=False``;
* require ``full_text.startswith(prompt_text)`` (prefix-stable) and that the
  prompt's token ids are an exact prefix of the full ids (no boundary merge);
* ``labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]`` — system and
  user tokens are masked; ONLY the assistant tool-call turn is supervised.

Import-safe without torch / transformers: the tokenizer is INJECTED, never
imported, so config checks and data-contract tests run in a bare env.

Target contract (frozen dataset): an assistant turn whose ``content`` is empty and
whose ``tool_calls[0].function`` carries ``name == capability_id`` and
``arguments`` = a JSON STRING of the flat ``{slot: value}`` map.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

TRAIN_SPLIT = "train"
VAL_SPLIT = "validation"
TEST_SPLIT = "test"
SPLITS = (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT)

IGNORE_INDEX = -100

# The model's native tool-call text wrapping (what the chat template emits and the
# base checkpoint was trained to produce).
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


class SftDataError(Exception):
    """A data / rendering contract violation. Fail loud — never train on a
    silently corrupted example."""


# ── loading + boundary guards ─────────────────────────────────────────────────────


def refuse_formal500(path) -> None:
    """Formal-500 is fully isolated: any path naming it is refused everywhere."""
    if "formal500" in str(path).lower():
        raise SftDataError(
            f"refusing {path}: Formal-500 is a frozen external benchmark, never a "
            f"data / tuning / checkpoint-selection source")


def load_rows(path, *, limit: int | None = None) -> list[dict]:
    p = Path(path)
    if not p.is_file():
        raise SftDataError(f"missing data file: {p}")
    rows = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not rows:
        raise SftDataError(f"no rows in {p}")
    return rows[:limit] if limit else rows


def load_split(data_dir, split: str, *, limit: int | None = None) -> list[dict]:
    if split not in SPLITS:
        raise SftDataError(f"unknown split {split!r} (want one of {SPLITS})")
    path = Path(data_dir) / f"{split}.jsonl"
    refuse_formal500(path)
    return load_rows(path, limit=limit)


# ── rendering + masked-label construction (the single protocol) ───────────────────


def render_prompt(tokenizer, msgs, tools, *, gen: bool = True) -> str:
    """Render system+user (optionally priming the assistant turn)."""
    kw = {"tools": tools, "tokenize": False, "add_generation_prompt": gen}
    try:
        return tokenizer.apply_chat_template(msgs, tool_choice="required", **kw)
    except TypeError:
        # Some templates reject the unknown kwarg — it is an API directive, not text.
        return tokenizer.apply_chat_template(msgs, **kw)


def render_full(tokenizer, msgs, tools) -> str:
    return tokenizer.apply_chat_template(msgs, tools=tools, tokenize=False,
                                         add_generation_prompt=False)


def build_example(sample: dict, tokenizer, *, max_seq_len: int | None = None) -> dict:
    """Return ``{input_ids, labels, prompt_len, target_text, cap, tools, sample}``
    with the system+user prompt masked. Raises :class:`SftDataError` when the
    template is not prefix-stable, the tokenizer merges across the assistant
    boundary, or the example exceeds ``max_seq_len`` (never silently truncated —
    that could drop the supervised target)."""
    msgs = sample["messages"]
    tools = sample["tools"]
    prompt_text = render_prompt(tokenizer, msgs[:2], tools, gen=True)
    full_text = render_full(tokenizer, msgs, tools)
    if not full_text.startswith(prompt_text):
        raise SftDataError("chat template is not prefix-stable (prompt != prefix of full render)")
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
    if list(full_ids[: len(prompt_ids)]) != list(prompt_ids):
        raise SftDataError("tokenizer merged across the assistant boundary; mask would be inexact")
    if max_seq_len is not None and len(full_ids) > int(max_seq_len):
        raise SftDataError(
            f"example {sample.get('id')!r} is {len(full_ids)} tokens > max_seq_len "
            f"{max_seq_len}; raise max_seq_len (do not truncate away the target)")
    labels = [IGNORE_INDEX] * len(prompt_ids) + list(full_ids[len(prompt_ids):])
    target_text = tokenizer.decode([i for i in labels if i != IGNORE_INDEX],
                                   skip_special_tokens=False)
    return {
        "input_ids": list(full_ids), "labels": labels, "prompt_len": len(prompt_ids),
        "target_text": target_text, "cap": sample["capability_id"], "tools": tools,
        "sample": sample,
    }


def collate_batch(examples: list[dict], *, pad_token_id: int) -> dict:
    """Right-pad a batch: input ids with ``pad_token_id``, labels with -100,
    attention mask 0 on the pads. (Examples are already length-checked upstream.)"""
    width = max(len(e["input_ids"]) for e in examples)
    input_ids, labels, attention = [], [], []
    for e in examples:
        ids = list(e["input_ids"])[:width]
        lab = list(e["labels"])[:width]
        pad = width - len(ids)
        input_ids.append(ids + [int(pad_token_id)] * pad)
        labels.append(lab + [IGNORE_INDEX] * pad)
        attention.append([1] * len(ids) + [0] * pad)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention}


# ── target parsing (shared by eval + smoke) ───────────────────────────────────────


def gold_target(sample: dict) -> dict:
    """The gold slot map: the assistant tool_call ``arguments`` JSON string."""
    fn = sample["messages"][-1]["tool_calls"][0]["function"]
    return json.loads(fn["arguments"])


def parse_tool_call_text(text: str) -> tuple[str | None, dict]:
    """Parse a generated reply into ``(name, arguments_dict)``. Raises
    :class:`SftDataError` on a missing block / non-JSON body / non-object
    arguments — the evaluator counts these as parse failures."""
    hit = _TOOL_CALL_RE.search(text or "")
    if not hit:
        raise SftDataError("no <tool_call>...</tool_call> block in the generation")
    try:
        obj = json.loads(hit.group(1))
    except ValueError as exc:
        raise SftDataError(f"tool_call body is not valid JSON: {exc}") from exc
    name = obj.get("name")
    args = obj.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError as exc:
            raise SftDataError(f"arguments string is not JSON: {exc}") from exc
    if not isinstance(args, dict):
        raise SftDataError(f"arguments is not a JSON object: {type(args).__name__}")
    return name, args
