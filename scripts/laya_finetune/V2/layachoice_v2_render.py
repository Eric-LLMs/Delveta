#!/usr/bin/env python3
"""LayaChoice-v2 rendering — the model-input boundary.

``build_item`` turns one ``V2Example`` into the exact payload the Laya model
sees:

    [CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1
    [MASK] opt2 [MASK] opt3 [SEP] state [SEP]

with one marker per option. The ONLY content that enters the sequence is the
row's ``state`` and its four option cards (``order`` + ``options`` plus the
manifest ``instructions``). Every other frozen key — ``part``,
``source_query_id``, ``src_sha``, ``shuffle``, ``lang``, ``gold_index``,
``target_kind``, ``gold_capability_id``, ``reject_index`` — is metadata and is
never rendered; the target is derived from ``gold_index`` separately.

``build_sequence`` is a VERBATIM copy of the frozen v1 builder
(``../V1/layachoice_finetune.py``): a copy of ``laya.common.build_sequence`` with
ONE change — the per-option hard cap ``max_length=48`` becomes
``OPTION_MAX_TOKENS`` (256). Keeping the stock ``opt_budget < 16`` even-share
fallback is deliberate; it is what stops a budget too tight for the options from
silently cutting the instruction head to 8 tokens.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import laya.common as C

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import layachoice_v2_spec as S  # noqa: E402
from layachoice_v2_dataset import V2Example  # noqa: E402


# ── the option builder (drop-in patch for laya.agent) ────────────────────────
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
    """Format: [CLS] <type> instructions [SEP] [MASK] opt0 ... [SEP] state [SEP].

    Verbatim copy of the frozen v1 builder (see module docstring) with the
    per-option cap raised from the library's 48 to ``OPTION_MAX_TOKENS``.
    """
    mask_tok = tok.mask_token
    opts = C.render_options(q)
    order = option_order if option_order is not None else list(range(len(opts)))
    ins = str(q["ins"]).replace(mask_tok, " ")
    head_ids = C._encode_question_text(tok, "%s question: %s" % (q["t"], ins), add_special_tokens=False)
    opt_ids = []
    for i in order:
        opt_tokens = C._encode_question_text(
            tok,
            " " + opts[i].replace(mask_tok, " "),
            add_special_tokens=False,
            truncation=True,
            max_length=S.OPTION_MAX_TOKENS,
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
    return ids, markers, {
        "options": len(opt_ids),
        "options_distinct": len({tuple(o) for o in opt_ids}),
        "tokens_per_option": per_option,
    }


def install() -> None:
    """Point laya.agent's build_sequence call site at the 256-cap builder."""
    import laya.agent as A
    A.build_sequence = build_sequence


# ── the model-input boundary ────────────────────────────────────────────────
def question_of(ex: V2Example, instructions: str) -> dict:
    """The frozen B_noprov card view, in the library's compact question shape.

    Reads ONLY ``order``/``options`` from the example — the closed model input.
    """
    return {"t": "choice", "ins": instructions,
            "crit": dict(zip(ex.order, ex.options))}


def build_item(tok, ex: V2Example, instructions: str,
               max_len: int = S.MAX_LEN, head_max_len: int = S.HEAD_MAX_LEN) -> dict:
    """One example -> the tensor payload for the model, plus its one-hot target.

    ``target[gold_index]`` is the single positive slot (the task is flat 4-way:
    3 capability classes + REJECT — REJECT is a normal class, not a separate
    NONE head).
    """
    q = question_of(ex, instructions)
    ids, markers = build_sequence(tok, ex.state, q, max_len, head_max_len)
    if len(markers) != len(ex.options):
        raise RuntimeError(f"{ex.id}: {len(ex.options)} options but {len(markers)} markers "
                           f"survive head_max_len={head_max_len}")
    target = [0.0] * len(markers)
    target[ex.gold_index] = 1.0
    return {"ids": ids, "markers": markers, "qtype": C.QTYPES["choice"],
            "target": target, "label": ex.gold_index, "id": ex.id}
