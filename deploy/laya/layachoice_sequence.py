"""Self-contained 256-token option builder for the LayaChoice inference service.

Delveta-LayaChoice-v1 was fine-tuned and audited under a per-option token cap of
256, but stock ``laya.common.build_sequence`` hard-codes ``max_length=48`` — which
would truncate EVERY frozen ``B_noprov`` option (all 2568 are 136-230 tokens).
The serving process must therefore install the 256-cap builder before it encodes
anything.

This module is a DELIBERATE, frozen copy of the builder the training/export path
used (``scripts/laya_finetune/layachoice_finetune.py``): the service must not
import from the fine-tuning script, so the two cannot share code. Only the source
of ``OPTION_MAX_TOKENS``/``HEAD_MAX_LEN``/``MAX_LEN`` differs — here they are
module constants, there they are the training constants. The body of
``build_sequence`` is byte-for-byte the stock ``laya.common`` function with the
single change ``max_length=48`` -> ``max_length=OPTION_MAX_TOKENS``; the
``opt_budget < 16`` even-share fallback and the ``head_ids[: max(8, opt_budget)]``
floor are KEPT verbatim — the fallback is what stops a budget too tight for the
options from silently cutting the instruction head down to 8 tokens.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Union

import laya.common as C

# The exported checkpoint's own input configuration (rl_agent_config.json:
# max_len 1024 / head_max_len 768). The model only ever needs at most three
# options per turn (business-layer K normalization, §26.2), so 768 comfortably
# covers head + 3x256.
OPTION_MAX_TOKENS = 256
HEAD_MAX_LEN = 768
MAX_LEN = 1024


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

    Identical to stock ``laya.common.build_sequence`` except the per-option hard
    cap is ``OPTION_MAX_TOKENS`` (256) instead of 48.
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
    return ids, markers, {
        "options": len(opt_ids),
        "options_distinct": len({tuple(o) for o in opt_ids}),
        "tokens_per_option": per_option,
    }


def install() -> None:
    """Point ``laya.agent``'s build_sequence call site at the 256-cap builder.

    ``laya.agent`` binds ``build_sequence`` at import time, so the patch must land
    on the module attribute AFTER ``laya.agent`` is imported and BEFORE any
    ``predict``/``system_one`` call.
    """
    import laya.agent as A

    A.build_sequence = build_sequence
