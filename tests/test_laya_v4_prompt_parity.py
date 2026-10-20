"""V4 model-input parity: training rows vs the real inference call chain.

The final criterion is the token sequence Laya actually receives, not JSON
similarity. This test takes V2 dynamic-option rows that passed the V2 adapter, builds the
training item with the real V2 ``build_item``, and compares it byte-for-byte
with the sequence produced by the serving side:

  dataset row -> production ``render_choice_question`` -> ``_to_internal``
              -> ``deploy/laya/layachoice_sequence.build_sequence``

The inference side appends REJECT last; training rows are shuffled. Therefore
the exact sequence is compared in the row's presented order (the only legal
difference), and the production-order question is checked to carry the same
instructions, the same cards and the same option count. A REJECT-last row is
also compared in exact production order.

Skipped when ``laya`` is not installed (the checkpoint's own environment).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parents[1]
_V2 = _REPO / "fine_tuning" / "laya_finetune" / "V2"
_VERSIONS = _REPO / "fine_tuning" / "laya_finetune" / "data_versions"
for _p in (_V2, _REPO / "packages", _REPO / "deploy" / "laya"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# `layachoice_v2_render` imports `laya.common` at module level, so the skip must
# happen before that import or collection itself errors.
if importlib.util.find_spec("laya") is None:
    pytest.skip("laya is only installed in .venv-laya", allow_module_level=True)

import layachoice_v2_dataset as D  # noqa: E402
import layachoice_v2_render as R  # noqa: E402
import layachoice_v2_spec as S  # noqa: E402
try:  # the renderer and its prompt helper are dependency-free
    from core.application.chat.intent_funnel.cap_router.card_renderer import (
        INSTRUCTIONS,
        REJECT_CARD,
        REJECT_LABEL,
        render_choice_question,
    )
except ModuleNotFoundError:
    # The checkpoint env (laya) does not carry the application's DB/config
    # dependencies. Import the same source files without executing unrelated
    # package __init__ side effects (funnel -> orchestrator -> config).
    import importlib
    from types import ModuleType

    _CHAIN = (
        "core", "core.application", "core.application.chat",
        "core.application.chat.intent_funnel",
        "core.application.chat.intent_funnel.tool_intent",
        "core.application.chat.intent_funnel.cap_router",
    )
    for _name in [m for m in sys.modules
                  if m == "core" or m.startswith("core.")]:
        del sys.modules[_name]
    for _name in _CHAIN:
        _stub = ModuleType(_name)
        _stub.__path__ = [
            str(_REPO.joinpath("packages", *_name.split(".")))]
        _stub.__package__ = _name
        sys.modules[_name] = _stub
    _RENDERER = importlib.import_module(
        "core.application.chat.intent_funnel.cap_router.card_renderer")
    INSTRUCTIONS = _RENDERER.INSTRUCTIONS
    REJECT_CARD = _RENDERER.REJECT_CARD
    REJECT_LABEL = _RENDERER.REJECT_LABEL
    render_choice_question = _RENDERER.render_choice_question

SNAPSHOT = _REPO / "tests" / "fixtures" / "laya_v4_registry_snapshot.json"
TAG = "recall_threshold_085"
_BUNDLE = _VERSIONS / TAG

pytestmark = [
    pytest.mark.skipif(importlib.util.find_spec("laya") is None,
                       reason="laya is only installed in .venv-laya"),
    pytest.mark.skipif(
        not (_BUNDLE / "manifest.json").is_file(),
        reason="V2 dynamic-option bundle not present"),
]


def _rows() -> list[dict]:
    path = _BUNDLE / "v2_train.jsonl"
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _entries() -> dict[str, SimpleNamespace]:
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    return {
        cid: SimpleNamespace(
            capability_id=e["capability_id"],
            tool_binding=e["tool_binding"],
            description=e["description"],
            negatives=e["negatives"],
            parameters=e["parameters"],
        )
        for cid, e in snapshot.items()
    }


@pytest.fixture(scope="module")
def tokenizer():
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    import laya.agent as A

    try:
        root = Path(snapshot_download(
            repo_id=S.BASE_REPO, revision=S.BASE_REVISION,
            allow_patterns=[f"{S.BASE_SUBFOLDER}/tokenizer/*"],
            local_files_only=True))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"pinned tokenizer not cached: {exc!r}")
    model_dir = root / S.BASE_SUBFOLDER
    A._fix_tokenizer_config(str(model_dir))
    return AutoTokenizer.from_pretrained(str(model_dir / "tokenizer"))


def _one_row_per_k() -> list[dict]:
    by_k: dict[int, dict] = {}
    for row in _rows():
        by_k.setdefault(len(row["input"]["options"]) - 1, row)
    missing = sorted(set(range(4)) - set(by_k))
    if missing:
        pytest.skip(f"smoke train split lacks K={missing}")
    return [by_k[k] for k in range(4)]


def test_training_and_serving_build_the_same_sequence(tokenizer) -> None:
    import laya.agent as A
    import layachoice_sequence as SEQ

    entries = _entries()
    for row in _one_row_per_k():
        ex = D.adapt_row(row, "train")

        # ── the real training item ─────────────────────────────────────────
        item = R.build_item(tokenizer, ex, INSTRUCTIONS,
                            S.MAX_LEN, S.HEAD_MAX_LEN)
        assert item["label"] == row["label"]
        assert len(item["markers"]) == len(ex.order)
        assert len(item["target"]) == len(item["markers"])
        assert sum(item["target"]) == 1.0
        assert item["target"][row["label"]] == 1.0

        # ── the real production question ───────────────────────────────────
        cand_ids = [cid for cid in ex.order if cid != REJECT_LABEL]
        q_prod = render_choice_question(
            [SimpleNamespace(capability_id=cid) for cid in cand_ids], entries)
        assert q_prod["instructions"] == INSTRUCTIONS
        assert list(q_prod["criteria"]) == cand_ids + [REJECT_LABEL]
        for cid in cand_ids:
            assert q_prod["criteria"][cid] == ex.options[ex.order.index(cid)]
        assert q_prod["criteria"][REJECT_LABEL] == REJECT_CARD

        internal = A.Agent._to_internal(q_prod)
        ids_prod, markers_prod = SEQ.build_sequence(
            tokenizer, ex.state, internal, S.MAX_LEN, S.HEAD_MAX_LEN)
        assert len(markers_prod) == len(ex.order)
        assert len(ids_prod) <= S.MAX_LEN

        # Same question, row's presented order: the ONLY difference from the
        # production question is where the shuffled REJECT card sits.
        q_row = {"type": "choice", "instructions": INSTRUCTIONS,
                 "criteria": dict(zip(ex.order, ex.options))}
        ids_row, markers_row = SEQ.build_sequence(
            tokenizer, ex.state, A.Agent._to_internal(q_row),
            S.MAX_LEN, S.HEAD_MAX_LEN)
        assert ids_row == item["ids"], f"{row['message_id']}: serving sequence drift"
        assert markers_row == item["markers"]
        assert len(markers_row) == len(ex.order)

        # If REJECT is already last, the production order is byte-identical.
        if ex.order[-1] == REJECT_LABEL:
            assert ids_prod == item["ids"], f"{row['message_id']}: production order drift"
            assert markers_prod == item["markers"]


def test_manifest_instructions_are_the_production_constant() -> None:
    manifest = json.loads(
        (_BUNDLE / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["instructions"] == INSTRUCTIONS
    assert manifest["reject"]["card"] == REJECT_CARD
    assert manifest["formal_card_view"] == "B_noprov"
