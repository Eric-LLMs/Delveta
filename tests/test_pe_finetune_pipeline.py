"""B2 — the PE fine-tune TRAINING PIPELINE (scripts only; no torch, no model).

This suite pins the script-level contract of the formal training chain under
``scripts/qwen_parameter_extraction_finetune/`` WITHOUT importing torch /
transformers / peft (the main test env has none of them):

* ``pe_config``  — the config loads and the two path conventions resolve;
* ``sft_data``   — the ONE render + mask + parse protocol (driven through a FAKE
  tokenizer, so the masking/prefix/max-len contracts are pinned with no model);
* ``train``      — the pure planning helpers (optimizer schedule, resume resolution)
  and the frozen-bundle gate;
* ``evaluate``   — the task-metric scoring + the explicit checkpoint-selection rule;
* ``export``     — the artifact guards and the GGUF plan that must NEVER execute.

A real checkpoint / GPU run is deliberately out of scope here.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

_SCRIPTS = (Path(__file__).resolve().parents[1] / "scripts"
            / "qwen_parameter_extraction_finetune")
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import evaluate as E  # noqa: E402
import export as X  # noqa: E402
import pe_config as pcfg  # noqa: E402
import pe_contract as C  # noqa: E402
import sft_data as D  # noqa: E402
import train as T  # noqa: E402

_HAS_BUNDLE = (_SCRIPTS / "data" / "train.jsonl").is_file()


# ── a fake tokenizer: char ids guarantee the prefix property, no model needed ─────


class FakeTok:
    chat_template = "fake"

    def apply_chat_template(self, msgs, tools=None, tokenize=False,
                            add_generation_prompt=False, tool_choice=None):
        parts = []
        for m in msgs:
            parts.append(f"<{m['role']}>" + m.get("content", ""))
            for tc in (m.get("tool_calls") or []):
                fn = tc["function"]
                parts.append("<tool_call>" + json.dumps(
                    {"name": fn["name"], "arguments": fn["arguments"]}) + "</tool_call>")
        text = "".join(parts)
        return text + "<assistant>" if add_generation_prompt else text

    def __call__(self, text, add_special_tokens=False):
        return types.SimpleNamespace(input_ids=[ord(c) for c in text])

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


class BadTok(FakeTok):
    def apply_chat_template(self, *a, **kw):
        return super().apply_chat_template(*a, **kw) + "TRAILING"  # breaks prefix-stability


def _sample(cap="cap-web-search", args=None):
    args = args if args is not None else {"query": "q"}
    tools = [{"type": "function", "function": {
        "name": cap, "description": "d", "parameters": C.model_schema(cap)}}]
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
            "name": cap, "arguments": json.dumps(args)}}]},
    ]
    return {"id": "s1", "capability_id": cap, "tools": tools, "messages": msgs}


# ── pe_config: load + the two path conventions ───────────────────────────────────


def test_config_loads_and_resolves_both_conventions():
    cfg = pcfg.load_config()
    for section in ("source", "output", "build", "model", "train", "eval", "export"):
        assert section in cfg, f"config missing section {section!r}"
    # source.* is repo-root-relative; everything else is script-dir-relative.
    assert pcfg.source_path(cfg, "formal500").is_absolute()
    assert str(pcfg.source_path(cfg, "formal500")).endswith("dataset_formal500.jsonl")
    local = pcfg.local_path(cfg, "train", "output_dir")
    assert local.parent == pcfg.SCRIPT_DIR


def test_config_missing_key_raises():
    with pytest.raises(pcfg.ConfigError):
        pcfg.local_path({"train": {}}, "train", "nope")
    with pytest.raises(pcfg.ConfigError):
        pcfg.source_path({"source": {}}, "nope")


def test_unknown_section_is_not_treated_as_source():
    with pytest.raises(pcfg.ConfigError):
        pcfg.local_path({"source": {"x": "1"}}, "source", "x")


# ── sft_data: the ONE protocol (fake tokenizer) ───────────────────────────────────


def test_build_example_masks_the_prompt_and_supervises_the_tool_call():
    ex = D.build_example(_sample(), FakeTok())
    assert ex["cap"] == "cap-web-search"
    assert ex["prompt_len"] == sum(1 for t in ex["labels"] if t == D.IGNORE_INDEX)
    supervised = [t for t in ex["labels"] if t != D.IGNORE_INDEX]
    assert supervised and supervised == ex["input_ids"][ex["prompt_len"]:]
    assert "<tool_call>" in ex["target_text"] and "cap-web-search" in ex["target_text"]


def test_build_example_rejects_a_non_prefix_stable_template():
    with pytest.raises(D.SftDataError):
        D.build_example(_sample(), BadTok())


def test_build_example_never_truncates_past_max_seq_len():
    with pytest.raises(D.SftDataError):
        D.build_example(_sample(), FakeTok(), max_seq_len=1)


def test_load_split_unknown_split_is_refused():
    with pytest.raises(D.SftDataError):
        D.load_split(_SCRIPTS / "data", "train2")


def test_formal500_is_refused():
    with pytest.raises(D.SftDataError):
        D.refuse_formal500("/x/formal500/train.jsonl")
    D.refuse_formal500("/x/train.jsonl")  # no error when absent


def test_gold_target_reads_the_arguments_json_string():
    assert D.gold_target(_sample(args={"query": "x", "top_k": 3})) == {"query": "x", "top_k": 3}


def test_parse_tool_call_text_ok_and_failures():
    name, args = D.parse_tool_call_text(
        '<tool_call>{"name": "cap-web-search", "arguments": {"query": "q"}}</tool_call>')
    assert name == "cap-web-search" and args == {"query": "q"}
    for bad in ("no block here", '<tool_call>{oops}</tool_call>',
                '<tool_call>{"name": "x", "arguments": [1,2]}</tool_call>'):
        with pytest.raises(D.SftDataError):
            D.parse_tool_call_text(bad)


def test_collate_batch_right_pads_ids_and_masks_labels():
    a = {"input_ids": [1, 2, 3], "labels": [1, 2, 3]}
    b = {"input_ids": [4], "labels": [4]}
    batch = D.collate_batch([a, b], pad_token_id=0)
    assert batch["input_ids"] == [[1, 2, 3], [4, 0, 0]]
    assert batch["labels"] == [[1, 2, 3], [4, D.IGNORE_INDEX, D.IGNORE_INDEX]]
    assert batch["attention_mask"] == [[1, 1, 1], [1, 0, 0]]


# ── train: pure planning helpers + the frozen-bundle gate ─────────────────────────


def test_plan_steps_converts_warmup_ratio_to_steps():
    cfg = pcfg.load_config()
    plan = T.plan_steps(cfg, 8679)
    assert plan["effective_batch"] == 16          # 8 * 2
    assert plan["steps_per_epoch"] == 543         # ceil(8679 / 16)
    assert plan["total_optimizer_steps"] == 1629  # * 3 epochs
    assert plan["warmup_steps"] == 49             # ceil(0.03 * 1629)


def test_resolve_resume_fresh_auto_and_missing(tmp_path, monkeypatch):
    cfg = pcfg.load_config()
    monkeypatch.setattr(pcfg, "SCRIPT_DIR", tmp_path)
    assert T.resolve_resume(cfg, None)[1] == "fresh"
    with pytest.raises(pcfg.ConfigError):
        T.resolve_resume(cfg, "auto")             # no checkpoints yet
    for step in (200, 1000, 400):
        (tmp_path / "checkpoints" / f"checkpoint-{step}").mkdir(parents=True)
    path, how = T.resolve_resume(cfg, "auto")
    assert how == "auto(latest)" and path.name == "checkpoint-1000"
    with pytest.raises(pcfg.ConfigError):
        T.resolve_resume(cfg, str(tmp_path / "nope"))


def test_train_modules_import_no_torch():
    for mod in (T, E, X):
        assert getattr(mod, "torch", None) is None, f"{mod.__name__} imported torch at top level"


@pytest.mark.skipif(not _HAS_BUNDLE, reason="frozen PE bundle not present")
def test_check_contract_gates_the_frozen_bundle():
    cfg = pcfg.load_config()
    summary = T.check_contract(cfg)
    assert summary["hash_verified"] is True
    assert summary["train_rows"] == 8679
    assert summary["validation_rows"] == 1041
    assert set(summary["by_capability"]) == set(C.SCHEMA)


# ── evaluate: scoring + the selection rule ────────────────────────────────────────


def test_canon_value_normalizes_int_slots():
    assert E.canon_value("top_k", "3") == E.canon_value("top_k", 3) == "3"
    assert E.canon_value("top_k", 3.0) == "3"
    assert E.canon_value("query", "  hi ") == "hi"


def test_score_sample_exact_match_and_illegal_keys():
    rec = E.score_sample("cap-web-search", {"query": "a", "top_k": 3},
                         "cap-web-search", {"query": "a", "top_k": "3", "platform": "x"})
    assert rec["exact_match"] is True            # illegal key does not break exact match
    assert rec["illegal_keys"] == ["platform"]
    assert rec["required_present"] == 1 and rec["required_correct"] == 1

    miss = E.score_sample("cap-create-folder", {"name": "n"}, "cap-create-folder", {})
    assert miss["exact_match"] is False and miss["required_present"] == 0


def test_aggregate_reports_parse_rate_and_groups():
    good = {"record": E.score_sample("cap-create-folder", {"name": "n"},
                                     "cap-create-folder", {"name": "n"}), "parse_ok": True}
    bad = {"record": E.score_sample("cap-create-folder", {"name": "n"},
                                    "cap-create-folder", {}), "parse_ok": False}
    m = E.aggregate([good, bad])
    assert m["overall"]["parse_rate"] == 0.5
    assert m["overall"]["required_missing_rate"] == 0.5
    assert m["by_capability"]["cap-create-folder"]["samples"] == 2


def _cand(name, step, exact, missing, field):
    return {"checkpoint": name, "step": step,
            "metrics": {"overall": {"exact_match": exact,
                                    "required_missing_rate": missing,
                                    "field_accuracy": field}}}


def test_select_best_uses_the_task_metric_then_tiebreaks():
    a = _cand("a", 100, 0.90, 0.02, 0.95)
    b = _cand("b", 200, 0.90, 0.01, 0.90)   # same exact_match, fewer required-missing
    c = _cand("c", 50, 0.80, 0.00, 0.99)
    assert E.select_best([a, b, c], "exact_match", "max")["checkpoint"] == "b"
    # min mode is honoured (lower is better)
    assert E.select_best([a, b, c], "exact_match", "min")["checkpoint"] == "c"


def test_select_best_rejects_unknown_metric_and_empty():
    with pytest.raises(pcfg.ConfigError):
        E.select_best([_cand("a", 1, 1.0, 0.0, 1.0)], "nope", "max")
    with pytest.raises(pcfg.ConfigError):
        E.select_best([], "exact_match", "max")


def test_evaluate_refuses_the_frozen_test_split():
    cfg = pcfg.load_config()
    with pytest.raises(pcfg.ConfigError):
        E.run_eval(cfg, split="test", allow_test=False, checkpoint=None,
                   base_only=True, limit=1, write=False)


# ── export: guards + the GGUF plan that must not execute ─────────────────────────


def test_guard_target_refuses_a_nonempty_dir(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    (d / "keep.txt").write_text("x", encoding="utf-8")
    with pytest.raises(pcfg.ConfigError):
        X._guard_target(d, force=False)
    X._guard_target(d, force=True)          # allowed with --force
    X._guard_target(tmp_path / "absent", force=False)   # absent is fine


def test_guard_target_refuses_formal500():
    with pytest.raises(D.SftDataError):
        X._guard_target(Path("/x/formal500/out"), force=True)


def test_gguf_plan_is_never_executed():
    cfg = pcfg.load_config()
    plan = X.gguf_plan(cfg, None)
    assert plan["executed"] is False
    assert plan["kind"] == "gguf_plan"
    assert any("convert_hf_to_gguf" in s for s in plan["steps"])
    assert any(cfg["export"]["gguf"]["quant"] in s for s in plan["steps"])
    assert "NOT overwritten" in plan["note"] or "not overwritten" in plan["note"]


def test_selected_checkpoint_reads_selection_json(tmp_path, monkeypatch):
    cfg = pcfg.load_config()
    monkeypatch.setattr(pcfg, "SCRIPT_DIR", tmp_path)
    ckpt = tmp_path / "checkpoints" / "checkpoint-400"
    ckpt.mkdir(parents=True)
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "selection.json").write_text(json.dumps(
        {"selection": {"checkpoint": str(ckpt)}}), encoding="utf-8")
    assert X.selected_checkpoint(cfg, None) == ckpt
    assert X.selected_checkpoint(cfg, str(ckpt)) == ckpt
    with pytest.raises(pcfg.ConfigError):
        X.selected_checkpoint(cfg, str(tmp_path / "missing"))


def test_export_preflight_checks_the_adapter_base(tmp_path):
    cfg = pcfg.load_config()
    adapter = tmp_path / "best_adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps(
        {"base_model_name_or_path": cfg["model"]["base"]}), encoding="utf-8")
    checks = X.preflight(cfg, adapter)
    by_name = {c["check"]: c for c in checks}
    assert by_name["adapter_config.json present"]["ok"] is True
    assert by_name["adapter base matches the pinned model"]["ok"] is True
    assert by_name["revision pinned"]["ok"] is True

    (adapter / "adapter_config.json").write_text(json.dumps(
        {"base_model_name_or_path": "someone/else"}), encoding="utf-8")
    by_name = {c["check"]: c for c in X.preflight(cfg, adapter)}
    assert by_name["adapter base matches the pinned model"]["ok"] is False
