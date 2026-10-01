"""deploy/laya sidecar — the self-contained 256-token option builder.

Delveta-LayaChoice-v1 was fine-tuned and audited under a per-option cap of 256,
but stock ``laya.common.build_sequence`` hard-codes ``max_length=48`` — which
would truncate EVERY frozen option. The sidecar therefore ships its OWN builder
and installs it on ``laya.agent`` before the model encodes anything.

The structural tests here run anywhere (AST-only, no ``laya`` import). The
behavioral install test is skipped unless ``laya`` is importable (it is only
present in the ``.venv-laya`` environment the checkpoint was validated in).
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

SEQUENCE = (Path(__file__).resolve().parents[1]
            / "deploy" / "laya" / "layachoice_sequence.py")


def _module_ast() -> ast.Module:
    return ast.parse(SEQUENCE.read_text(encoding="utf-8"))


def _const(name: str):
    for node in _module_ast().body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {SEQUENCE.name}")


def test_option_cap_is_256_not_stock_48():
    assert _const("OPTION_MAX_TOKENS") == 256
    assert _const("HEAD_MAX_LEN") == 768
    assert _const("MAX_LEN") == 1024


def test_build_sequence_caps_options_at_the_256_constant():
    fn = next(n for n in _module_ast().body
              if isinstance(n, ast.FunctionDef) and n.name == "build_sequence")
    caps = [kw.value.id for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            for kw in n.keywords
            if kw.arg == "max_length" and isinstance(kw.value, ast.Name)]
    assert "OPTION_MAX_TOKENS" in caps
    # the stock hard-coded 48 must be gone
    literals = [kw.value.value for n in ast.walk(fn)
                if isinstance(n, ast.Call)
                for kw in n.keywords
                if kw.arg == "max_length" and isinstance(kw.value, ast.Constant)]
    assert 48 not in literals


def test_install_rebinds_the_agent_call_site():
    fn = next(n for n in _module_ast().body
              if isinstance(n, ast.FunctionDef) and n.name == "install")
    # install() must assign `laya.agent`'s build_sequence attribute
    assigns = [n for n in ast.walk(fn)
               if isinstance(n, ast.Assign)
               and isinstance(n.targets[0], ast.Attribute)
               and n.targets[0].attr == "build_sequence"]
    assert assigns, "install() does not patch an agent.build_sequence attribute"


@pytest.mark.skipif(importlib.util.find_spec("laya") is None,
                    reason="laya not importable (only in .venv-laya)")
def test_install_patches_laya_agent_build_sequence():
    import laya.agent as A
    spec = importlib.util.spec_from_file_location("layachoice_sequence", SEQUENCE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    stock = A.build_sequence
    mod.install()
    try:
        assert A.build_sequence is mod.build_sequence
        assert A.build_sequence is not stock
        assert mod.OPTION_MAX_TOKENS == 256
    finally:
        A.build_sequence = stock        # never leak the patch to other tests
