"""K normalization (§26.2) — the business-layer candidate-count rule.

Pure, backend-agnostic: the SAME rule governs the stub lane and the laya lane,
and ``backend=off`` never enters it. The model's decision head must only ever be
handed at most K options, and a K=1 turn must be executed by the business layer
WITHOUT consulting any model (so no confidence can be fabricated).
"""
from __future__ import annotations

import pytest
from core.application.chat.intent_funnel.candidate_aggregation import (
    K_DEFAULT,
    normalize_top_k,
)
from core.application.chat.intent_funnel.contract import Candidate


def _c(cid, score):
    return Candidate(cid, score, origin="recall")


def test_k_default_is_three():
    assert K_DEFAULT == 3


def test_k0_empty_set_is_neither_direct_nor_selectable():
    n = normalize_top_k([])
    assert n.candidates == ()
    assert n.direct is None
    assert n.is_direct is False


def test_k1_is_direct_and_never_selectable():
    c = _c("cap-a", 0.9)
    n = normalize_top_k([c])
    assert n.direct is c
    assert n.is_direct is True
    assert n.candidates == ()


def test_k2_goes_whole_to_the_selector():
    cs = [_c("cap-a", 0.9), _c("cap-b", 0.7)]
    n = normalize_top_k(cs)
    assert n.direct is None
    assert n.candidates == (cs[0], cs[1])


def test_k3_goes_whole_to_the_selector():
    cs = [_c("cap-a", 0.9), _c("cap-b", 0.8), _c("cap-c", 0.7)]
    n = normalize_top_k(cs)
    assert n.candidates == (cs[0], cs[1], cs[2])


def test_k4_truncates_to_top3_keeping_the_given_order():
    cs = [_c("cap-a", 0.95), _c("cap-b", 0.90),
          _c("cap-c", 0.85), _c("cap-d", 0.80)]
    n = normalize_top_k(cs)
    assert [c.capability_id for c in n.candidates] == ["cap-a", "cap-b", "cap-c"]
    assert n.direct is None


def test_k_many_truncates_to_k():
    cs = [_c(f"cap-{i}", 1.0 - i / 100) for i in range(10)]
    n = normalize_top_k(cs)
    assert len(n.candidates) == K_DEFAULT
    assert [c.capability_id for c in n.candidates] == ["cap-0", "cap-1", "cap-2"]


def test_explicit_k_is_honored():
    cs = [_c("cap-a", 0.9), _c("cap-b", 0.8), _c("cap-c", 0.7)]
    n = normalize_top_k(cs, k=2)
    assert [c.capability_id for c in n.candidates] == ["cap-a", "cap-b"]


def test_normalize_does_not_mutate_its_input():
    cs = [_c("cap-a", 0.9), _c("cap-b", 0.8), _c("cap-c", 0.7), _c("cap-d", 0.6)]
    before = list(cs)
    normalize_top_k(cs)
    assert cs == before


@pytest.mark.parametrize("k", [1, 2, 3])
def test_single_candidate_is_always_direct_regardless_of_k(k):
    # the K=1 rule is about the CANDIDATE COUNT, not the configured k
    n = normalize_top_k([_c("cap-a", 0.9)], k=k)
    assert n.is_direct and n.candidates == ()
