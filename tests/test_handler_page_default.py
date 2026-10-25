"""The handler page default — the viewer's current page as a ``pages`` value.

Pins the deterministic half of the approved acquisition rule:

* only the CURRENT page is used; the visible-range fields are not a page request;
* the viewer must be showing the SAME document as the target ``asset_id``;
* a missing, zero, negative or non-numeric page yields no value — never a
  fabricated one, never ``"0"``, ``"无"`` or ``"null"``;
* a selection does not change the page value.
"""
from __future__ import annotations

import types

import pytest

from core.application.chat.intent_funnel.argument_acquisition.handler_page_default import (
    handler_page_default,
)


def _facts(**over):
    base = {"viewer_asset_id": "asset-1", "viewer_current_page": 4,
            "viewer_page_from": 4, "viewer_page_to": 6,
            "has_viewer_selection": False}
    base.update(over)
    return types.SimpleNamespace(**base)


def test_the_current_page_of_the_same_document_is_the_value():
    assert handler_page_default(_facts(), asset_id="asset-1") == "4"


def test_a_missing_page_yields_no_value():
    assert handler_page_default(
        _facts(viewer_current_page=None), asset_id="asset-1") is None


@pytest.mark.parametrize("page", [0, -1, "not-a-page", "", True, False])
def test_a_non_positive_or_non_numeric_page_yields_no_value(page):
    assert handler_page_default(
        _facts(viewer_current_page=page), asset_id="asset-1") is None


def test_a_viewer_page_of_another_document_is_never_reused():
    assert handler_page_default(
        _facts(viewer_asset_id="asset-other"), asset_id="asset-1") is None


def test_an_unknown_viewer_asset_yields_no_value():
    assert handler_page_default(
        _facts(viewer_asset_id=""), asset_id="asset-1") is None


def test_a_missing_target_asset_yields_no_value():
    assert handler_page_default(_facts(), asset_id="") is None


def test_the_visible_range_is_not_used_as_the_page_value():
    facts = _facts(viewer_current_page=None, viewer_page_from=2,
                   viewer_page_to=5)
    assert handler_page_default(facts, asset_id="asset-1") is None


def test_a_selection_does_not_change_the_page_value():
    facts = _facts(has_viewer_selection=True)
    assert handler_page_default(facts, asset_id="asset-1") == "4"
