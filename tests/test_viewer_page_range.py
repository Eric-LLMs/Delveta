"""Viewer page RANGE — UI/ViewerPayload -> TurnFacts context plumbing (Phase 2-A).

The only legal source of a page/range for a file-content capability is the UI:
``ViewerPayload.page`` (a lone current page) or the explicit ``page_from`` /
``page_to`` bounds. The turn-facts assembly settles those into
``TurnFacts.viewer_page_from`` / ``viewer_page_to``; nothing here parses a page
number out of the user's sentence. Three states are pinned:

* single page  -> a degenerate range (from == to == page);
* explicit range -> mapped faithfully, and it WINS over ``page``;
* no page at all -> None/None (the whole document).
"""
from __future__ import annotations

import types

import pytest
from api.schemas import ViewerPayload
from core.application.chat.intent_funnel.contract import TurnFacts


def _ctx(viewer):
    return types.SimpleNamespace(
        body=types.SimpleNamespace(message="x", attach=None, viewer=viewer),
        owned_asset_id=None, path_asset_id="", session_id=None,
    )


# ── ViewerPayload: the transport bounds ───────────────────────────────────────

def test_payload_accepts_an_explicit_range():
    v = ViewerPayload(name="p.pdf", kind="pdf", page_from=3, page_to=5)
    assert v.page_from == 3 and v.page_to == 5
    assert v.page is None  # a range alone does not invent a current page


def test_payload_single_page_leaves_range_unset():
    # A lone page stays a lone page on the wire; the degenerate-range
    # normalization happens at the turn-facts assembly, not here.
    v = ViewerPayload(name="p.pdf", kind="pdf", page=7)
    assert v.page == 7 and v.page_from is None and v.page_to is None


@pytest.mark.parametrize("field", ["page_from", "page_to"])
@pytest.mark.parametrize("bad", [0, 10001])
def test_payload_rejects_out_of_bounds_bounds(field, bad):
    with pytest.raises(ValueError, match=field):
        ViewerPayload(name="p.pdf", kind="pdf", **{field: bad})


def test_payload_rejects_a_reversed_range():
    with pytest.raises(ValueError, match="page_from"):
        ViewerPayload(name="p.pdf", kind="pdf", page_from=5, page_to=3)


# ── TurnFacts.of: the settled facts ───────────────────────────────────────────

def test_turn_facts_single_page_is_a_degenerate_range():
    f = TurnFacts.of(_ctx(ViewerPayload(name="p.pdf", kind="pdf", page=12)))
    assert f.viewer_current_page == 12
    assert f.viewer_page_from == 12 and f.viewer_page_to == 12


def test_turn_facts_explicit_range_maps_faithfully():
    f = TurnFacts.of(_ctx(ViewerPayload(name="p.pdf", kind="pdf", page_from=2, page_to=5)))
    assert f.viewer_page_from == 2 and f.viewer_page_to == 5
    assert f.viewer_current_page is None


def test_turn_facts_explicit_range_wins_over_current_page():
    f = TurnFacts.of(_ctx(
        ViewerPayload(name="p.pdf", kind="pdf", page=12, page_from=3, page_to=5)
    ))
    assert f.viewer_current_page == 12
    assert f.viewer_page_from == 3 and f.viewer_page_to == 5


def test_turn_facts_partial_bound_is_not_fabricated():
    # Only the low bound is known: the high bound stays None (never guessed from `page`).
    f = TurnFacts.of(_ctx(ViewerPayload(name="p.pdf", kind="pdf", page=12, page_from=3)))
    assert f.viewer_page_from == 3 and f.viewer_page_to is None


def test_turn_facts_no_page_and_no_viewer_are_both_none():
    no_page = TurnFacts.of(_ctx(ViewerPayload(name="p.pdf", kind="pdf")))
    assert no_page.viewer_page_from is None and no_page.viewer_page_to is None
    assert no_page.viewer_current_page is None

    no_viewer = TurnFacts.of(_ctx(None))
    assert no_viewer.viewer_page_from is None and no_viewer.viewer_page_to is None
