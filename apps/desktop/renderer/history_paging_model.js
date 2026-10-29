// Paging rules for one conversation's transcript: whether a page may be requested, how
// the cursor advances, and whether the log still needs another page to become scrollable.
//
// Pure and dependency-free so it runs unchanged in the renderer and under `node --test`.
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.HistoryPagingModel = factory();
})(typeof self !== "undefined" ? self : this, function () {
  // A log this close to its own height cannot scroll yet.
  const FILL_TOLERANCE_PX = 4;

  // Claim the single in-flight slot before starting a request. Synchronous by design: two
  // callers in the same tick (a scroll burst and the post-restore fill) can never both win.
  function claim(state) {
    if (!state || state.loading || !state.hasMore || !state.cursor) return false;
    state.loading = true;
    return true;
  }

  function release(state) {
    if (state) state.loading = false;
  }

  // Older pages remain only while the server says so AND the cursor moved. A cursor that
  // would repeat itself ends paging instead of re-querying the same page.
  function pageAfter(previousCursor, page) {
    const next = page && page.next_before ? String(page.next_before) : null;
    const hasMore = Boolean(page && page.has_more && next && next !== previousCursor);
    return { cursor: hasMore ? next : null, hasMore };
  }

  // A page that fits without a scrollbar leaves the reader with no way to ask for the next
  // one, so the pane keeps filling until the log can scroll. Hidden panes report a zero
  // height and are skipped — they re-arm when restored.
  function needsFill(state, metrics) {
    if (!state || !state.hasMore || state.loading || !state.cursor) return false;
    if (!metrics || !(metrics.clientHeight > 0)) return false;
    return metrics.scrollHeight <= metrics.clientHeight + FILL_TOLERANCE_PX;
  }

  return { FILL_TOLERANCE_PX, claim, release, pageAfter, needsFill };
});
