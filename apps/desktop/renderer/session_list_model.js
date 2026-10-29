// Paging rules for the session list: how a fetched page merges with what is already on
// screen, and whether a server response actually advances the cursor.
//
// Pure and dependency-free so it runs unchanged in the renderer and under `node --test`.
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.SessionListModel = factory();
})(typeof self !== "undefined" ? self : this, function () {
  // Merge rows without duplicates: the existing order is kept (callers pass pinned rows
  // first), and only ids not already present are appended.
  function mergeRows(current, added) {
    const out = [];
    const seen = new Set();
    for (const row of (current || []).concat(added || [])) {
      if (!row || row.id == null) continue;
      const key = String(row.id);
      if (seen.has(key)) continue;
      seen.add(key);
      out.push(row);
    }
    return out;
  }

  // A page advances paging only when the server reports more AND the cursor moved. A
  // cursor that would repeat itself ends the list instead of re-querying the same page.
  function cursorAfter(cursor, page) {
    const next = page && page.next_before ? String(page.next_before) : null;
    const hasMore = Boolean(page && page.has_more && next && next !== cursor);
    return { cursor: hasMore ? next : null, hasMore };
  }

  return { mergeRows, cursorAfter };
});
