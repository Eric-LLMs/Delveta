// Reload budget for research gate notes.
//
// A pending override's deterministic ``system`` note is committed to the session, but a
// chat restored on its newest page may not include it (long runs push it past the page
// window). Reloading cannot make an older row appear, so an unbounded "note missing ⇒
// reload" rule would reload the whole session on every status refresh. Each approval id
// gets a small, fixed number of attempts instead: enough to catch a commit that was still
// in flight when the status was read, bounded so polling can never loop.
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.ResearchGateNotes = factory();
})(typeof self !== "undefined" ? self : this, function () {
  const MAX_ATTEMPTS = 2;

  // ``visible`` is how many gate notes are on screen; ``pendingIds`` comes from the status
  // payload; ``attempts`` is a Map of approval id → reloads already charged.
  // Returns ``{ ids }`` — the approvals to charge for this reload — or null when no reload
  // should happen.
  function reloadAttempt(visible, pendingIds, attempts, maxAttempts = MAX_ATTEMPTS) {
    const ids = [];
    const seen = new Set();
    for (const raw of pendingIds || []) {
      if (!raw) continue;
      const id = String(raw);
      if (seen.has(id)) continue;
      seen.add(id);
      ids.push(id);
    }
    if (!ids.length || visible >= ids.length) return null;
    const charged = ids.filter((id) => (attempts.get(id) || 0) < maxAttempts);
    if (!charged.length) return null;
    return { ids: charged };
  }

  return { MAX_ATTEMPTS, reloadAttempt };
});
