// Paging rules for the conversation transcript: single-flight claims, auto-fill, cursor
// advancement. Run with:  node --test apps/desktop
const test = require("node:test");
const assert = require("node:assert");

const {
  claim,
  release,
  pageAfter,
  needsFill,
} = require("../renderer/history_paging_model.js");

const state = (over = {}) => ({
  cursor: "c1",
  hasMore: true,
  loading: false,
  ...over,
});

const metrics = (scrollHeight, clientHeight) => ({ scrollHeight, clientHeight });

test("claim refuses when there is nothing older to load", () => {
  assert.equal(claim(state({ hasMore: false })), false);
  assert.equal(claim(state({ cursor: null })), false);
  assert.equal(claim(null), false);
});

test("claim refuses while a page is already in flight", () => {
  const st = state();
  assert.equal(claim(st), true);
  assert.equal(st.loading, true);
  // The second caller (scroll burst / auto-fill in the same tick) must not start a request.
  assert.equal(claim(st), false);
  release(st);
  assert.equal(claim(st), true);
});

test("a page advances only when the server reports more and the cursor moved", () => {
  assert.deepEqual(pageAfter(null, { has_more: true, next_before: "c2" }), {
    cursor: "c2",
    hasMore: true,
  });
  assert.deepEqual(pageAfter("c2", { has_more: false, next_before: null }), {
    cursor: null,
    hasMore: false,
  });
  assert.deepEqual(pageAfter("c2", { has_more: true, next_before: "c2" }), {
    cursor: null,
    hasMore: false,
  });
  assert.deepEqual(pageAfter("c2", null), { cursor: null, hasMore: false });
});

test("a log without a scrollbar needs another page", () => {
  assert.equal(needsFill(state(), metrics(400, 600)), true);
  // Exactly at the tolerance is still "not scrollable".
  assert.equal(needsFill(state(), metrics(596, 600)), true);
});

test("a scrollable log does not auto-fill", () => {
  assert.equal(needsFill(state(), metrics(1200, 600)), false);
});

test("auto-fill stops when a request is running or history is exhausted", () => {
  assert.equal(needsFill(state({ loading: true }), metrics(400, 600)), false);
  assert.equal(needsFill(state({ hasMore: false }), metrics(400, 600)), false);
  assert.equal(needsFill(state({ cursor: null }), metrics(400, 600)), false);
});

test("a hidden pane (zero height) never issues a page request", () => {
  assert.equal(needsFill(state(), metrics(400, 0)), false);
});

test("fill loop issues one request per page and stops when the log scrolls", async () => {
  // A stand-in for the DOM: each page adds 300px of content.
  let scrollHeight = 100;
  const clientHeight = 600;
  const st = state();
  const requests = [];

  async function loadPage() {
    if (!claim(st)) return;
    try {
      requests.push(st.cursor);
      // A second trigger while this request is in flight must be a no-op.
      assert.equal(claim(st), false);
      scrollHeight += 300;
      const page = { has_more: true, next_before: `c${requests.length + 1}` };
      const next = pageAfter(st.cursor, page);
      st.cursor = next.cursor;
      st.hasMore = next.hasMore;
    } finally {
      release(st);
    }
  }

  while (needsFill(st, metrics(scrollHeight, clientHeight))) await loadPage();

  assert.deepEqual(requests, ["c1", "c2"]); // 400 → 700 → scrollable; no extra request
  assert.equal(st.hasMore, true);
  assert.equal(st.loading, false);
});

test("auto-fill never loops when a page reports no cursor progress", async () => {
  let scrollHeight = 400;
  const st = state();
  let requests = 0;

  async function loadPage() {
    if (!claim(st)) return;
    try {
      requests += 1;
      const next = pageAfter(st.cursor, { has_more: true, next_before: st.cursor });
      st.cursor = next.cursor;
      st.hasMore = next.hasMore;
      scrollHeight += 10; // still not scrollable — but paging is exhausted
    } finally {
      release(st);
    }
  }

  while (needsFill(st, metrics(scrollHeight, 600))) await loadPage();

  assert.equal(requests, 1);
  assert.equal(st.hasMore, false);
});

test("the browser global app.js reads is the one this module exports", () => {
  const fs = require("node:fs");
  const path = require("node:path");
  const vm = require("node:vm");
  const code = fs.readFileSync(
    path.join(__dirname, "..", "renderer", "history_paging_model.js"),
    "utf8",
  );
  const sandbox = { self: {} };
  vm.runInNewContext(code, sandbox);
  assert.equal(typeof sandbox.self.HistoryPagingModel.claim, "function");
  assert.equal(typeof sandbox.self.HistoryPagingModel.needsFill, "function");
});
