// Paging rules for the Chat session list: page merge / dedupe and cursor advancement.
// Run with:  node --test apps/desktop
const test = require("node:test");
const assert = require("node:assert");

const { mergeRows, cursorAfter } = require("../renderer/session_list_model.js");

const row = (id) => ({ id, title: id });

test("pinned rows come first and are not duplicated by the page", () => {
  const merged = mergeRows([row("p1"), row("p2")], [row("p2"), row("n1"), row("n2")]);
  assert.deepEqual(merged.map((r) => r.id), ["p1", "p2", "n1", "n2"]);
});

test("appending a page keeps the existing order and drops repeats", () => {
  const first = [row("a"), row("b")];
  const merged = mergeRows(first, [row("b"), row("c"), row("d")]);
  assert.deepEqual(merged.map((r) => r.id), ["a", "b", "c", "d"]);
});

test("ids are compared as strings, so one row can never appear twice", () => {
  const merged = mergeRows([{ id: 7 }], [{ id: "7" }]);
  assert.equal(merged.length, 1);
});

test("rows without an id are skipped instead of corrupting the list", () => {
  const merged = mergeRows([row("a")], [null, {}, row("b")]);
  assert.deepEqual(merged.map((r) => r.id), ["a", "b"]);
});

test("a page with more rows and a new cursor advances paging", () => {
  assert.deepEqual(cursorAfter(null, { has_more: true, next_before: "c1" }), {
    cursor: "c1",
    hasMore: true,
  });
  assert.deepEqual(cursorAfter("c1", { has_more: true, next_before: "c2" }), {
    cursor: "c2",
    hasMore: true,
  });
});

test("the last page stops paging", () => {
  assert.deepEqual(cursorAfter("c2", { has_more: false, next_before: null }), {
    cursor: null,
    hasMore: false,
  });
});

test("a cursor that does not move ends the list instead of looping", () => {
  assert.deepEqual(cursorAfter("c1", { has_more: true, next_before: "c1" }), {
    cursor: null,
    hasMore: false,
  });
  assert.deepEqual(cursorAfter("c1", { has_more: true }), { cursor: null, hasMore: false });
});

test("a missing page (failed fetch) never claims there is more", () => {
  assert.deepEqual(cursorAfter("c1", null), { cursor: null, hasMore: false });
});
