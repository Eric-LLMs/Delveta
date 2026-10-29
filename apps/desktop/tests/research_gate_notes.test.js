// Reload budget for research gate notes. Run with:  node --test apps/desktop
const test = require("node:test");
const assert = require("node:assert");

const { MAX_ATTEMPTS, reloadAttempt } = require("../renderer/research_gate_notes.js");

test("no pending override means no reload", () => {
  assert.equal(reloadAttempt(0, [], new Map()), null);
  assert.equal(reloadAttempt(3, [null, ""], new Map()), null);
});

test("every pending note already on screen means no reload", () => {
  assert.equal(reloadAttempt(1, ["a"], new Map()), null);
  assert.equal(reloadAttempt(2, ["a", "a"], new Map()), null); // duplicate ids count once
});

test("a missing note reloads once per refresh until the budget is spent", () => {
  const attempts = new Map();
  const first = reloadAttempt(0, ["a"], attempts);
  assert.deepEqual(first, { ids: ["a"] });
  attempts.set("a", (attempts.get("a") || 0) + 1);

  const second = reloadAttempt(0, ["a"], attempts);
  assert.deepEqual(second, { ids: ["a"] });
  attempts.set("a", (attempts.get("a") || 0) + 1);

  // A note older than the newest page can never appear by reloading; polling must stop.
  assert.equal(attempts.get("a"), MAX_ATTEMPTS);
  assert.equal(reloadAttempt(0, ["a"], attempts), null);
});

test("a new approval gets its own budget", () => {
  const attempts = new Map([["a", MAX_ATTEMPTS]]);
  assert.deepEqual(reloadAttempt(0, ["a", "b"], attempts), { ids: ["b"] });
});

test("a partially covered set charges only the approvals still under budget", () => {
  const attempts = new Map([["a", MAX_ATTEMPTS]]);
  assert.deepEqual(reloadAttempt(1, ["a", "b"], attempts), { ids: ["b"] });
  // Both ids are seen, but only b may be charged; the caller re-checks after reloading.
  assert.deepEqual(reloadAttempt(0, ["a"], attempts), null);
});

test("the browser global app.js reads is the one this module exports", () => {
  const fs = require("node:fs");
  const path = require("node:path");
  const vm = require("node:vm");
  const code = fs.readFileSync(
    path.join(__dirname, "..", "renderer", "research_gate_notes.js"),
    "utf8",
  );
  const sandbox = { self: {} };
  vm.runInNewContext(code, sandbox);
  assert.equal(typeof sandbox.self.ResearchGateNotes.reloadAttempt, "function");
});
