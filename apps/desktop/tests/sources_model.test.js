// Web-search source footer: label language, URL validity, de-duplication and the cap.
// Run with:  node --test apps/desktop
const test = require("node:test");
const assert = require("node:assert");

const { normalizeSources, sourcesLabel, MAX_SOURCES } = require("../renderer/sources_model.js");

test("a Chinese answer gets the Chinese label", () => {
  assert.equal(sourcesLabel("今天北京晴，最高 24 度。"), "来源");
});

test("an English answer gets the English label", () => {
  assert.equal(sourcesLabel("It is sunny in Beijing today."), "Sources");
});

test("an empty answer falls back to the English label", () => {
  assert.equal(sourcesLabel(""), "Sources");
  assert.equal(sourcesLabel(null), "Sources");
});

test("only http(s) sources survive", () => {
  const out = normalizeSources([
    { title: "ok", url: "https://example.com/a" },
    { title: "secure", url: "http://example.com/b" },
    { title: "js", url: "javascript:alert(1)" },
    { title: "file", url: "file:///etc/passwd" },
    { title: "empty", url: "" },
  ]);
  assert.deepEqual(out.map((s) => s.url), ["https://example.com/a", "http://example.com/b"]);
});

test("the same URL appears once and keeps its first title", () => {
  const out = normalizeSources([
    { title: "result", url: "https://example.com/a" },
    { title: "crawled page", url: "https://example.com/a" },
  ]);
  assert.equal(out.length, 1);
  assert.equal(out[0].title, "result");
});

test("the source cap bounds the popover", () => {
  const many = Array.from({ length: MAX_SOURCES + 4 }, (_, i) => ({
    title: `s${i}`,
    url: `https://example.com/${i}`,
  }));
  assert.equal(normalizeSources(many).length, MAX_SOURCES);
});

test("missing or malformed lists render nothing", () => {
  assert.deepEqual(normalizeSources(null), []);
  assert.deepEqual(normalizeSources([]), []);
  assert.deepEqual(normalizeSources([null, {}, { url: "https://example.com" }]).length, 1);
});
