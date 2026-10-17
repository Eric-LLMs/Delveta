// Regression tests for the Viewer → Message content-type decision (design rule §1.3).
// Run with:  node --test apps/desktop/tests
const test = require("node:test");
const assert = require("node:assert");

const { classifyPageContent, MIN_TEXT_CHARS } = require("../renderer/page_content.js");

const NO_SIGNALS = { hasImages: false, hasDrawings: false };

test("plain extractable text → text only, no screenshot", () => {
  const c = classifyPageContent("This page has a paragraph of ordinary prose.", NO_SIGNALS);
  assert.equal(c.hasText, true);
  assert.equal(c.needsImage, false);
  assert.equal(c.reason, "text");
});

test("scanned page (no extractable text) → screenshot", () => {
  const c = classifyPageContent("", NO_SIGNALS);
  assert.equal(c.hasText, false);
  assert.equal(c.needsImage, true);
  assert.equal(c.reason, "scanned");
});

test("near-empty page (only a page number / running head) counts as scanned", () => {
  const c = classifyPageContent("   12  ", NO_SIGNALS);
  assert.equal(c.hasText, false);
  assert.equal(c.needsImage, true);
  assert.equal(c.reason, "scanned");
});

test("char count exactly at the threshold is content", () => {
  const text = "x".repeat(MIN_TEXT_CHARS);
  const c = classifyPageContent(text, NO_SIGNALS);
  assert.equal(c.hasText, true);
  assert.equal(c.needsImage, false);
});

test("table/chart page (vector drawings) → text AND screenshot", () => {
  const c = classifyPageContent("Region | Revenue | Growth", { hasImages: false, hasDrawings: true });
  assert.equal(c.hasText, true);
  assert.equal(c.needsImage, true);
  assert.equal(c.reason, "table_or_chart");
});

test("image page (embedded picture) → text AND screenshot", () => {
  const c = classifyPageContent("Figure 3: system architecture overview.", { hasImages: true, hasDrawings: false });
  assert.equal(c.hasText, true);
  assert.equal(c.needsImage, true);
  assert.equal(c.reason, "image");
});

test("drawings take precedence over images in the reported reason", () => {
  const c = classifyPageContent("some page text here", { hasImages: true, hasDrawings: true });
  assert.equal(c.reason, "table_or_chart");
});

test("missing signals default to text-only (never blocks as scanned)", () => {
  const c = classifyPageContent("A normal paragraph of text.", undefined);
  assert.equal(c.needsImage, false);
  assert.equal(c.reason, "text");
});

test("text is trimmed in the returned payload", () => {
  const c = classifyPageContent("   padded text value here   ", NO_SIGNALS);
  assert.equal(c.text, "padded text value here");
});
