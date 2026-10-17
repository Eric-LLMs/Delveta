// Viewer → Message content-type decision (design rule §1.3).
//
// Given a page's extracted text and its rendering signals (does it draw pictures / vector
// line-work?), decide what the composer must stage for the user to confirm:
//
//   text only  → attach the extracted text
//   anything else the text can't carry → ALSO attach a page screenshot
//     * "scanned"        no extractable text at all → screenshot is the only content
//     * "table_or_chart" grid / plot drawn as vector ops → text loses the layout
//     * "image"          embedded picture → text loses it entirely
//
// Pure and dependency-free so it runs unchanged in the renderer and under `node --test`.
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.PageContent = factory();
})(typeof self !== "undefined" ? self : this, function () {
  // A handful of characters is a page number / running head, not content — treat it as
  // "no text" so a nearly-empty page still gets a screenshot.
  const MIN_TEXT_CHARS = 8;

  function classifyPageContent(text, signals) {
    const clean = (text || "").trim();
    const hasText = clean.length >= MIN_TEXT_CHARS;
    const hasDrawings = !!(signals && signals.hasDrawings);
    const hasImages = !!(signals && signals.hasImages);
    const needsImage = !hasText || hasDrawings || hasImages;
    const reason = !hasText
      ? "scanned"
      : hasDrawings
        ? "table_or_chart"
        : hasImages
          ? "image"
          : "text";
    return { text: clean, hasText, needsImage, reason };
  }

  return { classifyPageContent, MIN_TEXT_CHARS };
});
