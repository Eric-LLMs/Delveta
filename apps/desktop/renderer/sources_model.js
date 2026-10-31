// Web-search source footer: which sources an answer may cite and which label to show.
//
// The backend hands the turn's successful web_search results as ``web_sources``
// ({title, url}); this module decides what the footer may render. Pure and
// dependency-free so it runs unchanged in the renderer and under `node --test`.
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.SourcesModel = factory();
})(typeof self !== "undefined" ? self : this, function () {
  const MAX_SOURCES = 10;
  // CJK ideographs: an answer in Chinese gets 来源, everything else Sources.
  const CJK = /[㐀-䶿一-鿿豈-﫿]/;

  function isHttpUrl(url) {
    const u = String(url || "").trim().toLowerCase();
    return u.startsWith("http://") || u.startsWith("https://");
  }

  // Valid, de-duplicated http(s) sources in first-seen order. A repeated URL (the
  // crawled page entry repeats its result) keeps the first title; the cap bounds the
  // popover without dropping the top of the ranking.
  function normalizeSources(sources) {
    const out = [];
    const seen = new Set();
    for (const s of sources || []) {
      if (!s || !isHttpUrl(s.url)) continue;
      const url = String(s.url).trim();
      if (seen.has(url)) continue;
      seen.add(url);
      out.push({ title: String(s.title || "").trim(), url });
      if (out.length >= MAX_SOURCES) break;
    }
    return out;
  }

  function sourcesLabel(answerText) {
    return CJK.test(String(answerText || "")) ? "来源" : "Sources";
  }

  return { normalizeSources, sourcesLabel, isHttpUrl, MAX_SOURCES };
});
