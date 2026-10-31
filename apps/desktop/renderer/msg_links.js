// Inline source links inside a chat answer: which href a click may open externally.
//
// Answers cite sources as ordinary Markdown links — `[中国天气网](https://…)` — which
// markdown-it already renders; the one gap is the click, because an unhandled anchor
// navigates the whole Electron window away from the app. Pure and dependency-free so it
// runs unchanged in the renderer and under `node --test`.
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.MsgLinks = factory();
})(typeof self !== "undefined" ? self : this, function () {
  // The URL a click on a chat link should open in the system browser, or null when the
  // anchor is not an external page (relative / anchor / mailto / javascript:) and must
  // keep its default behaviour. Never returns a value the answer could have supplied
  // for an unsafe scheme.
  function resolveChatLink(href) {
    const u = String(href || "").trim();
    return /^https?:\/\//i.test(u) ? u : null;
  }

  // The chat transcript's click rule: an inline source link opens in the system
  // browser instead of navigating the Electron window. Returns true when it handled
  // the click (the caller then stops), false for every other click.
  function handleChatLinkClick(event, open) {
    const anchor = event.target && event.target.closest && event.target.closest("a[href]");
    if (!anchor) return false;
    const url = resolveChatLink(anchor.getAttribute("href"));
    if (!url) return false;
    event.preventDefault();
    open(url);
    return true;
  }

  return { resolveChatLink, handleChatLinkClick };
});
