// Icon helpers for renderer modules that build DOM imperatively.
//
// The icon set itself is the inline <symbol> sprite declared at the top of index.html —
// this file only turns a symbol name into markup/DOM. Keeping the sprite in the HTML means
// <use href="#i-…"> resolves during the first paint (no flash of missing icons), and keeping
// the helper tiny avoids a runtime registry that every module has to share.
(function () {
  "use strict";

  // Names are author-controlled, but validate anyway so a stray value can never inject
  // markup into the document.
  const NAME = /^[a-z0-9-]+$/;

  function dvIconHTML(name, className) {
    if (!NAME.test(String(name))) return "";
    const cls = className ? `icon ${className}` : "icon";
    return `<svg class="${cls}" aria-hidden="true" focusable="false"><use href="#i-${name}"></use></svg>`;
  }

  function dvIcon(name, className) {
    const tpl = document.createElement("template");
    tpl.innerHTML = dvIconHTML(name, className).trim();
    return tpl.content.firstElementChild;
  }

  window.dvIconHTML = dvIconHTML;
  window.dvIcon = dvIcon;
})();
