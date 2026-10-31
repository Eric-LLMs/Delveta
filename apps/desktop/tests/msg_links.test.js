// Inline source links in an answer: rendering, per-source hrefs and click routing.
// Run with:  node --test apps/desktop/tests
const test = require("node:test");
const assert = require("node:assert");

const { resolveChatLink, handleChatLinkClick } = require("../renderer/msg_links.js");
const markdownit = require("../renderer/vendor/markdown-it.min.js");

// The desktop bubble's exact config (app.js renderMarkdown).
const md = markdownit({ html: false, linkify: true, breaks: true });

const WEATHER = "https://www.weather.com.cn/weather/101010100.shtml";
const ACCU = "https://www.accuweather.com/zh/cn/beijing/101924/weather-forecast/101924";
const ANSWER = `数据来自 [中国天气网](${WEATHER})、[AccuWeather](${ACCU}) 等公开气象服务。`;

function visibleText(html) {
  return html.replace(/<[^>]+>/g, "");
}

test("two named sources render as links to their own URLs", () => {
  const html = md.render(ANSWER);
  assert.ok(html.includes(`<a href="${WEATHER}">中国天气网</a>`), html);
  assert.ok(html.includes(`<a href="${ACCU}">AccuWeather</a>`), html);
  // The names are the visible text; the URL lives only in the href.
  const text = visibleText(html);
  assert.ok(text.includes("中国天气网"));
  assert.ok(text.includes("AccuWeather"));
  assert.ok(!text.includes("http"), text);
});

test("each click resolves to that link's own URL", () => {
  const html = md.render(ANSWER);
  const hrefs = [...html.matchAll(/<a href="([^"]+)"/g)].map((m) => m[1]);
  assert.deepEqual(hrefs.map(resolveChatLink), [WEATHER, ACCU]);
});

test("repeated mentions of one source keep their own URLs", () => {
  const html = md.render(
    `[中国天气网](${WEATHER}) 的预报与 [AccuWeather](${ACCU}) 一致；` +
    `[中国天气网](${WEATHER}) 还给了逐小时数据。`);
  const hrefs = [...html.matchAll(/<a href="([^"]+)"/g)].map((m) => m[1]);
  assert.deepEqual(hrefs, [WEATHER, ACCU, WEATHER]);
});

test("a name without a URL stays plain text", () => {
  const html = md.render("数据来自中国天气网等公开气象服务。");
  assert.ok(!html.includes("<a "), html);
  assert.ok(visibleText(html).includes("中国天气网"));
});

test("only http(s) links are routed out; other anchors keep default behaviour", () => {
  assert.equal(resolveChatLink(WEATHER), WEATHER);
  assert.equal(resolveChatLink("  https://example.com/a  "), "https://example.com/a");
  assert.equal(resolveChatLink("javascript:alert(1)"), null);
  assert.equal(resolveChatLink("mailto:a@b.com"), null);
  assert.equal(resolveChatLink("#section"), null);
  assert.equal(resolveChatLink("/drive/file"), null);
  assert.equal(resolveChatLink(""), null);
  assert.equal(resolveChatLink(null), null);
});

// The smallest stand-in for a DOM click: the handler only uses target.closest,
// getAttribute and preventDefault.
function fakeClick(href) {
  const anchor = href === undefined ? null : { getAttribute: () => href };
  const event = {
    target: { closest: (sel) => (sel === "a[href]" ? anchor : null) },
    defaultPrevented: false,
    preventDefault() { this.defaultPrevented = true; },
  };
  return event;
}

test("clicking a source name opens that URL and does not navigate the app", () => {
  const opened = [];
  const event = fakeClick(WEATHER);

  assert.equal(handleChatLinkClick(event, (u) => opened.push(u)), true);
  assert.deepEqual(opened, [WEATHER]);
  assert.equal(event.defaultPrevented, true); // the window stays where it is
});

test("a click that is not an external link is left to the app", () => {
  for (const href of ["javascript:alert(1)", "#top", "/drive/x", "mailto:a@b.com"]) {
    const opened = [];
    const event = fakeClick(href);
    assert.equal(handleChatLinkClick(event, (u) => opened.push(u)), false, href);
    assert.deepEqual(opened, [], href);
    assert.equal(event.defaultPrevented, false, href);
  }
});

test("a click outside any link, or without a link, is not handled", () => {
  const opened = [];
  assert.equal(handleChatLinkClick(fakeClick(undefined), (u) => opened.push(u)), false);
  assert.equal(handleChatLinkClick({ target: {} }, (u) => opened.push(u)), false);
  assert.deepEqual(opened, []);
});

test("a reopened session re-renders the same stored text into the same links", () => {
  // History renders the stored markdown source through the same function, so a link
  // that worked live works after reopening the conversation.
  const live = md.render(ANSWER);
  const reopened = md.render(ANSWER);
  assert.equal(reopened, live);
  assert.deepEqual(
    [...reopened.matchAll(/<a href="([^"]+)"/g)].map((m) => resolveChatLink(m[1])),
    [WEATHER, ACCU]);
});
