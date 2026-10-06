"""TranslateHandler — argument acquisition for ``cap-translate``.

Owns ONLY this capability's single schema slot, ``text``. ``cap-translate`` (tool
binding ``translate``) has NO target-language parameter: the tool always renders
into Chinese, so any "翻译成中文" / "into English" directive in the sentence is
framing, never an argument.

The payload is resolved DETERMINISTICALLY — no model call, no Qwen. The formal-500
audit measured the extractor at 17/24 on this capability across four systematic
failure modes (echoing the instruction frame, inventing text when none was
supplied, dropping trailing punctuation, skipping payloads that follow a lead-in),
all of which a delimiter parser avoids by construction.

Two delimiter priorities, applied in order:

① A PAIRED QUOTE / CODE BLOCK — the first balanced span among ``"..."``, ``'...'``,
   ``“...”``, ``‘...’``, ``「...」``, ``『...』``, a backtick pair, or a ```` ```...``` ````
   fence. The payload is the INNER text (outer delimiters stripped, surrounding
   whitespace trimmed); everything around it — including any target-language
   directive — is framing.
② A TRANSLATION-CONTEXT COLON — ``：`` / ``:`` is a delimiter ONLY when the whole
   prefix before it is a bare translation lead-in (``翻译：``, ``翻译这句话：``,
   ``translate this sentence:``). A colon anywhere else is NOT a delimiter: there
   is deliberately NO mechanical ``split(":")``, so ``https://`` and
   ``localhost:8080`` are never split.

Fail-closed: a bare form, a pure instruction with no payload, or an empty sentence
returns ``None`` — the turn exits to the Agent (``ACQUISITION_MISSING``) rather
than fabricate or echo a ``text``.

The payload is copied VERBATIM (only surrounding whitespace trimmed): internal and
trailing punctuation are preserved, a CJK payload is NOT pre-translated, and the
instruction shell is never prepended. The returned draft goes through the SAME
Binder / ActionExecutor / ToolRuntime handoff every other capability uses.
"""
from __future__ import annotations

import re

# ── priority ①: paired quote / code-block delimiters ─────────────────────────────

# (open, close) delimiter pairs — straight and curly double/single quotes, CJK
# corner brackets, and a single-backtick pair. A triple-backtick fence is matched
# separately (below) because its inner span spans the two single backticks.
_QUOTE_PAIRS = (
    ('"', '"'),
    ("'", "'"),
    ("“", "”"),
    ("‘", "’"),
    ("「", "」"),
    ("『", "』"),
    ("`", "`"),
)

_FENCE = re.compile(r"```(.*?)```", re.DOTALL)


def _paired_payload(text: str) -> str | None:
    """The inner text of the FIRST balanced delimiter span, or ``None``.

    "First" is by the span's opening index, so a quote early in the sentence wins
    over one later. A span whose inner text is blank is not a payload.
    """
    candidates: list[tuple[int, str]] = []

    fence = _FENCE.search(text)
    if fence is not None:
        inner = fence.group(1).strip()
        if inner:
            candidates.append((fence.start(), inner))

    for open_delim, close_delim in _QUOTE_PAIRS:
        start = text.find(open_delim)
        if start == -1:
            continue
        end = text.find(close_delim, start + len(open_delim))
        if end == -1:
            continue
        inner = text[start + len(open_delim):end].strip()
        if inner:
            candidates.append((start, inner))

    if not candidates:
        return None
    candidates.sort(key=lambda candidate: candidate[0])
    return candidates[0][1]


# ── priority ②: a colon that is part of the translation instruction ──────────────

# The tokens a bare translation lead-in may be composed of — nothing else may
# appear before the colon. This is what makes the colon a DELIMITER here while a
# colon inside ``https://`` or ``localhost:8080`` (whose prefix is not such a
# lead-in) never splits.
_LEADIN_TOKENS = (
    # English instruction vocabulary
    "please", "help", "translate", "translation",
    "this", "that", "these", "those", "the",
    "following", "above", "below",
    "sentence", "phrase", "paragraph", "passage", "text", "content",
    "word", "words", "line", "lines",
    "into", "to", "in", "as",
    "chinese", "english", "japanese", "korean", "french", "german",
    "russian", "spanish", "arabic", "portuguese", "italian",
    # Chinese instruction vocabulary
    "请", "帮我", "帮忙", "麻烦", "把", "将",
    "翻译", "翻成", "翻为", "译成", "译为", "译", "转成", "转换",
    "成", "为",
    "这", "那", "这个", "那个", "这句", "那句", "这句话", "那句话",
    "这段", "那段", "这些", "那些", "一下", "一段",
    "句子", "句话", "句", "段", "段落", "文章",
    "文字", "文本", "内容", "短语", "词组", "词", "单词",
    "上面", "以上", "上述", "以下", "下面", "下列",
    "中文", "英文", "英语", "汉语", "日文", "日语", "韩文", "韩语",
    "法语", "法文", "德语", "德文", "俄语", "西班牙语",
)

_TOKEN_ALT = "|".join(
    re.escape(token) for token in sorted(_LEADIN_TOKENS, key=len, reverse=True)
)
_LEADIN_RE = re.compile(
    rf"^(?:{_TOKEN_ALT})(?:[\s,，、]*(?:{_TOKEN_ALT}))*$", re.IGNORECASE
)
_TRANSLATE_VERB = re.compile(r"翻译|翻|译|translat", re.IGNORECASE)

_COLON_CHARS = ("：", ":")


def _is_translation_leadin(prefix: str) -> bool:
    """True when ``prefix`` is composed ONLY of translation-instruction words and
    names a translation verb — i.e. the following colon is a delimiter."""
    prefix = prefix.strip()
    if not prefix or not _TRANSLATE_VERB.search(prefix):
        return False
    return _LEADIN_RE.match(prefix) is not None


def _colon_payload(text: str) -> str | None:
    """The text after the first instruction-embedded colon, or ``None``.

    Scans every colon in order and returns the first whose prefix is a bare
    translation lead-in; non-instruction colons (URLs, ports, prose) are skipped.
    """
    for index, ch in enumerate(text):
        if ch not in _COLON_CHARS:
            continue
        if not _is_translation_leadin(text[:index]):
            continue
        payload = text[index + 1:].strip()
        if payload:
            return payload
    return None


class TranslateHandler:
    """``cap-translate`` parameter handler (stateless).

    ``facts`` is accepted for the common handler contract but unused: the payload
    lives ONLY in the sentence. (A viewer selection is not reachable here — no
    :class:`~..contract.TurnFacts` field carries its text.)
    """

    capability_id = "cap-translate"

    async def acquire(self, *, query: str, facts) -> dict[str, object] | None:
        """Return the ``translate`` argument draft, or ``None`` when the sentence
        carries no extractable payload (fail-closed — the Agent owns the turn)."""
        message = str(query or "")
        payload = _paired_payload(message)
        if payload is None:
            payload = _colon_payload(message)
        if payload is None:
            return None
        return {"text": payload}
