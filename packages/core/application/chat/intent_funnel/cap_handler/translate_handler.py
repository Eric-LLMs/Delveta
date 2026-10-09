"""TranslateHandler — argument acquisition for ``cap-translate``.

Owns the two ``cap-translate`` schema slots, ``text`` (required) and
``target_language`` (optional). ``cap-translate`` (tool binding ``translate``)
translates a payload INTO a target language; the executor defaults an absent
``target_language`` to **English** (the confirmed contract — the former
always-Chinese behavior is retired, not preserved).

Two different dispositions, per the unified acquisition doctrine:

* ``text`` — the payload — is resolved DET-FIRST (the delimiter ladder below), and
  a payload the delimiters do NOT catch (e.g. a bare ``translate 你好 into
  English`` with no quote/colon) is authorized to the unified local-Qwen extractor
  via ``slot_plan()``. A DET miss returns the EMPTY ``{}`` draft (never ``None``),
  so the model is consulted before the turn can fail; a payload neither the rules
  nor the model resolves honestly lands MISSING for the Binder gate. The payload
  is the real text to translate — NEVER the instruction frame.
* ``target_language`` is a semantic slot (the directive rides on many surface
  forms: "翻译成中文" / "into English" / "用英语怎么表达" / "翻成英文"), so it is
  ALWAYS authorized to the model; the model fills it ONLY when the sentence names
  a target and OMITS it otherwise (never invents one). The deterministic English
  default is the EXECUTOR's own constant, not a model output — so a
  model-unavailable / omitted reply simply leaves the slot absent and the
  executor renders into English.

The formal-500 audit measured the extractor at 17/24 on this capability across
four systematic failure modes (echoing the instruction frame, inventing text when
none was supplied, dropping trailing punctuation, skipping payloads that follow a
lead-in); the DET delimiter parser handles the framed cases and the model covers
the rest.

``text`` delimiter priorities, applied in order:

① A PAIRED QUOTE / CODE BLOCK — the first balanced span among ``"..."``, ``'...'``,
   ``“...”``, ``‘...’``, ``「...」``, ``『...』``, a backtick pair, or a `````...``` ``
   fence. The payload is the INNER text (outer delimiters stripped, surrounding
   whitespace trimmed); everything around it — including any target-language
   directive — is framing.
② A TRANSLATION-CONTEXT COLON — ``：`` / ``:`` is a delimiter ONLY when the whole
   prefix before it is a bare translation lead-in (``翻译：``, ``翻译这句话：``,
   ``translate this sentence:``). A colon anywhere else is NOT a delimiter: there
   is deliberately NO mechanical ``split(":")``, so ``https://`` and
   ``localhost:8080`` are never split.

The payload is copied VERBATIM (only surrounding whitespace trimmed): internal and
trailing punctuation are preserved, a CJK payload is NOT pre-translated, and the
instruction shell is never prepended. The returned draft goes through the SAME
Binder / ActionExecutor / ToolRuntime handoff every other capability uses.
"""
from __future__ import annotations

import re

from .slot_plan import SlotPlan

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
        """Return the DET ``text`` draft when a delimiter yields a payload, else
        the EMPTY ``{}`` draft (never ``None``) so the model can still extract the
        payload. Only a blank query returns ``None`` (no input at all)."""
        message = str(query or "")
        if not message.strip():
            return None
        payload = _paired_payload(message)
        if payload is None:
            payload = _colon_payload(message)
        if payload is None:
            return {}
        return {"text": payload}

    def slot_plan(self, *, query: str, facts, draft: dict) -> SlotPlan:
        """Always authorize ``target_language`` (a semantic slot the model fills
        only when a target is named, omitting otherwise — never invents it).
        Authorize ``text`` ONLY when a delimiter did not resolve the payload; a
        DET-obtained payload is ACQUIRED and never re-asked. The English default
        for an absent ``target_language`` is the executor's, not a model slot."""
        model_slots = ["target_language"]
        if "text" not in (draft or {}):
            model_slots.append("text")
        return SlotPlan(model_slots=tuple(model_slots))
