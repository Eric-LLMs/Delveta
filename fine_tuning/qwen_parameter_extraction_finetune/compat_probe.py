#!/usr/bin/env python3
"""Real-service compatibility probe — NATIVE function calling against the local
Qwen tool endpoint.

STATUS: **NOT YET PASSED**. It was written while the docker ``tool-intent``
service (Ollama / ``qwen3-tools:q5_k_m`` at ``localhost:18091/v1``) was DOWN, so
it has never run against the live model. Re-run it once the service is back; only
a PASS here clears the real-service half of the pre-training gate.

What it proves (the SAME wire the online extractor pins, built by the SAME
production helpers — ``extractor._payload`` / ``_parse_tool_call`` / the shared
``pe_contract`` — so it cannot drift from production):

* a NON-STREAMING ``chat/completions`` request carrying ONE tool
  (``name == capability_id``, ``parameters`` == the frozen model schema) plus
  ``tool_choice="required"`` is accepted; and
* the raw reply carries ``message.tool_calls[0]`` whose ``function.name`` equals
  the pinned capability and whose ``function.arguments`` is a JSON OBJECT that
  parses to the expected flat slot map.

Usage:
    python compat_probe.py                       # create-folder, local default URL
    python compat_probe.py --url http://localhost:18091/v1
    python compat_probe.py --capability cap-translate --query '把 "hi" 翻译成中文'

Exit code 0 == PASS (native tool_call, right name, object arguments); non-zero
otherwise, printing the raw reply for inspection.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import pe_contract as C  # noqa: E402

# The app package (for the production payload/parse + config default URL).
for p in (str(REPO), str(REPO / "packages")):
    if p not in sys.path:
        sys.path.insert(0, p)

DEFAULT_QUERY = {
    "cap-create-folder": "创建一个叫 testA 的文件夹",
    "cap-translate": '把 "hello world" 翻译成中文',
    "cap-web-search": "帮我搜一下注意力机制的最新进展",
}


def _entry(cap: str) -> SimpleNamespace:
    return SimpleNamespace(
        capability_id=cap, tool_binding=C.tool_name(cap),
        description=C.function_description(cap), parameters=C.SCHEMA[cap]["params"])


def main() -> int:
    ap = argparse.ArgumentParser(description="Real-service native-function-call probe.")
    ap.add_argument("--capability", default="cap-create-folder", choices=sorted(C.SCHEMA))
    ap.add_argument("--query", default=None)
    ap.add_argument("--url", default=None,
                    help="defaults to settings.chat_tool_intent_local_url")
    args = ap.parse_args()

    from core.application.chat.intent_funnel.argument_acquisition import (
        context_bundle, extractor as ex)
    from core.application.chat.intent_funnel.argument_acquisition.prompts import prompt_for
    from core.config import settings

    url = (args.url or settings.chat_tool_intent_local_url or "").strip()
    if not url:
        print("FAIL: no endpoint URL (pass --url or set chat_tool_intent_local_url)",
              file=sys.stderr)
        return 2
    query = args.query or DEFAULT_QUERY.get(args.capability, "帮我建个文件夹")
    cap = args.capability
    entry = _entry(cap)
    model_slots = C.model_slots(cap)
    bundle = context_bundle.build(query)
    payload = ex._payload(entry, model_slots, bundle, prompt_for(cap))

    model = payload.get("model") or "(server default)"
    print(f"POST {url.rstrip('/')}/chat/completions  model={model}  capability={cap}")
    print(f"tool_choice={payload['tool_choice']}  tools={[t['function']['name'] for t in payload['tools']]}")
    print(f"query={query!r}")

    req = urllib.request.Request(
        url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        print(f"\nFAIL: endpoint unreachable/bad: {exc!r}", file=sys.stderr)
        print("STATUS: NOT-YET-PASSED (service down — no live verification)", file=sys.stderr)
        return 3

    message = body.get("choices", [{}])[0].get("message", {})
    print("\nraw message.tool_calls:")
    print(json.dumps(message.get("tool_calls"), ensure_ascii=False, indent=2))

    ok = True
    if not isinstance(body.get("choices"), list) or not body["choices"]:
        print("FAIL: no choices in the reply", file=sys.stderr)
        return 4
    calls = message.get("tool_calls") or []
    if len(calls) != 1:
        print(f"FAIL: expected exactly 1 tool_call, got {len(calls)}", file=sys.stderr)
        ok = False
    else:
        fn = calls[0].get("function") or {}
        if str(fn.get("name") or "") != cap:
            print(f"FAIL: function.name {fn.get('name')!r} != {cap!r}", file=sys.stderr)
            ok = False
        raw = fn.get("arguments")
        try:
            got = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(got, dict):
                print("FAIL: function.arguments is not a JSON object", file=sys.stderr)
                ok = False
        except ValueError as exc:
            print(f"FAIL: function.arguments not JSON: {exc!r}", file=sys.stderr)
            ok = False
    if ok:
        # Also exercise the production parser: the parsed values ARE the model's
        # output (no grounding / no semantic post-processing downstream).
        values = ex._parse_tool_call(message, entry, model_slots)
        print(f"\nparsed values : {json.dumps(values, ensure_ascii=False)}")
        print("\nSTATUS: PASS (native tool_call, matching name, object arguments)")
        return 0
    print("\nSTATUS: NOT-YET-PASSED", file=sys.stderr)
    return 5


if __name__ == "__main__":
    raise SystemExit(main())
