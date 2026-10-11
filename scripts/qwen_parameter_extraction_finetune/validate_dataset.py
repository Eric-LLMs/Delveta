#!/usr/bin/env python3
"""Verify the frozen parameter-extraction SFT bundle; fail loudly on any drift.

Stdlib only: reads config.yaml, the manifest (+ SHA256SUMS) and the three
``data/*.jsonl`` splits, and re-checks every invariant the builder guarantees —
file hashes, counts, the tool schema, the assistant ``tool_calls`` target, id and
query uniqueness, cross-split and Formal-500 leakage. Exits non-zero on the first
failing category (the training entry must refuse a bundle the verifier rejects).

When the app package is importable it also re-renders each system + user message
from the live prompts and requires an EXACT byte match, so a silent renderer drift
is caught here rather than at training time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

import pe_contract as C  # noqa: E402

SPLITS = ("train", "validation", "test")
PAYLOAD_SLOT = {
    "cap-web-search": "query", "cap-rag-search": "query", "cap-social-search": "query",
    "cap-translate": "text", "cap-add-term": "term", "cap-create-folder": "name",
}
FAILS: list[str] = []


def _payload(target: dict, cap: str) -> str:
    v = target.get(PAYLOAD_SLOT[cap])
    return v if isinstance(v, str) else json.dumps(target, sort_keys=True, ensure_ascii=False)


def fail(msg: str) -> None:
    FAILS.append(msg)


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s or "")).lower()
    s = s.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"[\W_]+", "", s, flags=re.UNICODE)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_prompts():
    """The live prompt registry, if the app package is importable."""
    for p in (str(REPO), str(REPO / "packages")):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        from core.application.chat.intent_funnel.argument_acquisition.prompts import prompt_for
        from core.application.chat.intent_funnel.argument_acquisition.extractor import (
            _user_prompt,
        )
        from types import SimpleNamespace
        return prompt_for, _user_prompt, SimpleNamespace
    except Exception as exc:  # pragma: no cover
        print(f"[validate] live prompt registry unavailable ({type(exc).__name__}); "
              f"skipping byte-match render check", file=sys.stderr)
        return None, None, None


def check_sample(s: dict, split: str, seen_ids: set, prompts) -> None:
    prompt_for, _user_prompt, SimpleNamespace = prompts
    sid = s.get("id")
    cap = s.get("capability_id")
    for key in ("id", "capability_id", "tool_name", "language", "tools", "tool_choice",
                "messages", "meta"):
        if key not in s:
            fail(f"[{split}] {sid}: missing key {key!r}")
            return
    if sid in seen_ids:
        fail(f"[{split}] duplicate id across the bundle: {sid}")
    seen_ids.add(sid)
    if cap not in C.SCHEMA:
        fail(f"[{split}] {sid}: unknown capability {cap!r}")
        return
    if s["tool_name"] != C.tool_name(cap):
        fail(f"[{split}] {sid}: tool_name {s['tool_name']!r} != {C.tool_name(cap)!r}")
    if s["tool_choice"] != "required":
        fail(f"[{split}] {sid}: tool_choice != 'required'")

    # tools: exactly one function, schema == the frozen model schema
    tools = s["tools"]
    if not isinstance(tools, list) or len(tools) != 1:
        fail(f"[{split}] {sid}: tools must be a single-element list")
        return
    fn = tools[0].get("function", {})
    if fn.get("name") != cap:
        fail(f"[{split}] {sid}: function.name {fn.get('name')!r} != capability_id {cap!r}")
    if fn.get("parameters") != C.model_schema(cap):
        fail(f"[{split}] {sid}: function.parameters != the frozen model schema")
    if fn.get("description") != C.function_description(cap):
        fail(f"[{split}] {sid}: function.description != the frozen registry description")

    # messages: system / user / assistant(tool_calls)
    msgs = s["messages"]
    if not isinstance(msgs, list) or len(msgs) != 3:
        fail(f"[{split}] {sid}: messages must be [system, user, assistant]")
        return
    if [m.get("role") for m in msgs] != ["system", "user", "assistant"]:
        fail(f"[{split}] {sid}: message roles != [system, user, assistant]")
        return
    sys_c, usr_c, asst = msgs[0].get("content"), msgs[1].get("content"), msgs[2]
    if not isinstance(sys_c, str) or not sys_c.strip():
        fail(f"[{split}] {sid}: empty system content")
    if not isinstance(usr_c, str) or "<user_sentence>" not in usr_c:
        fail(f"[{split}] {sid}: user content missing <user_sentence>")
    if prompt_for is not None:
        if sys_c != prompt_for(cap):
            fail(f"[{split}] {sid}: system content != live prompt_for({cap})")
        # reconstruct the query from the user message and re-render it live
        q = _extract_query(usr_c)
        if q is not None:
            rendered = _user_prompt(SimpleNamespace(parameters=C.SCHEMA[cap]["params"]),
                                    C.model_slots(cap),
                                    SimpleNamespace(query=q, user_turns=()))
            if usr_c != rendered:
                fail(f"[{split}] {sid}: user content != live-rendered user message")

    # assistant tool_calls target
    tcs = asst.get("tool_calls")
    if not isinstance(tcs, list) or len(tcs) != 1:
        fail(f"[{split}] {sid}: assistant must carry exactly one tool_call")
        return
    tfn = tcs[0].get("function", {})
    if tfn.get("name") != cap:
        fail(f"[{split}] {sid}: tool_call.function.name != capability_id")
    args = tfn.get("arguments")
    if not isinstance(args, str):
        fail(f"[{split}] {sid}: tool_call arguments must be a JSON string")
        return
    try:
        target = json.loads(args)
    except json.JSONDecodeError:
        fail(f"[{split}] {sid}: tool_call arguments is not valid JSON")
        return
    if not isinstance(target, dict):
        fail(f"[{split}] {sid}: tool_call arguments must decode to an object")
        return
    if "arguments" in target and "arguments" not in C.model_slots(cap):
        fail(f"[{split}] {sid}: nested {{'arguments': ...}} wrapper is forbidden")
    extra = set(target) - set(C.model_slots(cap))
    if extra:
        fail(f"[{split}] {sid}: target has non-model slots {sorted(extra)}")
    params = C.SCHEMA[cap]["params"]
    for slot, val in target.items():
        if params[slot]["type"] == "integer":
            if not isinstance(val, int) or isinstance(val, bool):
                fail(f"[{split}] {sid}: {slot} must be an integer, got {val!r}")
        else:
            if not isinstance(val, str) or not val.strip():
                fail(f"[{split}] {sid}: {slot} must be a non-empty string, got {val!r}")
            elif params[slot].get("max_len") and len(val) > params[slot]["max_len"]:
                fail(f"[{split}] {sid}: {slot} exceeds max_len {params[slot]['max_len']}")
    for slot in C.required_model_slots(cap):
        if slot not in target:
            fail(f"[{split}] {sid}: missing required model slot {slot!r}")
    # meta should agree with the arguments
    if s["meta"].get("target_slots") != list(target):
        fail(f"[{split}] {sid}: meta.target_slots != arguments keys")


def _extract_query(user_content: str) -> str | None:
    m = re.search(r"<user_sentence>(.*)</user_sentence>", user_content, flags=re.DOTALL)
    return m.group(1) if m else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify the frozen PE SFT bundle.")
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out = cfg["output"]

    def _out(p: str) -> Path:
        return Path(p) if Path(p).is_absolute() else (HERE / p)

    # output.* is script-dir-relative; source.* is repo-root-relative.
    data_dir, man_dir = _out(out["data_dir"]), _out(out["manifests_dir"])
    formal500 = REPO / cfg["source"]["formal500"]

    man_p = man_dir / "manifest.json"
    if not man_p.is_file():
        raise SystemExit(f"missing manifest: {man_p}")
    m = json.loads(man_p.read_text(encoding="utf-8"))

    # file hashes + counts
    for s in SPLITS:
        p = data_dir / f"{s}.jsonl"
        if not p.is_file():
            fail(f"missing data file: {p}")
            continue
        want = m.get("file_sha256", {}).get(f"{s}.jsonl")
        if want and sha256(p) != want:
            fail(f"{s}.jsonl: sha256 mismatch vs manifest")
    sums_p = man_dir / "SHA256SUMS"
    if sums_p.is_file():
        recorded = {}
        for line in sums_p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                h, name = line.split(None, 1)
                recorded[name.strip()] = h
        for s in SPLITS:
            p = data_dir / f"{s}.jsonl"
            if p.is_file() and recorded.get(f"{s}.jsonl") not in (None, sha256(p)):
                fail(f"SHA256SUMS disagrees with {s}.jsonl")

    prompts = _load_prompts()
    seen_ids: set = set()
    split_norm = {}
    split_target = {}
    actual_counts = {}
    for s in SPLITS:
        p = data_dir / f"{s}.jsonl"
        if not p.is_file():
            continue
        rows = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        actual_counts[s] = len(rows)
        seen_here: set = set()
        seen_target: set = set()
        for sample in rows:
            check_sample(sample, s, seen_ids, prompts)
            q = _extract_query(sample["messages"][1]["content"]) or ""
            nq = normalize(q)
            if nq in seen_here:
                fail(f"[{s}] duplicate normalized query within split: {sample['id']}")
            seen_here.add(nq)
            tgt = json.loads(sample["messages"][2]["tool_calls"][0]["function"]["arguments"])
            seen_target.add(normalize(_payload(tgt, sample["capability_id"])))
        split_norm[s] = seen_here
        split_target[s] = seen_target

    for s in SPLITS:
        if m["counts"].get(s) != actual_counts.get(s):
            fail(f"{s}: manifest count {m['counts'].get(s)} != actual {actual_counts.get(s)}")

    # cross-split + Formal-500 leakage
    union: set = set()
    union_t: set = set()
    for s in SPLITS:
        leak = union & split_norm.get(s, set())
        if leak:
            fail(f"{s} shares {len(leak)} normalized queries with an earlier split")
        union |= split_norm.get(s, set())
        leak_t = union_t & split_target.get(s, set())
        if leak_t:
            fail(f"{s} shares {len(leak_t)} normalized TARGET payloads with an earlier split")
        union_t |= split_target.get(s, set())
    if formal500.is_file():
        f5 = set()
        for ln in formal500.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                o = json.loads(ln)
                f5.add(normalize(o.get("query") or o.get("user_query") or ""))
        overlap = union & f5
        if overlap:
            fail(f"{len(overlap)} normalized queries overlap the frozen Formal-500")

    # per-group counts vs manifest
    grouped = Counter()
    for s in SPLITS:
        p = data_dir / f"{s}.jsonl"
        if p.is_file():
            for ln in p.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    r = json.loads(ln)
                    grouped[f"{r['capability_id']}/{r['language']}"] += 1
    for key, g in m.get("per_group_split", {}).items():
        tot = g.get("train", 0) + g.get("validation", 0) + g.get("test", 0)
        if grouped.get(key, 0) != tot:
            fail(f"per_group_split[{key}]: parts sum {tot} != {grouped.get(key, 0)} rows on disk")

    if FAILS:
        print(f"VALIDATION FAILED — {len(FAILS)} problem(s):", file=sys.stderr)
        for f in FAILS[:80]:
            print(f"  - {f}", file=sys.stderr)
        if len(FAILS) > 80:
            print(f"  ... and {len(FAILS) - 80} more", file=sys.stderr)
        raise SystemExit(1)

    print("VALIDATION PASSED")
    print(json.dumps(actual_counts, ensure_ascii=False))
    print(f"byte-render check: {'on' if prompts[0] is not None else 'skipped'}")


if __name__ == "__main__":
    main()
