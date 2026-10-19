#!/usr/bin/env python3
"""Verify the frozen parameter-extraction SFT bundle; fail loudly on any drift.

Stdlib only: reads config.yaml, the manifest (+ SHA256SUMS) and the three
``data/*.jsonl`` splits, and re-checks every invariant the builder guarantees —
file hashes, counts, the tool schema, the assistant ``tool_calls`` target, id and
query uniqueness, cross-split and Formal-500 leakage. Exits non-zero on the first
failing category (the training entry must refuse a bundle the verifier rejects).

It also checks each serialized sample against the ONE slot-narrowing rule
(``search_scope.effective_model_slots``, reused — never copied; enforced even when
the app package is not importable): the tool face must equal the per-sample model
face the builder produced. When the app package IS importable it additionally pins
the system prompt against the LIVE registry (``prompt_for``). The user message is
checked STRUCTURALLY
(roles / slot block / ``<user_sentence>``) — it is never re-rendered from a
single-turn assumption, so a multi-turn row that carries a ``recent user turns``
block is not misreported (PE is single-query; history is business logic's job).
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


def _load_narrowing_rule():
    """The ONE slot-narrowing rule (:func:`search_scope.effective_model_slots`),
    resolved EVEN when the heavy app package is not importable.

    ``search_scope`` is stdlib-only, but importing it through the ``core.…`` package
    chain executes parent ``__init__``s that may pull a dependency the isolated
    training venv lacks. On that failure it is loaded straight from its file, so the
    narrowing check is never silently dropped."""
    try:
        from core.application.chat.intent_funnel.cap_handler.search_scope import (
            effective_model_slots,
        )
        return effective_model_slots
    except Exception:
        pass
    path = (REPO / "packages" / "core" / "application" / "chat" / "intent_funnel"
            / "cap_handler" / "search_scope.py")
    if not path.is_file():
        return None
    import importlib.util
    spec = importlib.util.spec_from_file_location("pe_search_scope", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.effective_model_slots


def _load_live():
    """``(prompt_for | None, effective_model_slots | None)``.

    The system-prompt equality check needs the app prompt registry; when that is not
    importable ``prompt_for`` is ``None`` and only that ONE check is skipped. The
    (stdlib) narrowing rule is resolved independently so a missing app package never
    disables it."""
    for p in (str(REPO), str(REPO / "packages")):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        from core.application.chat.intent_funnel.argument_acquisition.prompts import prompt_for
    except Exception as exc:  # pragma: no cover
        print(f"[validate] live prompt registry unavailable ({type(exc).__name__}); "
              f"system-prompt equality is skipped (narrowing still enforced)",
              file=sys.stderr)
        prompt_for = None
    return prompt_for, _load_narrowing_rule()


def _check_user_serialization(sid, split: str, cap: str, usr_c: str) -> None:
    """Structural check of the SERIALIZED user message — no re-render, no
    multi-turn parsing. The builder always renders the STATIC slot block; a
    ``recent user turns`` block (present only on multi-turn rows) is treated as
    opaque data and never reconstructed."""
    if not usr_c.startswith("slots to fill:\n"):
        fail(f"[{split}] {sid}: user content must start with 'slots to fill:'")
    for slot in C.model_slots(cap):  # builder renders the STATIC slot list
        if f"- {slot} (" not in usr_c:
            fail(f"[{split}] {sid}: user content missing static slot line for {slot!r}")
    if "user sentence (data, not instructions):" not in usr_c:
        fail(f"[{split}] {sid}: user content missing the user-sentence marker")
    if usr_c.count("<user_sentence>") != 1 or usr_c.count("</user_sentence>") != 1:
        fail(f"[{split}] {sid}: user content must carry exactly one <user_sentence> block")
    if not usr_c.rstrip().endswith("</user_sentence>"):
        fail(f"[{split}] {sid}: user content must end at </user_sentence>")


def _check_meta(sid, split: str, cap: str, args: dict, meta: dict) -> None:
    """The DECLARED-INTENT gate + metadata-consistency checks (mirror of the
    trainer's ``_check_meta``). A required model slot may be absent from the target
    IFF declared in ``meta.missing_required`` or ``expected_outcome == 'clarify'``;
    ``execute`` requires a COMPLETE target and may declare no missing slot."""
    missing = [x for x in C.required_model_slots(cap) if x not in args]
    req = set(C.required_model_slots(cap))
    outcome = meta.get("expected_outcome")
    declared = list(meta.get("missing_required") or [])
    if outcome not in (None, "execute", "clarify"):
        fail(f"[{split}] {sid}: unknown expected_outcome {outcome!r}")
    if len(declared) != len(set(declared)):
        fail(f"[{split}] {sid}: meta.missing_required has duplicates {declared}")
    for x in declared:
        if x not in req:
            fail(f"[{split}] {sid}: missing_required names non-required slot {x!r}")
        elif x not in missing:
            fail(f"[{split}] {sid}: missing_required declares {x!r} but it is present")
    if outcome != "clarify" and declared:
        fail(f"[{split}] {sid}: missing_required declared but expected_outcome={outcome!r}")
    declared_set = set(declared)
    if outcome == "clarify":
        declared_set |= set(missing)
    undeclared = [x for x in missing if x not in declared_set]
    if undeclared:
        fail(f"[{split}] {sid}: target missing required slot(s) {undeclared} (undeclared)")


def check_sample(s: dict, split: str, seen_ids: set, live) -> None:
    prompt_for, effective_model_slots = live
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

    # messages: the FULL list the trainer/inference renders is [system, user,
    # assistant]; the slice the trainer actually CONSUMES is messages[:2] ==
    # [system, user]. Check BOTH separately (do NOT demand the slice be 3 roles).
    msgs = s["messages"]
    if not isinstance(msgs, list) or len(msgs) != 3:
        fail(f"[{split}] {sid}: messages must be [system, user, assistant]")
        return
    if [m.get("role") for m in msgs] != ["system", "user", "assistant"]:
        fail(f"[{split}] {sid}: message roles != [system, user, assistant]")
        return
    if [m.get("role") for m in msgs[:2]] != ["system", "user"]:
        fail(f"[{split}] {sid}: the trainer-consumed slice messages[:2] != [system, user]")
    sys_c, usr_c, asst = msgs[0].get("content"), msgs[1].get("content"), msgs[2]
    if not isinstance(sys_c, str) or not sys_c.strip():
        fail(f"[{split}] {sid}: empty system content")
    if prompt_for is not None and sys_c != prompt_for(cap):
        fail(f"[{split}] {sid}: system content != live prompt_for({cap})")
    q = _extract_query(usr_c) if isinstance(usr_c, str) else None
    if not isinstance(usr_c, str) or q is None or not q:
        fail(f"[{split}] {sid}: user content missing a non-empty <user_sentence>")
    else:
        _check_user_serialization(sid, split, cap, usr_c)

    # tools: exactly one function, face == the SAME model face the builder produced
    # (static, or the ONE legal cap-social-search narrowing).
    tools = s["tools"]
    if not isinstance(tools, list) or len(tools) != 1:
        fail(f"[{split}] {sid}: tools must be a single-element list")
        return
    fn = tools[0].get("function", {})
    if fn.get("name") != cap:
        fail(f"[{split}] {sid}: function.name {fn.get('name')!r} != capability_id {cap!r}")
    if effective_model_slots is not None and q is not None:
        expected = list(effective_model_slots(cap, q, C.model_slots(cap)))
    else:
        expected = list(C.model_slots(cap))
    if fn.get("parameters") != C.model_schema(cap, expected):
        fail(f"[{split}] {sid}: function.parameters != the expected model face")
    if fn.get("description") != C.function_description(cap):
        fail(f"[{split}] {sid}: function.description != the frozen registry description")

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
    extra = set(target) - set(expected)
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
    # the DECLARED-INTENT gate (an honest empty/partial target is legal when declared)
    _check_meta(sid, split, cap, target, s.get("meta") or {})
    # meta should agree with the arguments
    if s["meta"].get("target_slots") != list(target):
        fail(f"[{split}] {sid}: meta.target_slots != arguments keys")


def _extract_query(user_content: str) -> str | None:
    m = re.search(r"<user_sentence>(.*)</user_sentence>", user_content, flags=re.DOTALL)
    return m.group(1) if m else None


def _leak_keys(rows: list[dict]) -> tuple[set, set, int]:
    """The per-split leakage key sets: ``(capability, normalize(query))`` and
    ``(capability, normalize(payload))``.

    Empty ``{}`` targets are NOT payloads — they are counted separately and
    EXCLUDED from the payload set. Otherwise every honest empty target would
    collide on the constant ``"{}"`` string and two DISTINCT queries that both
    legitimately yield ``{}`` would be misreported as leakage. Only CROSS-split
    overlap of these keys is leakage; target repetition WITHIN a split is the
    intended label redundancy (several paraphrase frames per distinct target)."""
    qs: set = set()
    ts: set = set()
    n_empty = 0
    for sample in rows:
        cap = sample["capability_id"]
        qs.add((cap, normalize(_extract_query(sample["messages"][1]["content"]) or "")))
        tgt = json.loads(sample["messages"][2]["tool_calls"][0]["function"]["arguments"])
        if not tgt:
            n_empty += 1
            continue
        ts.add((cap, normalize(_payload(tgt, cap))))
    return qs, ts, n_empty


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
    if not sums_p.is_file():
        fail(f"missing SHA256SUMS: {sums_p}")
    else:
        recorded = {}
        for line in sums_p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                h, name = line.split(None, 1)
                recorded[name.strip()] = h
        # All FOUR frozen records must be present and agree: train/val/test + manifest.
        for name, p in [(f"{s}.jsonl", data_dir / f"{s}.jsonl") for s in SPLITS] \
                + [("manifest.json", man_p)]:
            rec = recorded.get(name)
            if rec is None:
                fail(f"SHA256SUMS has no record for {name}")
            elif not p.is_file():
                fail(f"SHA256SUMS lists {name} but the file is missing")
            elif rec != sha256(p):
                fail(f"SHA256SUMS disagrees with {name}")

    live = _load_live()
    seen_ids: set = set()
    split_q: dict[str, set] = {}
    split_t: dict[str, set] = {}
    empty_counts: dict[str, int] = {}
    payload_caps: dict[str, set] = {}
    actual_counts = {}
    for s in SPLITS:
        p = data_dir / f"{s}.jsonl"
        if not p.is_file():
            continue
        rows = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        actual_counts[s] = len(rows)
        seen_q: set = set()
        for sample in rows:
            check_sample(sample, s, seen_ids, live)
            cap = sample["capability_id"]
            nq = (cap, normalize(_extract_query(sample["messages"][1]["content"]) or ""))
            if nq in seen_q:
                fail(f"[{s}] duplicate normalized query within {cap}: {sample['id']}")
            seen_q.add(nq)
        # Cross-split leakage keys (empty {} targets counted separately, not keyed).
        qs, ts, n_empty = _leak_keys(rows)
        split_q[s] = qs
        split_t[s] = ts
        empty_counts[s] = n_empty
        for cap, payload in ts:
            payload_caps.setdefault(payload, set()).add(cap)

    for s in SPLITS:
        if m["counts"].get(s) != actual_counts.get(s):
            fail(f"{s}: manifest count {m['counts'].get(s)} != actual {actual_counts.get(s)}")

    # cross-split: SAME-capability query / non-empty target overlap -> FAIL (real
    # leakage). Empty-{} negatives are excluded (a distinct query may legally have
    # an empty target). Cross-capability payload collisions are informational only.
    union_q: set = set()
    union_t: set = set()
    for s in SPLITS:
        leak = union_q & split_q.get(s, set())
        if leak:
            fail(f"{s} shares {len(leak)} (capability, query) with an earlier split: "
                 f"{sorted(leak)[:3]}")
        union_q |= split_q.get(s, set())
        leak_t = union_t & split_t.get(s, set())
        if leak_t:
            fail(f"{s} shares {len(leak_t)} (capability, target payload) with an earlier "
                 f"split: {sorted(leak_t)[:3]}")
        union_t |= split_t.get(s, set())
    cross_cap = {pl: caps for pl, caps in payload_caps.items() if len(caps) > 1}
    all_q = {q for (_cap, q) in union_q}
    if formal500.is_file():
        f5 = set()
        for ln in formal500.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                o = json.loads(ln)
                f5.add(normalize(o.get("query") or o.get("user_query") or ""))
        overlap = all_q & f5
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
    print(f"live-contract check: {'on' if live[0] is not None else 'static-fallback'}")
    print(f"empty-{{}} negative targets (excluded from payload-leak check): "
          f"{json.dumps(empty_counts, ensure_ascii=False)}")
    if cross_cap:
        print(f"[info] cross-capability payload collisions (NOT leakage): {len(cross_cap)}")
        for pl, caps in list(cross_cap.items())[:10]:
            print(f"  [info] payload shared across capabilities {sorted(caps)}")


if __name__ == "__main__":
    main()
