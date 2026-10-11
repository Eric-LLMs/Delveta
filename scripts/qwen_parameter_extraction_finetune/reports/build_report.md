# PE v2 → tool_calls SFT bundle (v1) — build report

source: PE v2 candidates @ repo commit `fdf96cf` · render source: `production` · split policy: `cluster`

## Counts

- raw candidates: **12000**
- isolated: **1200**  by reason `{"clarify_required_missing": 1200}`
- exact input duplicates dropped: **0**
- kept after audit + exact dedup: **10800**
- emitted (kept rows assigned to a split): **10800**
- train / validation / test: **8679 / 1041 / 1080**

## Label redundancy (why target-level clustering, not input dedup)

```{"distinct_targets": 4203, "rows": 10800, "mean_frames_per_target": 2.57, "max_frames_per_target": 15}```

PE v2 repeats the same extraction TARGET across many instruction frames, so raw text dedups to ~0 while targets are far fewer than rows. Near-dup is clustered on the TARGET, and each cluster (one target + its paraphrase frames) is assigned to a single split, so no target spans train/validation/test.

## Label re-derivation

Model target = `{slot: expected_arguments[slot] for slot in MODEL_SLOTS[cap]}`, with `domain` taken from its human-readable surface (`domain_surface`).

### domain surface corrections (slug → name)

```
{
  "cap-add-term": 1868,
  "cap-rag-search": 360
}
```

### derivation flag counts

```
{
  "domain_surface_applied": 2228
}
```

### audit warning counts

```
{}
```

## Dedup / leakage

- clusters: **2600**
- cluster size histogram: `{"1": 1800, "6": 1, "7": 10, "8": 32, "9": 84, "10": 132, "11": 162, "12": 185, "13": 132, "14": 56, "15": 6}`
- overlap with Formal-500 (normalized query): **0** (must be 0)
- cross-split normalized query overlap: **0** (must be 0)
- cross-split normalized target overlap: **0** (must be 0)

## Per (capability/language) split

```
{
  "cap-add-term/en": {
    "rows": 900,
    "clusters": 80,
    "train": 726,
    "validation": 84,
    "test": 90
  },
  "cap-add-term/zh": {
    "rows": 900,
    "clusters": 80,
    "train": 721,
    "validation": 89,
    "test": 90
  },
  "cap-create-folder/en": {
    "rows": 900,
    "clusters": 80,
    "train": 720,
    "validation": 90,
    "test": 90
  },
  "cap-create-folder/zh": {
    "rows": 900,
    "clusters": 80,
    "train": 725,
    "validation": 85,
    "test": 90
  },
  "cap-rag-search/en": {
    "rows": 900,
    "clusters": 80,
    "train": 725,
    "validation": 85,
    "test": 90
  },
  "cap-rag-search/zh": {
    "rows": 900,
    "clusters": 80,
    "train": 724,
    "validation": 86,
    "test": 90
  },
  "cap-social-search/en": {
    "rows": 900,
    "clusters": 80,
    "train": 725,
    "validation": 85,
    "test": 90
  },
  "cap-social-search/zh": {
    "rows": 900,
    "clusters": 80,
    "train": 720,
    "validation": 90,
    "test": 90
  },
  "cap-translate/en": {
    "rows": 900,
    "clusters": 900,
    "train": 720,
    "validation": 90,
    "test": 90
  },
  "cap-translate/zh": {
    "rows": 900,
    "clusters": 900,
    "train": 720,
    "validation": 90,
    "test": 90
  },
  "cap-web-search/en": {
    "rows": 900,
    "clusters": 80,
    "train": 727,
    "validation": 83,
    "test": 90
  },
  "cap-web-search/zh": {
    "rows": 900,
    "clusters": 80,
    "train": 726,
    "validation": 84,
    "test": 90
  }
}
```

## File SHA-256

```
{
  "train.jsonl": "dbd06f3a3e8e30a0d6395ec0fb5d80f3bf0ba85ea4bb7a9dfd0cd78ddab51222",
  "validation.jsonl": "addc0baa8fa61624ad054a54dfcac43e15ec27f03b985a61543db18d3185f949",
  "test.jsonl": "507bc036b4d9a15a797097e8074ba7471abcf2188f3ab0ff85458b49281327e2"
}
```

## Notes

- Protocol: single function (name == capability_id) + tool_choice=required; schema exposes ONLY model-owned slots (Phase-3b face narrowing).
- assistant target uses tool_calls; function.arguments is a JSON string of {slot:value}, with no nested {"arguments": {...}} wrapper.
- ONLINE ALIGNED: the live extractor sends the SAME tools request (one function named for the pinned capability_id, tool_choice=required) and reads the reply from message.tool_calls; a name mismatch or non-object arguments is a hard ExtractionUnavailable (no content fallback).
