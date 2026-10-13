# PE v2 → tool_calls SFT bundle (v1) — build report

source: PE v2 candidates @ repo commit `fdf96cf` · render source: `production` · split policy: `cluster`

## Counts

- raw candidates: **12883**
- isolated: **0**  by reason `{}`
- exact input duplicates dropped: **0**
- kept after audit + exact dedup: **12883**
- emitted (kept rows assigned to a split): **12883**
- train / validation / test: **10303 / 1290 / 1290**

## Label redundancy (why target-level clustering, not input dedup)

```{"distinct_targets": 4432, "rows": 12883, "mean_frames_per_target": 2.907, "max_frames_per_target": 158}```

PE v2 repeats the same extraction TARGET across many instruction frames, so raw text dedups to ~0 while targets are far fewer than rows. Near-dup is clustered on the TARGET, and each cluster (one target + its paraphrase frames) is assigned to a single split, so no target spans train/validation/test.

## Label re-derivation

Model target = `{slot: expected_arguments[slot] for slot in MODEL_SLOTS[cap]}`, with `domain` taken from its human-readable surface (`domain_surface`).

### domain surface corrections (slug → name)

```
{}
```

### derivation flag counts

```
{}
```

### audit warning counts

```
{}
```

## Dedup / leakage

- clusters: **4375**
- cluster size histogram: `{"1": 3465, "2": 88, "6": 9, "7": 9, "8": 41, "9": 84, "10": 126, "11": 158, "12": 181, "13": 138, "14": 64, "15": 11, "16": 1}`
- overlap with Formal-500 (normalized query): **0** (must be 0)
- cross-split normalized query overlap: **0** (must be 0)
- cross-split normalized target overlap: **0** (must be 0)

## Per (capability/language) split

```
{
  "cap-add-term/en": {
    "rows": 1011,
    "clusters": 128,
    "train": 809,
    "validation": 101,
    "test": 101
  },
  "cap-add-term/zh": {
    "rows": 1016,
    "clusters": 133,
    "train": 812,
    "validation": 102,
    "test": 102
  },
  "cap-create-folder/en": {
    "rows": 1110,
    "clusters": 236,
    "train": 835,
    "validation": 111,
    "test": 111
  },
  "cap-create-folder/zh": {
    "rows": 1095,
    "clusters": 236,
    "train": 928,
    "validation": 110,
    "test": 110
  },
  "cap-rag-search/en": {
    "rows": 1011,
    "clusters": 191,
    "train": 809,
    "validation": 101,
    "test": 101
  },
  "cap-rag-search/zh": {
    "rows": 1016,
    "clusters": 196,
    "train": 812,
    "validation": 102,
    "test": 102
  },
  "cap-social-search/en": {
    "rows": 1210,
    "clusters": 389,
    "train": 968,
    "validation": 121,
    "test": 121
  },
  "cap-social-search/zh": {
    "rows": 1315,
    "clusters": 495,
    "train": 1051,
    "validation": 132,
    "test": 132
  },
  "cap-translate/en": {
    "rows": 1035,
    "clusters": 991,
    "train": 827,
    "validation": 104,
    "test": 104
  },
  "cap-translate/zh": {
    "rows": 1037,
    "clusters": 993,
    "train": 829,
    "validation": 104,
    "test": 104
  },
  "cap-web-search/en": {
    "rows": 1013,
    "clusters": 193,
    "train": 811,
    "validation": 101,
    "test": 101
  },
  "cap-web-search/zh": {
    "rows": 1014,
    "clusters": 194,
    "train": 812,
    "validation": 101,
    "test": 101
  }
}
```

## File SHA-256

```
{
  "train.jsonl": "2fadeaeb33ccba0876f4e058fe5ca6578025cfa31faa8505446916b9ad12b63d",
  "validation.jsonl": "5974e0a9acbd14025632986e240b27b6e3a162f92ac74904ac98f56af70589e1",
  "test.jsonl": "5c7649326e46b710a915b0c41e0f1f625526280d43712a683cd0c6e851905c4c"
}
```

## Notes

- Protocol: single function (name == capability_id) + tool_choice=required; schema exposes ONLY model-owned slots (Phase-3b face narrowing).
- assistant target uses tool_calls; function.arguments is a JSON string of {slot:value}, with no nested {"arguments": {...}} wrapper.
- ONLINE ALIGNED: the live extractor sends the SAME tools request (one function named for the pinned capability_id, tool_choice=required) and reads the reply from message.tool_calls; a name mismatch or non-object arguments is a hard ExtractionUnavailable (no content fallback).
