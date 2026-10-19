# LayaChoice-v2 — Fine-Tuning Laya with an Explicit REJECT Option

> **Status: released.** The formal run is complete and the artifacts are closed.
> This document is the record of that experiment; the stable architectural facts distilled
> from it live in [architecture.md §26](../architecture.md#26-layachoice-capability-selection).
>
> **Production integration status.** v2 is wired into production: the `cap_router` card
> renderer emits the `REJECT` criterion (§6), the `deploy/laya` sidecar loads the v2
> checkpoint, and the service backend maps a `REJECT` answer to a normal fourth decision
> routed to the Agent. It runs as the `cap-router` Compose service.
>
> Everything below is derived from the run artifacts (`fine_tuning/laya_finetune/V2/` and the
> frozen bundle `fine_tuning/laya_finetune/V2/data/`), not from recollection.

> **Contract correction — V1 is fixed, V2 is dynamic.** V1 is the fixed-K build (**K = 3**).
> V2's current contract is **dynamic K+1**: K capability cards (0..3) plus one `REJECT` card,
> so a question carries **1..4 options**. The formal run recorded in this document used the
> earlier **flat 4-way** build (K = 3 + `REJECT`); the installed `data/recall_threshold_*`
> bundles are V4 (dynamic K+1) and no V4 training run has been recorded here. Sections that
> state fixed counts describe that historical run, not the current V2 contract.

## 1. Experiment Overview

| | |
|---|---|
| Task | Choose the correct capability for a user request from the recalled candidates, **or** reject the set. Current V2: dynamic K+1 (K = 0..3, 1..4 options). The formal run below used the flat 4-way build (K = 3 + `REJECT`) |
| Base model | `convaiinnovations/laya`, subfolder `multilingual`, revision `55cf4c4eb…` |
| Output | `Delveta LayaChoice v2` — full-parameter fine-tune, FP32 export |
| Data | Train 2 112 / Calibration 211 / Validation v3 300 / Final Test v3 1 400 (all frozen) |
| Input config | `OPTION_MAX_TOKENS=256`, `head_max_len=768`, `max_len=1024` |
| Best epoch | **7** (selected by Validation v3 top-1 only) |
| Validation v3 (best) | **0.966667** = 290/300 |
| Final Test v3 | **0.877857** — EN 0.861272 / ZH 0.894068 |
| REJECT on Final Test | recall 0.719298 / precision 0.557823 / FPR 0.050544 |
| Export | `model.safetensors`, FP32 preserve, 1 287 653 720 B, sha256 `c6331bd5…` |

One sentence: **v1 is the fixed 3-way build; v2 adds an explicit `REJECT` option and its
current contract is dynamic K+1 (1..4 options)**. The formal run recorded here used the flat
4-way build (K = 3 + `REJECT`), with the best checkpoint chosen on a separate validation
split the test set never touched.

## 2. Objective

v1 answered "which of these 3 capabilities handles the request?". That framing has no way
to say *"none of them"* — a turn whose true capability was not retrieved is forced onto a
wrong candidate. The objective of v2 was to close that gap with a number:

> Can the same base decision model, fine-tuned with a dedicated `REJECT` option as a
> fourth class, learn to abstain when the correct capability is not among the candidates —
> without giving up capability-selection accuracy?

Not objectives: not a general chat model, not a production-traffic model, and not a
re-opening of the v1 dataset (v1 is frozen, §5).

## 3. From fixed 3-way (V1) to dynamic K+1 (V2)

V1 is **fixed**: every row carries exactly three capability cards, and the model has no
"none" output. V2 makes abstention a first-class decision and makes the option count
dynamic:

- **Current V2 contract:** K capability cards (K = 0..3) plus one `REJECT` card, so every
  question carries **K + 1 options (1..4)** (`MIN_OPTION_SLOTS = 1`,
  `MAX_OPTION_SLOTS = 4`). `OPTION_SLOTS = 4` remains the frozen flat-v2 constant.
- `REJECT` is a **normal option**, not a threshold and not a post-hoc rule — the model
  assigns it probability mass directly.
- The option order is a **per-row deterministic K+1 shuffle** (seed prefix
  `reject-shuffle-v1`, recorded in each row's `shuffle` field), so the model cannot learn a
  positional prior.
- `gold_index` marks the target slot within the row's actual option count and
  `target_kind ∈ {capability, reject}` records which kind of target the row has.

**Historical run.** The formal run recorded in this document used the earlier flat 4-way
build: every row carried three capability cards plus one `REJECT` card, and
`gold_index ∈ {0,1,2,3}`. That build is the source of the measured numbers below. The
installed `data/recall_threshold_*` bundles are V4 (dynamic K+1); no V4 training run is
recorded here.

In both builds, **a caller must render the `REJECT` card**. A question without it does not
match the trained pipeline and its scores are not meaningful.

## 4. Base Model

| | |
|---|---|
| Repo | `convaiinnovations/laya` |
| Revision | `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` (asserted on load) |
| Subfolder | `multilingual` |
| Encoder | `jhu-clsp/mmBERT-base` (~322 M params) |
| Package | `laya==0.3.21` (pinned; inference only — the wheel ships no trainer) |
| Checkpoint temperature | `[1.0, 1.0, 1.0]` on the base |

The upstream repository went private after the v1 work started, which is why the recipe is
pinned to an exact revision and to `laya==0.3.21`.

## 5. Dataset Definition

**LayaChoice-v2** is the v1 capability-selection benchmark extended with a `REJECT` option.
Its current contract is dynamic K+1; the formal run recorded here used the flat 4-way build.
It is derived from the frozen v1 bundle (`B_noprov` view), not rebuilt from the Registry:

| | |
|---|---|
| Card view | `B_noprov` — the production card with the `query examples:` block and the `evidence:` provenance line removed |
| Candidates per row | **Current V2: K capability cards (0..3) + 1 `REJECT` card, 1..4 options.** Formal run: **K = 3 + `REJECT`** (`OPTION_SLOTS = 4`) |
| Capabilities | 17 + `REJECT` = 18 possible target slots in the formal run; the current Registry has 18 active capabilities (see §9) |
| Languages | `zh` / `en` |
| Row shape | Current V2: `{id, lang, state, order[K+1], options[K+1], gold, gold_index, target_kind, part, …}`; formal run: `order[4]`, `options[4]` |

The `REJECT` card text is frozen for all rows:

> `### REJECT` / `tool: none` / `does: No listed capability correctly handles the user's
> request. Choose this option only when none of the other options is the right
> capability.` plus a Chinese line.

`REJECT` rows are of two kinds: *natural* (the source query genuinely had no matching
capability) and *synthetic* (a capability row whose gold card was swapped for `REJECT`).
Both are training/exam items; the frozen bundle records which is which, and that metadata
is **never rendered** — only `state` and the four option cards enter the sequence.

v1's label-leakage argument still holds unchanged: the stored `evidence:` line names the
answer, so it stays dropped. The `B_noprov` view is what makes the task a decision rather
than a lookup.

## 6. Splits, Calibration and the Adjustment

```
source_train_rows            2 112  ── exclude the frozen 211 calibration rows (seed 20260922)
calibration_rows               211  ──► ids recorded in manifest.json
effective_train_rows         1 901  ── after exclusion (world-size aligned)
Validation v3                  300  ── per-epoch top-1; the ONLY selection signal
Final Test v3                1 400  ── benchmark of best_epoch only, never used to pick or gate
```

- **Language** — train `zh` 1 084 / `en` 1 028; val `zh` 154 / `en` 146; test `en` 692 /
  `zh` 708.
- **Partitions** — every split is half `part1` / half `part2` (train 1 056/1 056, val
  150/150, test 700/700). `part1` is the v1-style candidate set (REJECT added); `part2` is
  the Recall-based candidate set (self-hit removed). The final schema is identical across
  parts.
- **Calibration rule** — `min(400, n_train // 10)` over the FINAL (post-shuffle) train
  order, seed `20260922`, 211 ids recorded by index; excluded from training by
  construction.
- **Test→Train adjustment** — 200 whole source queries (400 rows: one `p1` + one `p2` per
  source) were moved from Test into Train; the **only** field changed on any row is
  `split`. The pre-adjustment 1 800-row test is kept as backup material
  (`V2/frozen_backup/v2_test_1800_pre_adjustment.jsonl`). The FINAL splits are the ones
  above.

Train / validation / test are disjoint, and the calibration ids are excluded before any
world-size truncation, so no rank can pull a calibration row into training.

## 7. Frozen Dataset SHA256

Verified unchanged at the end of the run:

| Artifact | SHA256 |
|---|---|
| `v2_train.jsonl` | `c92c0b5f6b5fbb019f6cf1e89a107ad4d1c4fee952f4d1a024c6cb319873b720` |
| `v2_val.jsonl` | `f7d97670c96c13330f07dad2a3aecabbae42b3194e54c2cf0dafddc8af51de7b` |
| `v2_test.jsonl` | `eb14e0455156d6dd3f2b0ddf2fa82feaf5248b10f993a4ded57cea45261fb9ea` |
| `manifest.json` | `224acf2f3bdec01ab6284942c1f8b87acaf9daa7c924e89a5c8e97b485dbf57e` |
| `v2_build_report.json` | `e2a3dae1d7228ea31b7d318473edd6a795a4561b62b7ebc18f4aad8dd6fc0703` |
| `v2_construction_audit.jsonl` | `2464e187a667d515b97f0082e18d37b66b2b3c377b6f54f681a265c7e4b6091a` |

Source (v1) bundles the construction read from:

| Artifact | SHA256 |
|---|---|
| `layachoice_v1_train_b_noprov.jsonl.gz` | `e968fdeddc8f02eec19f37ed29fafc5f05a591ea259c97315639d366e3de4f13` |
| `layachoice_v1_val_b_noprov.jsonl.gz` | `3fa38aa099580b6f9a5146ac2a8e7beed0111d79d2f994adb6ca317547c90364` |
| `layachoice_v1_test_b_noprov.jsonl.gz` | `c665004a127496f4456a0b3ccd81bd0ce10654fef7808f27e45500d3a9d9b8bb` |

## 8. Candidate Contract

V1 is fixed at **K = 3**. V2's current contract is **dynamic K+1**: the model receives K
capability cards plus one `REJECT` card, and emits a choice over **K + 1 options (1..4)**.

| Candidate count | Behaviour |
|---|---|
| K = 0 | caller short-circuits on the empty set; LayaChoice is not called |
| K = 1 | 2 options (1 capability + `REJECT`) |
| K = 2 | 3 options (2 capabilities + `REJECT`) |
| K = 3 | 4 options (3 capabilities + `REJECT`) — the formal-run case |
| K ≥ 4 | the business layer truncates to top-3 **first** (`normalize_top_k`), then LayaChoice |

**Production candidate normalization must happen before LayaChoice**, not inside it. The
formal run recorded here was trained and measured at exactly three capability candidates
**plus the `REJECT` card**; K=0..2 and K≥4 post-truncation are **not** exercised by any
number in this document. The current dynamic bundle (V4) has not been trained. In every
case, a caller must render `REJECT` as one of the criteria.

## 9. Token Budget

The v2 input builder is a **byte-verbatim copy** of `laya.common.build_sequence` with
exactly one line changed (`max_length=48` → `256`); the stock even-share fallback and the
head-budget floor are preserved. The library default of 48 tokens per option would cut
every frozen card, so the cap is raised to 256.

The frozen configuration:

| Constant | Value | Meaning |
|---|---|---|
| `OPTION_MAX_TOKENS` | **256** | per-option hard cap (capability cards and `REJECT`) |
| `HEAD_MAX_LEN` | **768** | budget the question head **and all options (up to four) share** |
| `MAX_LEN` | **1024** | whole-sequence ceiling |

### Capability Card Token Audit and Compression

#### Budget Rationale

The three constants constrain different levels of one input sequence and must not be
treated interchangeably:

- `OPTION_MAX_TOKENS = 256` — a **hard per-option cap**. It applies independently to each
  capability card and to the `REJECT` card.
- `HEAD_MAX_LEN = 768` — a **shared budget**, not a per-card length. The question head and
  all options (K + 1, up to four) draw from this one pool
  (`opt_budget = head_max_len − Σoptions`). When the pool drops below 16 tokens the
  even-share fallback shrinks the options and the instruction head together.
- `MAX_LEN = 1024` — the ceiling on the **complete assembled sequence**, including the
  `[CLS]`, question head, option markers, option cards, serialized state and `[SEP]`.
  State text is right-truncated at this level.

The library default for the per-option cap is 48 tokens (`laya.common.build_sequence`
hard-codes `max_length=48`). The frozen `B_noprov` cards are longer than that, so the cap
is raised to 256 in both the training builder and the serving builder. This does not
change the meaning of `HEAD_MAX_LEN` or `MAX_LEN`: it only lifts the per-option ceiling
inside the shared head+options budget.

#### Capability Card Token Statistics

**Audit method.**

- **Source:** the live Registry (`capabilities`, 18 enabled cards), read through the same
  loader the serving path uses (`registry.store.list_capabilities`).
- **Renderer:** the production `render_card()` in
  `packages/core/application/chat/intent_funnel/cap_router/card_renderer.py`.
- **Tokenizer:** the project's actual Laya multilingual tokenizer at the pinned base
  revision, called the same way `build_sequence` calls it (`add_special_tokens=False` on
  `" " + card`; truncation at `OPTION_MAX_TOKENS`).
- **Raw tokens:** the card body only, with no `cap-<id>:` label.
- **Special Tokens:** everything the option slot adds around the card body — the
  `cap-<id>:` label and the leading `[MASK]` marker. It is measured as Total − Raw, so it
  varies with the label's tokenization rather than being a fixed 1.
- **Total tokens:** the complete option encoding that reaches the model — the
  `cap-<id>:` label, the card body and the `[MASK]` marker — after the 256-token
  per-option truncation. It is measured, not a sum of the other columns.
- **Per-option cap:** `OPTION_MAX_TOKENS = 256`.

| Capability card | Raw tokens | Special Tokens | Total tokens | Per-option cap |
|---|---:|---:|---:|---:|
| `cap-social-search` | 243 | 7 | 250 | 256 |
| `cap-read-document` | 224 | 7 | 231 | 256 |
| `cap-vision` | 224 | 5 | 229 | 256 |
| `cap-rag-search` | 216 | 7 | 223 | 256 |
| `cap-add-term` | 204 | 7 | 211 | 256 |
| `cap-edit-file` | 200 | 7 | 207 | 256 |
| `cap-pdf-extract-text` | 195 | 9 | 204 | 256 |
| `cap-web-search` | 193 | 7 | 200 | 256 |
| `cap-pdf-table-to-text` | 191 | 11 | 202 | 256 |
| `cap-artifact` | 190 | 5 | 195 | 256 |
| `cap-translate` | 164 | 5 | 169 | 256 |
| `cap-bash` | 149 | 5 | 154 | 256 |
| `cap-create-folder` | 148 | 7 | 155 | 256 |
| `cap-slides` | 148 | 5 | 153 | 256 |
| `cap-mindmap` | 147 | 6 | 153 | 256 |
| `cap-summary` | 141 | 5 | 146 | 256 |
| `cap-read-file` | 99 | 7 | 106 | 256 |
| `cap-research` | 40 | 5 | 45 | 256 |

All 18 cards fit within the 256-token per-option budget; no card loses text to
truncation. `cap-social-search` is the longest (243 raw / 250 total), followed by
`cap-read-document` (224 / 231) and `cap-vision` (224 / 229). The remaining cards are at or
below 216 raw tokens.

This is not a re-measurement of the frozen v2 training bundle. That run documented 17
capabilities and cards of 136–230 tokens, which is why the "no option card is truncated"
statement above holds for the released experiment. The current Registry has 18 active
capability cards. Which capability was absent from the frozen 17-card set has not been
verified and is not asserted here.

#### Full Card Prompts

The card texts below are the measured final forms used for the token counts.

##### `cap-vision` (224 tokens)

```
### cap-vision
tool: vision
does: Analyze an image (screenshot, chart, photo, diagram). Use it when the user asks what an image contains or means. Not for reading text documents or extracting PDF text.

中文：分析附件或阅读器中的图片（截图、图表、照片、示意图），回答图里有什么或表达什么。不适用于 PDF 或文本文档。
negative examples:
- (none curated)
params:
- page (integer, optional): 1-based PDF page to render and read; omitted -> the page open in the viewer (a multi-page PDF with no page in view is refused)
- region (object, optional): optional sub-region of the rendered page as fractions {x, y, w, h} (0..1, top-left origin)
- asset_id (string, required, max_len=64): asset id of the attached/opened image (from turn facts, not the sentence)
- question (string, optional): optional question about the image, copied from the sentence
```

##### `cap-social-search` (243 tokens; parameter block 111)

```
### cap-social-search
tool: search_social
does: Search social platforms such as Reddit for discussions, recommendations, and user opinions. Use it when the user asks what people or online communities are saying on a social platform. It does not apply to general public-web search or searches over the user's own materials.

中文：在 Reddit 等社交平台上搜索大家的讨论、推荐与评价。当用户点名社交平台，或要“社区／论坛里大家怎么说、网友评价”时适用。一般网页搜索、搜索用户自己的资料，都不属于它。
negative examples:
- (none curated)
params:
- limit (integer, optional): optional max results per platform; only when the sentence states one
- query (string, required): the user's search TOPIC, extracted from the sentence and kept VERBATIM — do not rewrite, translate, summarize, expand or paraphrase
- platform (string, optional): optional platform (reddit / x / zhihu); only when the sentence names one; OMIT when no specific platform is named
- subreddit (string, optional): optional reddit-only scope; only when the sentence names one
```

##### `cap-read-document` (224 tokens)

```
### cap-read-document
tool: read_document
does: Read and understand the structured content of a resolved document asset. Use it when the user asks to read a document and answer questions based on its contents. It does not apply to exporting the document's raw text, extracting only tables, or reading a workspace-local file by path.

中文：阅读当前附件或阅读器中已打开的那份文档的结构化内容，可用于回答关于该文档的问题。当用户要“读这份文档／通读它并据此作答”时适用。把文档原文整段导出、或只转表格，都不属于它；按路径读工作区里的普通文件也不属于它。
negative examples:
- (none curated)
params:
- pages (string, optional): optional page/slide spec like "2" or "1-3"; only when the sentence states one
- asset_id (string, required, max_len=64): asset id of the attached/opened document (from turn facts, not the sentence)
```

##### `cap-rag-search` (216 tokens)

```
### cap-rag-search
tool: rag_search
does: Search the user's own imported materials or document library. Use it when the user asks to find information in their imported or uploaded materials. It does not apply to public-web search or social-platform search.

中文：在用户自己导入的材料（个人文档库、已上传的资料）里检索内容。当用户要在“我导入的资料／我的资料库”中查找某个问题时适用。搜索公开网络、或搜索社交平台，都不属于它。
negative examples:
- (none curated)
params:
- query (string, required): the user's search TOPIC, extracted from the sentence and kept VERBATIM — do not rewrite, translate, summarize, expand or paraphrase
- top_k (integer, optional): optional result count; only when the sentence states one
- domain (string, optional): optional domain NAME to scope the search to (resolved to a real domain id by the tool); only when the sentence names one
```

##### `cap-add-term` (204 tokens)

```
### cap-add-term
tool: add_term
does: Add a word, term, or expression to a named vocabulary domain. Use it when the user asks to add or save a term to their vocabulary, glossary, or vocabulary list. It does not apply to creating files or folders or editing existing file content.

中文：把一个词、术语或表达收进用户词汇库（生词本、术语表、词库里的某个命名域）。当用户要求将某个词、术语或表达加入/收录到词汇库的指定命名域时适用。它只新增一条词汇记录；新建文件或文件夹、修改已有文件的内容，都不属于它。
negative examples:
- (none curated)
params:
- term (string, required, max_len=120): the word/term to add, quotes removed
- domain (string, required, max_len=60): the vocabulary domain to add it into
```

##### `cap-pdf-extract-text` (195 tokens)

```
### cap-pdf-extract-text
tool: pdf_extract_text
does: Extract the complete raw text from an attached or resolved document. Use it when the user asks to extract or export the document's full text. It does not apply to extracting only tables or to reading and understanding the document for question answering.

中文：提取附件/已解析文档的完整原始文字内容。当用户要求提取全文或导出文档原始文字时适用。只转换表格的应交给表格转换；要求理解、阅读并据此回答问题的，不属于它。
negative examples:
- (none curated)
params:
- pages (string, optional): optional page spec like "2" or "1-3" from the viewer context; omitted = whole document
- asset_id (string, required, max_len=64): asset id of the attached/opened document (from turn facts, not the sentence)
```

##### `cap-web-search` (193 tokens)

```
### cap-web-search
tool: web_search
does: Search the public web for external information, including webpages, reference material, news, and current information. Use it when the user asks to search or look up information on the internet. It does not apply to searching the user's own materials or searches restricted to a specific social platform.

中文：在公开互联网上搜索外部信息，包括网页、资料、新闻和时事。当用户要求上网查找公开网络信息时适用。搜索用户自己的资料或限定在特定社交平台上搜索，不属于它。
negative examples:
- (none curated)
params:
- query (string, required): the user's search TOPIC, extracted from the sentence and kept VERBATIM — do not rewrite, translate, summarize, expand or paraphrase
- top_k (integer, optional): optional number of results; only when the sentence states one
```

##### `cap-pdf-table-to-text` (191 tokens)

```
### cap-pdf-table-to-text
tool: pdf_table_to_text
does: Convert tables inside an attached PDF into text while preserving their row and column structure. Use it when the user explicitly asks to convert PDF tables to text or CSV. It does not apply to extracting the full document text.

中文：只把附件 PDF 里的表格转换为文字，并保留行列结构。当用户明确要求“把这份 PDF 里的表格转成文字／CSV”时适用。提取整份文档的全文不属于它。
negative examples:
- (none curated)
params:
- pages (string, optional): optional page spec like "2" or "1-3" from the viewer context; omitted = whole document
- asset_id (string, required, max_len=64): asset id of the attached/opened document (from turn facts, not the sentence)
```

##### `cap-edit-file` (200 tokens)

```
### cap-edit-file
tool: edit_file
does: Modify the contents of an existing workspace-local file. Use it when the user asks to change, replace, update, or write content in an existing file. It does not apply to only reading a file, creating a folder, or adding a term to a vocabulary domain.

中文：修改工作区里某个已存在文件的内容——替换其中一段文字，或写入新内容。当用户要求“改动／更新／替换某个文件里的内容”时适用。只读取文件、新建文件夹、把词存进词汇库，都不属于它。
negative examples:
- (none curated)
params:
- path (string, required): Workspace-relative file path, or a drive path "My Drive/<path>".
- new_text (string, optional): Replacement / new file content.
- old_text (string, optional): Exact text to replace (optional).
```

##### `cap-translate` (164 tokens)

```
### cap-translate
tool: translate
does: Translate provided text from one language into another. Use it when the user explicitly asks to translate text into a different language. It does not apply to summarizing, explaining, or rewriting text in the same language.

中文：把给定文本从一种语言翻译成另一种语言。当用户要求“翻译成某种语言”时适用，附件或引用中的文字同样可以翻译。要求总结、解释或同语言改写的，都不属于它。
negative examples:
- (none curated)
params:
- text (string, required): the text to translate, copied from the sentence
- target_language (string, optional): optional target language name (e.g. "English", "Chinese"); omitted -> executor default (English)
```

##### `cap-create-folder` (148 tokens)

```
### cap-create-folder
tool: create_folder
does: Create a new folder with an explicit name in the workspace. Use it when the user asks to create or make a folder or directory. It does not apply to creating or editing files, renaming existing folders, or adding terms to a vocabulary domain.

中文：新建一个名称明确的文件夹（目录、子文件夹）。当用户要求“新建／创建文件夹”时适用。新建文档或文件、给已有文件夹改名、把词收进词汇库，都不属于它。
negative examples:
- (none curated)
params:
- name (string, required, max_len=120): the folder name, quotes removed
```

##### `cap-slides` (148 tokens)

```
### cap-slides
tool: slides_gen
does: Create a slide deck with a canonical 16:9 PDF plus Marp and .pptx exports. Use it when the user asks for slides, a presentation, or a PowerPoint deck. It does not apply to summaries, mind maps, or published research PDF artifacts.

中文：把材料做成幻灯片：输出正式的 16:9 PDF，并同时导出 Marp 与 .pptx。当用户要“PPT／演示文稿／slides”时适用。要摘要、思维导图或研究 PDF 成果物的，都不属于它。
negative examples:
- (none curated)
params: none (arguments must be {})
```

##### `cap-bash` (149 tokens)

```
### cap-bash
tool: bash
does: Execute a command or script in the workspace sandbox. Use it when the user explicitly asks to run or execute a command or script. It does not apply when the user only wants to read a file or edit file contents.

中文：在工作区沙箱中执行用户明确要求运行的命令或脚本。当用户要求“运行／执行某条命令或脚本”时适用。只想读取文件、或只想修改文件内容而没有要求执行命令的，都不属于它。
negative examples:
- (none curated)
params:
- command (string, required): The shell command to run.
- timeout (integer, optional): Timeout in seconds.
```

##### `cap-mindmap` (147 tokens)

```
### cap-mindmap
tool: mindmap_gen
does: Create a Mermaid mind map (.mmd) from one or more workspace files. Use it when the user asks for a mind map, brain map, or knowledge tree. It does not apply to producing a written summary, slide deck, or published research PDF.

中文：把一个或多个工作区文件的内容整理成分支结构的 Mermaid 思维导图（.mmd）。当用户要“思维导图／脑图／知识树”时适用。要文字摘要、幻灯片或正式 PDF 成果物的，都不属于它。
negative examples:
- (none curated)
params: none (arguments must be {})
```

##### `cap-artifact` (190 tokens)

```
### cap-artifact
tool: artifact
does: Compile a research artifact into a formally published PDF. Use it when the user explicitly asks to compile, generate, or publish research results as a final PDF artifact. It does not apply to ordinary slide decks, summaries, or mind maps.

中文：将研究成果物编译为正式发布的 PDF。当用户明确要求将研究成果编译、生成或发布为正式 PDF 成果物时适用。普通幻灯片、摘要、思维导图不属于它。
negative examples:
- (none curated)
params:
- action (string, required): Which action to run. One of: compile_pdf, status.
- run_id (string, optional): status: the compile run id; compile_pdf: optional replay id override.
- project_id (string, optional): compile_pdf: research project whose primary manuscript is compiled.
```

##### `cap-summary` (141 tokens)

```
### cap-summary
tool: summary_gen
does: Generate a grounded Markdown research summary with [file:line] citations. Use it when the user asks to summarize, condense, or extract key points from the available research materials. It does not apply to creating slides, mind maps, translations, or published research PDF artifacts.

中文：把本轮材料提炼成带有出处的 Markdown 摘要。当用户要“总结／概括要点／提炼重点”时适用。要幻灯片、思维导图、翻译或研究 PDF 成果物的，都不属于它。
negative examples:
- (none curated)
params: none (arguments must be {})
```

##### `cap-read-file` (99 tokens)

```
### cap-read-file
tool: read_file
does: Read the contents of an existing file from the workspace or My Drive using its path. It does not apply to attachments or viewer documents, or to image analysis.
negative examples:
- (none curated)
params:
- path (string, required): Workspace-relative file path, or a drive path "My Drive/<path>".
- max_chars (integer, optional): Optional read cap.
```

##### `cap-research` (40 tokens)

```
### cap-research
tool: research
does: Multi-step Research OS project driver (plugin).
negative examples:
- (none curated)
params: none (arguments must be {})
```

#### Validation and Follow-up

Verified in this audit:

- All 18 cards were rendered with the production `render_card()` from the live Registry
  (`capabilities`), read through the same loader the serving path uses
  (`registry.store.list_capabilities`).
- Token counts use the project's actual Laya multilingual tokenizer at the pinned base
  revision, with the same call shape the sequence builder uses.
- All 18 cards fit within the 256-token cap. `cap-social-search` is the longest
  (243 raw / 250 total), followed by `cap-read-document` (224 / 231) and `cap-vision`
  (224 / 229); the remaining 15 are ≤216 raw tokens.
- The card texts above were each rendered with the production `render_card()` and tokenized
  in full.
- The card texts preserve every slot name, type, requiredness, `max_len` and extraction
  constraint.

Pending or unverified:

- No capability definition, migration, training row or model artifact was modified.

`head_max_len` is a **combined budget**, not a length. At four option slots the
construction audit confirmed the 256/768/1024 budget is sufficient **for the frozen v2
bundle** — no option card is truncated and the `REJECT` card enters the sequence in full.
The current-Registry measurements above also fit within the per-card cap.

**Deployment consequence:** a stock `laya` install renders at 48 tokens per option and
`head_max_len=256`. Reproducing v2's behaviour requires applying the same override before
the Agent is constructed (`R.install()` in `layachoice_v2_render.py`) and passing the same
256/768/1024 budget; under stock rendering expect a materially lower score.

## 10. Fine-Tuning Hyperparameters

| | |
|---|---|
| Epochs | **10** |
| Micro batch / grad accum | 8 / 4 (effective batch 32) |
| Group size (policy-sampling term) | 4 |
| `lr_encoder` / `lr_head` | 2.5e-5 / 1.0e-4 |
| Weight decay | 0.01 |
| σ schedule | 0.4 → 0.1 across epochs |
| Loss | `loss_rl + 1.0 * loss_ce`; `w_sph=0.75`, `w_rps=1.0` |
| Gradient clip | `clip_grad_norm_(1.0)` |
| Total optimizer updates | 600 |
| Precision | bf16 autocast (no `GradScaler`) |
| Hardware | 1× NVIDIA L4 |
| Runtime | torch `2.14.1+cu130`, transformers `5.17.0`, python `3.11`, `laya==0.3.21` |
| Seed base | `42` |
| Fine-tuning | **full-parameter** (no PEFT/LoRA) |

## 11. Epoch 1–10 Training Results

From the per-epoch validation records. Validation v3 (300 rows, `B_noprov`); top-1 is
invariant under the scalar calibration temperature. Validation `REJECT` counts are small
(16 gold `REJECT` rows), so per-epoch `REJECT` recall/FPR are noisy.

| Epoch | Train loss | Val top-1 | Val ECE | Val REJECT recall | Val REJECT FPR | Val EN | Val ZH |
|---|---|---|---|---|---|---|---|
| 1 | 0.641706 | 85.67% | 0.080275 | 81.25% | 8.45% | 91.78% | 79.87% |
| 2 | 0.384151 | 90.00% | 0.036385 | 93.75% | 8.10% | 93.15% | 87.01% |
| 3 | 0.256366 | 94.00% | 0.056254 | 93.75% | 2.46% | 96.58% | 91.56% |
| 4 | 0.132143 | 89.67% | 0.101781 | 100.00% | 8.10% | 89.73% | 89.61% |
| 5 | 0.079365 | 94.67% | 0.054404 | 100.00% | 2.82% | 97.26% | 92.21% |
| 6 | 0.039875 | 94.33% | 0.056691 | 100.00% | 2.82% | 97.26% | 91.56% |
| **7** | **0.013119** | **96.67%** | **0.030319** | **100.00%** | **1.76%** | **100.00%** | **93.51%** |
| 8 | 0.000358 | 96.33% | 0.035276 | 100.00% | 2.11% | 98.63% | 94.16% |
| 9 | 0.000000 | 96.33% | 0.035276 | 100.00% | 2.11% | 98.63% | 94.16% |
| 10 | 0.000000 | 96.33% | 0.035276 | 100.00% | 2.11% | 98.63% | 94.16% |

- Train loss collapses from 0.642 to 0; validation peaks at epoch 7.
- Epochs 9–10 equal epoch 8 because the train loss had reached 0 and the weights stopped
  moving.
- ZH validation sits above EN in the later epochs, the same direction as the final test.

## 12. Validation Selection

The selection rule is fixed and deterministic, and was frozen before the run:

```
best_epoch = argmax(validation_top1); ties resolved to the EARLIER epoch
train loss is recorded and NEVER used to select
```

**Best epoch = 7, Validation v3 top-1 = 0.966667 = 290/300.** No tie-break was needed
(epochs 8–10 tie at 0.963333, below epoch 7).

Top-1 is invariant under the scalar temperature, so the selection does not depend on the
calibration step that follows it. **The Final Test never participated in selection or in
the export gate.**

## 13. Temperature Calibration

Temperature scaling is applied to the **logits**, not to probabilities. A single scalar is
fitted for the `choice` slot only and written into the length-3 temperature vector:

| | |
|---|---|
| Calibration rows | 211 (seed 20260922, the same rows removed from training) |
| Fitted on | epoch-7 (best checkpoint) |
| Raw fitted value | `6.389379501342773` |
| Clamp range | `[0.5, 5.0]` (the Agent's own `clamp_temperature`) |
| Final written value | `5.0` — **clamped** |
| Exported vector | `[5.0, 1.0, 1.0]` |
| Fallback value | `1.2` (not used) |

The clamp range is deliberately the Agent's own `[0.5, 5.0]`, so the fitted value, the
exported config and the Agent's effective value cannot silently disagree. The raw fitted
value exceeded the cap and was clamped to 5.0.

Calibration is a confidence-sharpening step only; it does not change top-1 (§12). It was
fitted on in-distribution held-out rows only and does **not** guarantee reliable confidence
under distribution shift (§16, §19).

## 14. Final Test v3 Results

Best epoch 7, calibrated temperature `[5.0, 1.0, 1.0]`, 1 400 rows:

| Metric | Value |
|---|---|
| **Top-1** | **0.877857** |
| Error rate (= 1 − top-1) | 0.122143 |
| EN (692 rows) | 0.861272 |
| ZH (708 rows) | 0.894068 |
| Capability targets (1 286 rows) | 0.891913 |
| REJECT targets (114 rows) | 0.719298 |
| p1 partition (700 rows) | 0.910000 |
| p2 partition (700 rows) | 0.845714 |
| ECE | 0.097189 |
| Mean confidence | 0.975046 |
| Mean gold probability | 0.871354 |

REJECT detection (positive class = `REJECT`):

| | Value |
|---|---|
| Gold `REJECT` n | 114 |
| Predicted `REJECT` n | 147 |
| TP / FP / FN / TN | 82 / 65 / 32 / 1221 |
| REJECT recall | 0.719298 |
| REJECT precision | 0.557823 |
| REJECT false-positive rate | 0.050544 |

**Scope statement — read this before quoting 87.79 %.** 0.877857 is **benchmark accuracy on
the frozen Final Test v3 split**. It is **not** production-traffic accuracy and **not**
open-world generalization accuracy. The split is a fixed, curated, 1 400-row artifact over
17 known capabilities with a known hard-negative construction and K=3 + `REJECT`; it says
nothing about capabilities introduced after the freeze or about distribution shift.

## 15. Error / Confusion Summary

Top confusion pairs on the Final Test (gold → predicted, count):

| Gold → predicted | n |
|---|---|
| cap-edit-file → REJECT | 12 |
| cap-mindmap → cap-vision | 12 |
| cap-vision → REJECT | 10 |
| cap-pdf-extract-text → REJECT | 9 |
| cap-bash → REJECT | 8 |
| REJECT → cap-edit-file | 7 |
| REJECT → cap-read-file | 6 |
| cap-social-search → cap-web-search | 6 |
| cap-mindmap → REJECT | 6 |
| cap-bash → cap-read-file | 5 |

The residual error splits into two families. First, the **REJECT boundary is the weakest
part**: `REJECT` recall is 71.93 % and precision 55.78 % — 65 capability rows are wrongly
rejected and 32 `REJECT` rows are wrongly routed to a capability. Second, the
**search cluster** (`web-search` / `social-search` / `rag-search`) and `bash`→`edit-file`
remain, exactly as in v1 — those are the pairs whose `B_noprov` cards are most similar by
construction.

## 16. Export and Reload Validation

`--export-dtype preserve` writes the trained **FP32** weights with no conversion.

| Check | Result |
|---|---|
| Export dir | `V2/out/epoch-7/` → published as the model repository root |
| Export dtype | `preserve` (FP32) |
| Size | 1 287 653 720 B |
| `model.safetensors` sha256 | `c6331bd5b6c044a8e4841bee8e9e69962cc2deaf1fd39bffb178703c07168b6f` |
| Export == best checkpoint | **true** — the export is the epoch-7 weights byte-for-byte |
| `laya.Agent` reload | passed, `load_state_dict(..., strict=True)` |
| Reload smoke | **11 / 12 hits** on 12 held-out rows |
| Temperature after reload | `[5.0, 1.0, 1.0]` — fitted == config == effective |
| Exported `max_len` / `head_max_len` / `option_max_tokens` | 1024 / 768 / 256 |

The reload is run through the public entry point (`laya.Agent(dir, device="cpu")`), not by
re-reading the files, so it also proves the published directory is Agent-loadable.

## 17. Artifact Sizes and SHA256

Published per-epoch checkpoint weights (`eric-ml-nlp/Delveta-LayaChoice-v2-checkpoints`):

| Artifact | Size | SHA256 |
|---|---|---|
| `epoch-1/model.safetensors` | 1 287 653 720 B | `30f28e23cf8b3e0fa76fd7b1f74337b49d69b6eec3932e21ebc221314be8203f` |
| `epoch-2/model.safetensors` | 1 287 653 720 B | `ea0f866aed715141733b25b00a643f46392b3a61230ee55e399ce1688866c2d2` |
| `epoch-3/model.safetensors` | 1 287 653 720 B | `27dd2756ec2a7547555fcae2b2acf6d37cc10d456a132fc6754357f1487fcb5b` |
| `epoch-4/model.safetensors` | 1 287 653 720 B | `e04b5375cf46273744d7a6e47698a258278735d74c0061eed227921fc80bc56f` |
| `epoch-5/model.safetensors` | 1 287 653 720 B | `4572dd2d3986b58476685bdc9151458a308d15808904a54d9b09b39120d832dc` |
| `epoch-6/model.safetensors` | 1 287 653 720 B | `a2f1046c5f8068ac13afe86516bc5ece7961eea410adabf590c1ca829218a207` |
| `model.safetensors` (epoch 7, selected) | 1 287 653 720 B | `c6331bd5b6c044a8e4841bee8e9e69962cc2deaf1fd39bffb178703c07168b6f` |

Epochs 8–10 weight files are **not** published (history only; they tie epoch 8 at
0.963333, below epoch 7). Latency/throughput numbers are deliberately omitted: no
performance baseline was established for this model, and inventing one after the fact
would be measuring the wrong thing (see architecture rule 17).

### Published artifacts

| Artifact | Location |
|---|---|
| Selected model (epoch 7), FP32 weights + model card | [Delveta-LayaChoice-v2](https://huggingface.co/eric-ml-nlp/Delveta-LayaChoice-v2) (public) |
| Non-selected checkpoints (epochs 1–6) | [Delveta-LayaChoice-v2-checkpoints](https://huggingface.co/eric-ml-nlp/Delveta-LayaChoice-v2-checkpoints) (public) |
| Frozen dataset (splits + `manifest.json` + audit) | [Delveta-LayaChoice-v2-Data](https://huggingface.co/datasets/eric-ml-nlp/Delveta-LayaChoice-v2-Data) (public) |
| Training / evaluation code | [`fine_tuning/laya_finetune/V2/`](https://github.com/Eric-LLMs/Delveta/tree/main/fine_tuning/laya_finetune/V2) |

## 18. Reproducibility

| | |
|---|---|
| GPU | 1× NVIDIA L4 |
| torch | 2.14.1+cu130 |
| transformers | 5.17.0 |
| Python | 3.11 |
| laya | 0.3.21 |
| Base revision | `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` (subfolder `multilingual`) |
| Seed base | 42; calibration split seed 20260922 |

What must hold for a rerun to be comparable:

1. `laya==0.3.21` and the pinned base revision (asserted on load).
2. The frozen bundle unchanged — the six SHA256s in §7.
3. The same three budget constants, and the same 256-token builder override (§9), on
   training, validation, final test, export and reload.
4. The same selection rule (§12) and the same calibration ids (§13).

## 19. Known Limitations

1. **The 87.79 % is benchmark accuracy on a frozen split**, not production accuracy and not
   open-world generalization (§14).
2. **The `REJECT` class is the weakest part.** On the final test, `REJECT` recall is
   71.93 % / precision 55.78 %; 65 capability rows are wrongly rejected and 32 `REJECT`
   rows are wrongly routed. `REJECT` recall on the small validation split (16 gold rejects)
   is not a reliable estimate.
3. **K is not validated outside the formal run's 3 + `REJECT`.** The formal run trained and
   measured only K=3. The current V4 bundle is dynamic K+1 but has not been trained; K=1..2
   and K≥4 post-truncation are unmeasured, and no number here covers them.
4. **17 capabilities in the formal run.** The current Registry has 18 active capability
   cards; capabilities added after the freeze are out of distribution by construction.
5. **The released weights were trained on the flat 4-way build.** Serving the current dynamic
   bundle requires the dynamic adapter; the measured numbers here do not cover that bundle.
6. **The dataset is synthetic / curated**, not sampled from live traffic (§5).
7. **EN/ZH only.** No claim is made for any other language.
8. **Base-model license is undetermined.** The upstream repository went private and its
   snapshot ships no license; no license is asserted over the derived weights.

## 20. Final Conclusion

The formal result, on the frozen Final Test v3 (1 400 rows, K=3 + `REJECT`):

| Arm | Top-1 |
|---|---|
| **256/768 fine-tuned, best epoch 7, calibrated** | **0.877857** |
| capability-only subset (1 286 rows) | 0.891913 |
| `REJECT` subset (114 rows) | 0.719298 |

- Validation v3 selected epoch 7 at **0.966667**; the Final Test never participated in
  selection or in the export gate.
- The exported artifact is byte-identical to the selected checkpoint (`c6331bd5…`), passes
  `laya.Agent` `strict=True` reload, and reproduces the fitted temperature exactly.
- v2's job is the one v1 could not do: **abstain when the right capability is not among the
  candidates.** It does so at a **`REJECT` recall of 71.93 %** — useful but the weakest part
  of the model, and the first target of any v3.
- The two splits are **not directly comparable**: v1's 93.56 % was a 3-way 900-row split with
  no `REJECT`; v2's 87.79 % is a harder 4-way split that adds the abstain class.

Scope, restated once more because it is the easiest thing to over-read: **0.877857 is
benchmark accuracy on a frozen, curated, K=3 + `REJECT`, 17-capability, EN/ZH split.** It is
not a production-traffic number and not an open-world claim.
