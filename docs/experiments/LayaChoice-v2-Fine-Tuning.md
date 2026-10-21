# LayaChoice-v2 — Fine-Tuning Laya with an Explicit REJECT Option

> **Status: released.** The formal run `layachoice_v2_082_r1` is complete and its artifacts
> are closed. This document is the record of that run; the stable architectural facts
> distilled from it live in
> [architecture.md §26](../architecture.md#26-layachoice-capability-selection).
>
> **Production integration status.** v2 is wired into production: the `cap_router` card
> renderer emits the `REJECT` criterion (§8), the `deploy/laya` sidecar loads the v2
> checkpoint, and the service backend maps a `REJECT` answer to a normal fourth decision
> routed to the Agent. It runs as the `cap-router` Compose service.
>
> Everything below is derived from the run artifacts (`fine_tuning/laya_finetune/V2/` and the
> frozen bundle `recall_threshold_082` in `V2/data/`), not from recollection.

> **Contract — V1 is fixed, V2 is dynamic.** V1 is the fixed-K build (**K = 3**). V2 is
> **dynamic K+1**: K capability cards (0..3) plus one `REJECT` card, so a question carries
> **1..4 options**. Run `layachoice_v2_082_r1`, recorded here, was trained and evaluated on
> that dynamic contract, on bundle `recall_threshold_082`. It supersedes the earlier flat
> 4-way epoch-7 run; that run's numbers are not part of this document.

## 1. Experiment Overview

| | |
|---|---|
| Task | Choose the correct capability for a user request from the recalled candidates, **or** reject the set. Dynamic K+1 (K = 0..3, 1..4 options) |
| Base model | `convaiinnovations/laya`, subfolder `multilingual`, revision `55cf4c4eb…` |
| Output | `Delveta LayaChoice v2` — full-parameter fine-tune, FP32 export + FP16 storage copy |
| Data | Train 36 197 / Calibration 400 / Validation 2 407 / Final Test 9 729 (bundle `recall_threshold_082`, all frozen) |
| Input config | `OPTION_MAX_TOKENS=256`, `head_max_len=768`, `max_len=1024` |
| Best epoch | **9** (selected by validation top-1 only) |
| Validation (best) | **0.996676** = 2 399/2 407 |
| Final Test | **0.998355** = 9 713/9 729 — en 0.998197 / zh 0.998306 / mixed 0.999137 |
| REJECT on Final Test | recall 0.996441 / precision 0.995261 / FPR 0.000450 |
| Zero-shot baseline | 0.416795 (fine-tuned delta **+0.581560**) |
| Export | `model.safetensors`, FP32, 1.29 GB, sha256 `964f8fe5…` |

One sentence: **v1 is the fixed 3-way build; v2 adds an explicit `REJECT` option under a
dynamic K+1 contract (1..4 options)**. The released run was trained on 36 197 rows and
reaches 0.998355 on the frozen 9 729-row test set, with the best checkpoint chosen on a
separate validation split the test set never touched.

## 2. Objective

v1 answered "which of these 3 capabilities handles the request?". That framing has no way
to say *"none of them"* — a turn whose true capability was not retrieved is forced onto a
wrong candidate. The objective of v2 was to close that gap with a number:

> Can the same base decision model, fine-tuned with a dedicated `REJECT` option in a
> dynamic K+1 candidate set, learn to abstain when the correct capability is not among the
> candidates — without giving up capability-selection accuracy?

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
  `reject-shuffle-v1`); only the final order is stored, in `input.options`, so the model
  cannot learn a positional prior.
- The gold slot is the `label` index within the row's actual option count; the adapter
  derives `target_kind ∈ {capability, reject}` from the option at that index.

**This run.** `layachoice_v2_082_r1` was trained and measured on the dynamic build: every
row carries K capability cards (K = 0..3) plus one `REJECT` card, and the gold slot is the
position of the correct option within that row's 1..4 options. The earlier flat 4-way build
(K = 3 + `REJECT` only) is not part of this document.

**A caller must render the `REJECT` card.** A question without it does not match the
trained pipeline and its scores are not meaningful.

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
The released run uses the V4 dynamic bundle `recall_threshold_082`, installed at
`fine_tuning/laya_finetune/V2/data/`. It is derived from the frozen v1 bundle (`B_noprov`
view), not rebuilt from the Registry:

| | |
|---|---|
| Card view | `B_noprov` — the production card with the `query examples:` block and the `evidence:` provenance line removed |
| Candidates per row | **K capability cards (0..3) + 1 `REJECT` card, 1..4 options** |
| Capabilities | 18 active capability cards (see §9), plus `REJECT` as the abstain target |
| Languages | `zh` / `en` / `mixed` |
| Row shape | `{message_id, language, input{query, options[K+1]}, label}`; derived views are computed by the adapter and never stored |

The `REJECT` card text is frozen for all rows:

> `### REJECT` / `tool: none` / `does: No listed capability correctly handles the user's
> request. Choose this option only when none of the other options is the right
> capability.` plus a Chinese line.

`REJECT` rows are of two kinds: *natural* (the source query genuinely had no matching
capability) and *synthetic* (a capability row whose gold card was swapped for `REJECT`).
Both are training/exam items; the frozen bundle records which is which, and that metadata
is **never rendered** — only the query and the option cards enter the sequence.

v1's label-leakage argument still holds unchanged: the stored `evidence:` line names the
answer, so it stays dropped. The `B_noprov` view is what makes the task a decision rather
than a lookup.

## 6. Splits and Calibration

```
source_train_rows           36 197  ── exclude the frozen 400 calibration rows (seed 20260922)
calibration_rows               400  ──► indices and ids recorded in manifest.json
effective_train_rows        35 797  ── after exclusion (world-size aligned)
Validation                   2 407  ── per-epoch top-1; the ONLY selection signal
Final Test                   9 729  ── benchmark of best_epoch only, never used to pick or gate
```

- **Language** — train `en` 16 499 / `zh` 15 505 / `mixed` 4 193; val `en` 1 116 /
  `zh` 1 030 / `mixed` 261; test `en` 4 438 / `zh` 4 132 / `mixed` 1 159.
- **Option counts** — the manifest declares `K+1` with `min = 1`, `max = 4`, and the
  observed values are 1, 2, 3 and 4. The bundle carries 48 333 rows per recall threshold
  (train 36 197 + val 2 407 + test 9 729); only threshold `0.82` was trained.
- **Calibration rule** — `min(400, n_train // 10)` over the FINAL (post-shuffle) train
  order, seed `20260922`, 400 indices recorded; excluded from training by construction.
- **Disjointness** — train / validation / test are disjoint by `id` and
  `source_query_id`, and calibration rows are excluded before any world-size truncation,
  so no rank can pull a calibration row into training.

The earlier flat-bundle Test→Train adjustment is not part of this bundle; the V4 splits
were assigned during construction and recorded in `manifest.json`.

## 7. Frozen Dataset SHA256

Verified unchanged at the end of the run (`V2/data/SHA256SUMS`), and still identical in
the published dataset repository:

| Artifact | SHA256 |
|---|---|
| `train.jsonl` | `97ed1b54a205d3fac4dd3837e93abb779104568b94019c9a161752c7ca670c77` |
| `val.jsonl` | `833829ce30d929a8ce502036e3f7b049ef8279770558a632efaaf6bd9e54cb03` |
| `test.jsonl` | `66c6318d57d2021a85384020f8fb52f5903f37353ec7593e3e6b07c6215dbac1` |
| `manifest.json` (installed run bundle) | `e103b2f019454e07188f60257c72ec73ecf745fda54d830a1585595309f9f85e` |
| `manifest.json` (published dataset repo) | `92514b46c1e090d50bcae9464112aa80ad8ba48f4e929f098afb929d7488adf8` |

The two `manifest.json` hashes differ because the installed copy and the published
repository package the same records differently; the three split files are byte-identical in
both. The published dataset is
[`Delveta-LayaChoice-v2-Data`](https://huggingface.co/datasets/eric-ml-nlp/Delveta-LayaChoice-v2-Data).

## 8. Candidate Contract

V1 is fixed at **K = 3**. V2's current contract is **dynamic K+1**: the model receives K
capability cards plus one `REJECT` card, and emits a choice over **K + 1 options (1..4)**.

| Candidate count | Behaviour |
|---|---|
| K = 0 | 1 option (`REJECT` only); the bundle contains such rows |
| K = 1 | 2 options (1 capability + `REJECT`) |
| K = 2 | 3 options (2 capabilities + `REJECT`) |
| K = 3 | 4 options (3 capabilities + `REJECT`) |
| K ≥ 4 | the business layer truncates to top-3 **first** (`normalize_top_k`), then LayaChoice |

**Production candidate normalization must happen before LayaChoice**, not inside it. The
released run was trained and measured across all four observed option counts (1..4), and
the test metrics in §14 aggregate them. K≥4 post-truncation is not represented in the
bundle. In every case, a caller must render `REJECT` as one of the criteria.

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
| `cap-social-search` | 216 | 7 | 223 | 256 |
| `cap-read-document` | 221 | 7 | 228 | 256 |
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
truncation. `cap-vision` is the longest (224 raw / 229 total), followed by
`cap-read-document` (221 / 228) and `cap-social-search` (216 / 223). The remaining cards
are at or below 216 raw tokens.

This audit covers the 18 capability cards used by the released run. All 18 fit within the
256-token budget, so no option card is truncated in the training sequence.

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

##### `cap-social-search` (216 tokens; parameter block 84)

```
### cap-social-search
tool: search_social
does: Search social platforms such as Reddit for discussions, recommendations, and user opinions. Use it when the user asks what people or online communities are saying on a social platform. It does not apply to general public-web search or searches over the user's own materials.

中文：在 Reddit 等社交平台上搜索大家的讨论、推荐与评价。当用户点名社交平台，或要“社区／论坛里大家怎么说、网友评价”时适用。一般网页搜索、搜索用户自己的资料，都不属于它。
negative examples:
- (none curated)
params:
- limit (integer, optional): max results per platform; only if stated
- query (string, required): search query copied verbatim from the user's sentence; do not rewrite, translate, summarize, expand or paraphrase
- platform (string, optional): platform reddit / x / zhihu; omit unless named
- subreddit (string, optional): reddit-only subreddit; omit unless named
```

##### `cap-read-document` (221 tokens)

```
### cap-read-document
tool: read_document
does: Read and understand the structured content of a resolved document asset. Use it when the user asks to read a document and answer questions based on its contents. It does not apply to exporting the document's raw text, extracting only tables, or reading a workspace-local file by path.

中文：阅读当前附件或阅读器中已打开的那份文档的结构化内容，可用于回答关于该文档的问题。当用户要“读这份文档／通读它并据此作答”时适用。把文档原文整段导出、或只转表格，都不属于它；按路径读工作区里的普通文件也不属于它。
negative examples:
- (none curated)
params:
- pages (string, optional): page/slide spec "2" or "1-3"; only if stated
- asset_id (string, required, max_len=64): ID of the attached or opened document, obtained from turn facts, not the user's sentence
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
- All 18 cards fit within the 256-token cap. `cap-vision` is the longest
  (224 raw / 229 total), followed by `cap-read-document` (221 / 228) and
  `cap-social-search` (216 / 223); the remaining 15 are ≤216 raw tokens.
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
| Total optimizer updates | 11 190 (planned == actual) |
| Precision | bf16 autocast (no `GradScaler`) |
| Hardware | 1× NVIDIA L4 24 GB |
| Training time | 17 673 s ≈ 4.91 h |
| Runtime | torch `2.14.1+cu130`, transformers `5.17.0`, python `3.10.20`, `laya==0.3.21` |
| Seed base / calibration seed | `42` / `20260922` |
| Fine-tuning | **full-parameter** (no PEFT/LoRA) |

## 11. Epoch 1–10 Training Results

From the per-epoch validation records. Validation (2 407 rows, `B_noprov`); top-1 is
invariant under the scalar calibration temperature. Validation contains 220 gold `REJECT`
rows.

| Epoch | Mean train loss | Sigma | Seed | Train s | Val top-1 | Val ECE | REJECT recall | REJECT FPR | REJECT precision |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.354976 | 0.4000 | 42 | 1678.69 | 0.990445 | 0.009463 | 0.945455 | 0.003201 | 0.967442 |
| 2 | 0.114831 | 0.3667 | 43 | 1677.86 | 0.981305 | 0.018557 | 0.945455 | 0.013260 | 0.877637 |
| 3 | 0.052656 | 0.3333 | 44 | 1678.71 | 0.989198 | 0.009794 | 0.909091 | 0.000914 | 0.990099 |
| 4 | 0.033588 | 0.3000 | 45 | 1677.71 | 0.989614 | 0.010101 | 0.931818 | 0.002743 | 0.971564 |
| 5 | 0.020002 | 0.2667 | 46 | 1678.19 | 0.972995 | 0.026366 | 0.922727 | 0.019204 | 0.828571 |
| 6 | 0.012646 | 0.2333 | 47 | 1677.47 | 0.994184 | 0.005269 | 0.963636 | 0.001372 | 0.986047 |
| 7 | 0.004379 | 0.2000 | 48 | 1677.15 | 0.995015 | 0.005989 | 0.963636 | 0.000914 | 0.990654 |
| 8 | 0.003872 | 0.1667 | 49 | 1680.51 | 0.995430 | 0.004793 | 0.968182 | 0.000914 | 0.990698 |
| **9** | **0.004162** | **0.1333** | **50** | **1677.18** | **0.996676** | **0.003710** | **0.981818** | **0.000914** | **0.990826** |
| 10 | 0.001076 | 0.1000 | 51 | 1676.60 | 0.995430 | 0.004196 | 0.968182 | 0.000914 | 0.990698 |

- Train loss falls from 0.355 to 0.001; validation top-1 peaks at epoch 9.
- Epoch 10 has the lowest training loss but a lower validation top-1, so it is not selected.

**Final test of the selected epoch (epoch 9).** Measured on `test.jsonl` (9 729 rows)
after temperature calibration; the test file was not used for training, selection or
calibration. Input budget: 256 tokens per option, 768 shared by the question head and all
options, 1 024 for the full sequence (§9).

| Model | Test rows | Top-1 |
|---|---:|---:|
| Fine-tuned epoch 9, calibrated | 9 729 | **0.998355** (9 713 / 9 729) |
| — capability targets | 8 886 | 0.998537 |
| — `REJECT` targets | 843 | 0.996441 |
| Zero-shot base model | 9 729 | 0.416795 |

## 12. Validation Selection

The selection rule is fixed and deterministic, and was frozen before the run:

```
best_epoch = argmax(validation_top1); ties resolved to the EARLIER epoch
train loss is recorded and NEVER used to select
```

**Best epoch = 9, validation top-1 = 0.996676 = 2 399/2 407.** No tie-break was needed.

Top-1 is invariant under the scalar temperature, so the selection does not depend on the
calibration step that follows it. **The Final Test never participated in selection or in
the export gate.**

## 13. Temperature Calibration

Temperature scaling is applied to the **logits**, not to probabilities. A single scalar is
fitted for the `choice` slot only and written into the length-3 temperature vector:

| | |
|---|---|
| Calibration rows | 400 (seed 20260922, the same rows removed from training) |
| Fitted on | epoch-9 (best checkpoint) |
| Raw fitted value | `3.070228338241577` |
| Clamp range | `[0.5, 5.0]` (the Agent's own `clamp_temperature`) |
| Final written value | `3.070228338241577` — no clamping |
| Exported vector | `[3.070228338241577, 1.0, 1.0]` |
| Fallback value | `1.2` (not used) |

The clamp range is deliberately the Agent's own `[0.5, 5.0]`, so the fitted value, the
exported config and the Agent's effective value cannot silently disagree.

Calibration is a confidence-sharpening step only; it does not change top-1 (§12). It was
fitted on in-distribution held-out rows only and does **not** guarantee reliable confidence
under distribution shift (§19).

## 14. Final Test Results

Best epoch 9, calibrated temperature `[3.070228338241577, 1.0, 1.0]`, 9 729 rows:

| Metric | Value |
|---|---|
| **Top-1** | **0.998355** (9 713 / 9 729) |
| Error rate (= 1 − top-1) | 0.001645 |
| ECE | 0.002576 |
| Mean confidence | 0.995827 |
| Mean gold probability | 0.994382 |

By language (test top-1):

| Language | n | Top-1 |
|---|---:|---:|
| en | 4 438 | 0.998197 |
| zh | 4 132 | 0.998306 |
| mixed | 1 159 | 0.999137 |

By target kind (test top-1):

| Target | n | Top-1 |
|---|---:|---:|
| capability | 8 886 | 0.998537 |
| reject | 843 | 0.996441 |

REJECT detection (positive class = `REJECT`):

| | Value |
|---|---|
| Gold `REJECT` n | 843 |
| Predicted `REJECT` n | 844 |
| TP / FP / FN / TN | 840 / 4 / 3 / 8 882 |
| REJECT recall | 0.996441 |
| REJECT precision | 0.995261 |
| REJECT false-positive rate | 0.000450 |

Zero-shot comparison on the same frozen test split (checkpoint temperature, no
calibration):

| Metric | Zero-shot baseline | Fine-tuned epoch 9 |
|---|---:|---:|
| Top-1 | 0.416795 | **0.998355** |
| ECE | 0.206293 | **0.002576** |
| REJECT recall | 0.595492 | **0.996441** |
| REJECT precision | 0.128751 | **0.995261** |
| REJECT FPR | 0.382287 | **0.000450** |

Top-1 improvement over the baseline: **+0.581560**.

**Scope statement — read this before quoting 99.84 %.** 0.998355 is **benchmark accuracy
on the frozen 9 729-row test split**. It is **not** production-traffic accuracy and **not**
open-world generalization accuracy. The split is a fixed, curated artifact over 18 known
capabilities with dynamic K+1; it says nothing about capabilities introduced after the
freeze or about distribution shift.

## 15. Error / Confusion Summary

The 9 729-row test contains **16 top-1 errors** (error rate 0.001645):

| Family | n |
|---|---:|
| Gold capability, predicted `REJECT` (FP) | 4 |
| Gold `REJECT`, predicted a capability (FN) | 3 |
| Gold capability, predicted a different capability | 9 |
| **Total** | **16** |

Of the 8 886 capability rows, 8 873 were correct, 9 were mapped to the wrong capability and
4 were rejected. Of the 843 `REJECT` rows, 840 were correct and 3 were routed to a
capability. The `REJECT` boundary therefore accounts for 7 of the 16 errors; the other 9
are capability-to-capability confusions. The per-row report in the training-evidence
repository records the individual confusion pairs; this document keeps to the aggregate
because the row-level dump is not reproduced here.

## 16. Export and Reload Validation

`--export-dtype preserve` writes the trained **FP32** weights with no conversion.

| Check | Result |
|---|---|
| Export dir | `V2/out/epoch-9/` → published as `FP32/` in the model repository |
| Export dtype | `preserve` (FP32) |
| Size | 1.29 GB (decimal) |
| `model.safetensors` sha256 | `964f8fe5dde2812ad3902d8237035bfb109885d4fefef590307bc681bd8fab14` |
| Export == best checkpoint | **true** — the export is the epoch-9 weights byte-for-byte |
| `laya.Agent` reload | passed, `load_state_dict(..., strict=True)` |
| Temperature after reload | `[3.070228338241577, 1.0, 1.0]` — fitted == config == effective |
| Exported `max_len` / `head_max_len` / `option_max_tokens` | 1024 / 768 / 256 |

The reload is run through the public entry point (`laya.Agent(dir, device="cpu")`), not by
re-reading the files, so it also proves the published directory is Agent-loadable.

## 17. Artifact Sizes and SHA256

Published weights (sizes are decimal):

| Artifact | Repository | Size | SHA256 |
|---|---|---|---|
| `epoch-2/model.safetensors` | checkpoints | 1.29 GB | `514c7e2f1e73bafc4a523e90c99eb14a25045dff9f8b3117c323cb9acb3fb492` |
| `epoch-4/model.safetensors` | checkpoints | 1.29 GB | `a831061b4c3218262c713bbafeb50d13265b53a6e8e906d3fa1d6c6ecd7d3994` |
| `epoch-6/model.safetensors` | checkpoints | 1.29 GB | `31e00d302f5f7590a53f1c379862ace463b89566bff295b6e4b6b5b51476c44a` |
| `epoch-8/model.safetensors` | checkpoints | 1.29 GB | `2d512ec730c6f06279c1ffe877ab5b00a6af495e58a2670ae3d05bc3ebfa59c7` |
| `epoch-10/model.safetensors` | checkpoints | 1.29 GB | `131d557409dceefb546480093c2a0707848997cbe2e998a8afb5f24ec4ca1625` |
| `FP32/model.safetensors` (epoch 9, selected) | model | 1.29 GB | `964f8fe5dde2812ad3902d8237035bfb109885d4fefef590307bc681bd8fab14` |
| `FP16/model.safetensors` (epoch 9, storage copy) | model | 644 MB | `ce550b6a85d0a3a96083f842fa0432c2efa9d19f3acea6f9a093fa7140949f97` |

Epochs 1, 3, 5 and 7 weight files are not published. The five non-deployed checkpoints are
in `Delveta-LayaChoice-v2-checkpoints`; the epoch-9 FP32 and FP16 files are in the model
repository. Latency/throughput numbers are deliberately omitted: no performance baseline was
established for this model, and inventing one after the fact would be measuring the wrong
thing (see architecture rule 17).

### Published artifacts

| Artifact | Location |
|---|---|
| Selected model (epoch 9), FP32 + FP16 directories + model card | [Delveta-LayaChoice-v2](https://huggingface.co/eric-ml-nlp/Delveta-LayaChoice-v2) (public) |
| Non-deployed checkpoints (epochs 2, 4, 6, 8, 10) | [Delveta-LayaChoice-v2-checkpoints](https://huggingface.co/eric-ml-nlp/Delveta-LayaChoice-v2-checkpoints) (public) |
| Frozen dataset (splits + `manifest.json` + audit) | [Delveta-LayaChoice-v2-Data](https://huggingface.co/datasets/eric-ml-nlp/Delveta-LayaChoice-v2-Data) (public) |
| Training / evaluation code | [`fine_tuning/laya_finetune/V2/`](https://github.com/Eric-LLMs/Delveta/tree/main/fine_tuning/laya_finetune/V2) |

### 17.1 Precision directories: FP32 vs FP16

> **Scope.** The released model repository ships two self-contained precision directories,
> `FP32/` and `FP16/`, carrying the same epoch-9 weights from the run recorded in this
> document. They contain the same parameters; §17.1 measures what changes at runtime and
> what does not.

#### What was measured

The two directories were compared through the production chat path: the real Electron
client, the full `apps.api.main:app` backend, the Intent Funnel and the cap_router service
running `laya 0.3.21` on CPU. Both arms used the same code (`857d482f`), backend process,
database, embedding service, recall configuration, threshold and deterministic 100-row
subset (50 `zh` + 50 `en`) of the frozen 500-row V2 set. The only variable was the mounted
model directory. LLM argument extraction and Agent fallback were disabled, so the scores
measure capability selection only.

| Metric | FP32 | FP16 |
|---|---:|---:|
| Tool Selection Accuracy (n = 100) | 65.00% | 65.00% |
| Tool Selection Accuracy (valid decisions, n = 73) | 89.04% | 89.04% |
| Chinese accuracy (n = 50) | 60.00% | 60.00% |
| English accuracy (n = 50) | 70.00% | 70.00% |
| Prediction agreement | — | **100 / 100** |
| Decision-type agreement | — | **100 / 100** |
| FP32 correct / FP16 wrong | — | **0** |
| FP32 wrong / FP16 correct | — | **0** |
| Parameter memory after load | 1.29 GB (1228.0 MiB) | 1.29 GB (1228.0 MiB) |
| cap_router selection median | 450.0 ms | 459.5 ms |
| Container memory after batch | 1.672 GiB | 1.670 GiB |
| Download / disk | 1261.5 MiB | 647.5 MiB |

Both arms completed all 100 UI turns with zero UI, request, timeout or SSE failures, and
produced the same decision distribution (`SELECTED 65`, `REJECT 7`, `MATCHER_HIT 1`, no
decision 27). Confidence values differ by at most `1.0e-4` (median `0.0`), with no argmax
change. End-to-end wall-clock differences come from the client, IPC and host load; the
`selection_ms` that contains the inference is essentially unchanged.

#### Three precisions, not one

```text
checkpoint storage dtype      runtime parameter dtype      compute dtype
        F16              →            F32               →      F32
   (model.safetensors)      (build_model + copy_)       (CPU, no autocast)
```

- **Checkpoint storage dtype** — how the weights are stored on disk.
- **Runtime parameter dtype** — how the loaded weights are represented in memory.
- **Compute dtype** — the precision used by inference operations.

Only the second determines steady-state weight memory; only the third determines arithmetic
behavior.

`laya 0.3.21` builds the model with float32 parameters and copies the checkpoint tensors into
them:

1. `build_model(..., pretrained=False)` allocates the parameters as `float32`;
2. `load_file` reads the checkpoint tensors, which are `float16`;
3. `load_state_dict` copies each tensor into its float32 slot; PyTorch casts the dtype during
   the copy, without checking or warning.

The parameter container therefore stays float32, and memory is charged at four bytes per
parameter. The values remain the FP16-rounded ones; only the container is float32. The loader
exposes no dtype option. Setting `torch.set_default_dtype(torch.float16)` before construction
does not help: much of the encoder explicitly creates float32 tensors, producing a mixed
model (134 float32 + 35 float16 parameters) that raises `mixed dtype` on CPU.

#### CPU and GPU

| | CPU | GPU |
|---|---|---|
| Stock loader | float32 parameters, ~1.29 GB weights in memory | float32 parameters, ~1.29 GB weights in VRAM |
| Compute precision | float32; BF16 only with `LAYA_CPU_AMP=bf16` | autocast on by default (BF16/FP16); can be disabled explicitly, on OOM fallback, or when an operator is unsupported; MPS also gates per call by batch size |
| Explicit `model.half()` | 614.4 MiB parameters; slower on CPUs without native FP16 units | 614.4 MiB parameters; typically faster on recent GPUs |

An FP16 file does not reduce weight memory on either device, because both construct float32
parameters first. The difference is compute: a GPU defaults to mixed precision and may be
faster; a CPU defaults to float32.

#### True half-precision parameters

Measured in one process with the FP16 directory on CPU:

| Parameter form | Parameter memory | Warm forward |
|---|---:|---|
| float32 (stock) | 1228.0 MiB | 290 / 372 / 353 / 382 ms |
| float16 + float32 `act_head` | **614.4 MiB** | 784 / 840 / 747 / 793 ms |
| bfloat16 + float32 `act_head` | **614.4 MiB** | 847 / 972 / 1036 / 1055 ms |

A fully-half model fails because `DecisionModel.forward` performs
`pooled = h[:, 0].float()` before `act_head` (`laya/common.py:347`). Keeping the 0.8 MiB
`act_head` in float32 makes the forward run. This probe does not modify laya, and it has not
been re-validated on the full case set, so it is not a production result.

#### INT8 scope

Quantization savings depend on which tensors are converted:

| Component | Size |
|---|---:|
| Token embedding table (`encoder.embeddings`) | 750.0 MiB |
| Encoder linear and attention weights | 420.9 MiB |
| Decision head (`head`) | 54.1 MiB |
| `scorer` / `act_head` / `type_emb` | 3.1 MiB |

Quantizing Linear layers alone leaves the 750 MiB embedding table in float32, for roughly
870 MiB total — more than the 614 MiB of an FP16 parameter cast. Going materially below
614 MiB requires compressing the embedding table as well.

#### Conclusion

- The FP16 directory is a **storage and distribution artifact**. It halves download and disk
  footprint and leaves decisions unchanged.
- Under the stock `laya==0.3.21` loader it does **not** reduce CPU memory or GPU VRAM, and it
  does not change model latency.
- A true reduced-memory runtime requires casting the parameters (`model.half()`, with
  `act_head` kept in float32) or a runtime that genuinely stores and executes quantized
  weights. Both change speed and require their own accuracy measurement; neither is implied
  by the FP16 directory.

Raw evidence (per-case rows, logs, model manifests) is kept in the scratch directory
`logs/_e2e_formal500_v2_fp32_vs_fp16_20261021/`.

## 18. Reproducibility

| | |
|---|---|
| GPU | 1× NVIDIA L4 |
| torch | 2.14.1+cu130 |
| transformers | 5.17.0 |
| Python | 3.10.20 |
| laya | 0.3.21 |
| Base revision | `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` (subfolder `multilingual`) |
| Seed base | 42; calibration split seed 20260922 |

What must hold for a rerun to be comparable:

1. `laya==0.3.21` and the pinned base revision (asserted on load).
2. The frozen bundle unchanged — the hashes in §7.
3. The same three budget constants, and the same 256-token builder override (§9), on
   training, validation, final test, export and reload.
4. The same selection rule (§12) and the same calibration ids (§13).

## 19. Known Limitations

1. The model is trained for in-domain capability selection on the frozen
   `recall_threshold_082` bundle. It does not generalise automatically to unseen capability
   taxonomies, card renderings, or domains.
2. Validation and test rows come from the same frozen benchmark family; the very high test
   metrics should be read as in-domain performance, not open-domain robustness.
3. The model assumes the question contains exactly one `REJECT` card and 0–3 capability
   cards. Feeding a question without `REJECT`, or with more than three capability cards, is
   outside the trained contract.
4. Confidence is calibrated on 400 held-out in-distribution rows. Under distribution shift,
   both top-1 and calibration may degrade.
5. The 5-row loader smoke in the export stage is a load check, not an accuracy measurement;
   the reported accuracy figures come from the full test evaluation.
6. FP16 is a storage-precision copy. With the stock `laya` loader it runs as FP32 and does
   not reduce runtime parameter memory or VRAM. The drift measured in §17.1 is a 100-row
   end-to-end check, not a full test re-evaluation.

## 20. Final Conclusion

LayaChoice-v2 is trained for one task: given a question and a few candidate capability
cards (plus one `REJECT` card), pick the correct card — or pick `REJECT` when the right
capability is not among the candidates.

### Which epoch was selected

The model was trained for 10 epochs. Epoch 9 was chosen using
[`val.jsonl`](https://huggingface.co/datasets/eric-ml-nlp/Delveta-LayaChoice-v2-Data/blob/main/val.jsonl)
(2 407 rows), where it scored **0.996676** top-1. The test file was not involved in this
choice.

### Final score

On [`test.jsonl`](https://huggingface.co/datasets/eric-ml-nlp/Delveta-LayaChoice-v2-Data/blob/main/test.jsonl)
(9 729 rows), epoch 9 scores **0.998355** top-1 — its first choice was the correct option in
9 713 of 9 729 rows. The test file was not used for training, for choosing the epoch, for
calibration, or for deciding which checkpoint to export. The per-target breakdown is in
[§11](#11-epoch-110-training-results); the full test report is in
[§14](#14-final-test-results).

### The capability v1 did not have

v1 always had to pick one of three fixed options, even when none of them was correct. v2 can
pick `REJECT` instead:

- Of the 843 test rows whose correct answer is `REJECT`, it chose `REJECT` 840 times
  → **recall 99.64 %**.
- Of the 844 rows where it chose `REJECT`, 840 were correct → **precision 99.53 %**.
- Of the 8 886 rows whose correct answer is a capability, only 4 were wrongly rejected
  (**0.045 %**).

### What was published

The exported
[FP32 weights](https://huggingface.co/eric-ml-nlp/Delveta-LayaChoice-v2/blob/main/FP32/model.safetensors)
are byte-identical to the selected checkpoint (sha256 `964f8fe5…`), load with
`laya.Agent(strict=True)`, and reproduce the fitted temperature exactly.

### How this compares with v1

Not directly. [v1](LayaChoice-v1-Fine-Tuning.md) reports 93.56 %, but it was trained on
856 rows instead of 36 197, offered a fixed 3-way choice instead of dynamic K+1, and was
tested on 900 rows instead of 9 729.

### What the number does not mean

0.998355 is accuracy on this frozen benchmark: 18 capability cards, dynamic K+1,
zh/en/mixed rows from the same benchmark family as the training data. It is not a
production-traffic number and says nothing about unseen capabilities or open-domain
questions.
