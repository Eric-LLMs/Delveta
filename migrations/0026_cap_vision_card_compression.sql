-- Delveta data migration 0026 — compress the cap-vision card to fit the Laya budget.
--
-- WHY APPEND-ONLY: the runner (``core.infrastructure.db.init_db``) records each
-- applied file by filename stem in ``schema_migrations`` and never re-runs an
-- applied version; there is no content checksum. The correction must therefore
-- travel forward as a new version.
--
-- The previous cap-vision card was 283 tokens before the per-option cap, so the
-- Laya builder truncated it at ``OPTION_MAX_TOKENS = 256`` and dropped the tail
-- of ``asset_id`` plus the whole optional ``question`` slot. The compressed text
-- below measures 224 tokens (229 including the ``cap-vision:`` label and
-- ``[MASK]``), so no part of the card is clipped.
--
-- Scope: registry TEXT only — the capability description and two parameter
-- descriptions. No parameter name, type, requiredness, ``max_len``, arg slot,
-- permission or tool schema is changed. Both surfaces are aligned: the live
-- Registry truth (``capabilities``) and the static inventory (``action_catalog``,
-- which a fresh bootstrap copies from in 0014). ``action_catalog`` has no
-- parameter column, so the parameter text lives only on ``capabilities``.
--
-- Idempotent: re-applying writes the same strings and the same ``||`` merge.

UPDATE public.capabilities
   SET description = $desc$Analyze an image (screenshot, chart, photo, diagram). Use it when the user asks what an image contains or means. Not for reading text documents or extracting PDF text.

中文：分析附件或阅读器中的图片（截图、图表、照片、示意图），回答图里有什么或表达什么。不适用于 PDF 或文本文档。$desc$,
       parameters = coalesce(parameters, '{}'::jsonb) || $json${
           "page": {
               "type": "integer",
               "required": false,
               "description": "1-based PDF page to render and read; omitted -> the page open in the viewer (a multi-page PDF with no page in view is refused)"
           },
           "region": {
               "type": "object",
               "required": false,
               "description": "optional sub-region of the rendered page as fractions {x, y, w, h} (0..1, top-left origin)"
           }
       }$json$::jsonb,
       updated_at = now()
 WHERE capability_id = 'cap-vision';

UPDATE public.action_catalog
   SET description = $desc$Analyze an image (screenshot, chart, photo, diagram). Use it when the user asks what an image contains or means. Not for reading text documents or extracting PDF text.

中文：分析附件或阅读器中的图片（截图、图表、照片、示意图），回答图里有什么或表达什么。不适用于 PDF 或文本文档。$desc$,
       updated_at = now()
 WHERE action_key = 'vision';
