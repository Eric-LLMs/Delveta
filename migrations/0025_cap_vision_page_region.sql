-- Delveta data migration 0025 — the vision tool gains an optional page/region.
--
-- The ``vision`` tool now renders a PDF page (or a sub-region) to an image
-- LOCALLY before its single vision call, so its JSON schema declares two more
-- optional slots: ``page`` (a 1-based PDF page) and ``region`` (a sub-area as
-- fractions {x, y, w, h}). A page-scoped question reaches that locator from the
-- caller OR from the viewer context the tool already binds this turn; neither
-- slot is model-owned — the acquisition lane is unchanged.
--
-- PARAMETER-SCHEMA sync only: this mirrors the tool schema into the live
-- Registry ``parameters`` so the 0014 write gate (registry slot-name set ==
-- runtime tool schema set) stays green, exactly like 0016 did for the PDF
-- ``pages`` slot. Required slots are untouched (``page``/``region`` are
-- optional) and ``arg_slots`` / declaration are deliberately NOT changed here.
--
-- Idempotent: the ``||`` merge re-applies the same two objects.

UPDATE public.capabilities
SET parameters = coalesce(parameters, '{}'::jsonb) || $json${
        "page": {
            "type": "integer",
            "required": false,
            "description": "1-based PDF page to render and read; omitted -> the page open in the viewer (a multi-page PDF with no page in view is refused, never a silent page 1)"
        },
        "region": {
            "type": "object",
            "required": false,
            "description": "optional sub-region of the rendered page as fractions {x, y, w, h} (0..1, top-left origin), to focus a table/chart/figure; omitted -> the whole page"
        }
    }$json$::jsonb
WHERE capability_id = 'cap-vision';
