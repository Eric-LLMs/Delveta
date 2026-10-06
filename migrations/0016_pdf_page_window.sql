-- Delveta data migration 0016 — the PDF read tools gain an optional page window.
--
-- Phase 2 (file-content Scope/Range): a file-content read capability must carry
-- a page/range. Both PDF tools now declare an OPTIONAL ``pages`` slot (a 1-based
-- spec like "3" or "1-3") sourced from the viewer context and threaded down to
-- the extractor, so a page-scoped request reads ONLY those pages — a spec that is
-- out of range fails the call rather than silently widening to the whole file.
--
-- PARAMETER-SCHEMA sync only: this mirrors the tool schema into the live Registry
-- ``parameters`` so the 0014 write gate (registry slot-name set == runtime tool
-- schema set) stays green. Required slots are untouched (``pages`` is optional),
-- and ``arg_slots`` / declaration are deliberately NOT changed here — page-window
-- ownership on the acquisition lane is a separate (Phase 2-C) decision, so the
-- derived declaration keeps the name-default (MODEL), exactly like the existing
-- ``cap-read-document.pages`` row.
--
-- Idempotent: the ``||`` merge re-applies the same ``pages`` object.

BEGIN;

UPDATE public.capabilities
SET parameters = coalesce(parameters, '{}'::jsonb) || $json${
        "pages": {
            "type": "string",
            "required": false,
            "description": "optional page spec like \"2\" or \"1-3\" from the viewer context; omitted = whole document"
        }
    }$json$::jsonb
WHERE capability_id IN ('cap-pdf-extract-text', 'cap-pdf-table-to-text');

COMMIT;
