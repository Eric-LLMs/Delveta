-- Delveta data migration 0017 — the page-window slot converges on the viewer context.
--
-- Phase 2-C (file-content Scope/Range): a file-content read's page window is settled
-- UPSTREAM from the viewer context (``TurnFacts.viewer_page_from`` / ``viewer_page_to``)
-- by the per-capability acquisition handler — the model never parses a page number from
-- the sentence and never mediates the range. The three page-addressed capabilities
-- therefore declare ``pages`` with the legacy ``viewer.current_page`` source, which the
-- declaration bridge (``argument_acquisition.declaration``) maps to SYSTEM_BINDER.
--
-- Phase 2-B left ``pages`` at the MODEL name-default deliberately (see 0016); this is the
-- Phase 2-C correction. It touches acquisition metadata ONLY — ``parameters`` (the schema
-- the write gate cross-checks) is unchanged, and ``required_slots`` is untouched (``pages``
-- stays optional).
--
-- Idempotent: the ``||`` merge re-applies the same ``pages`` source object.

BEGIN;

UPDATE public.capabilities
SET arg_slots = coalesce(arg_slots, '{}'::jsonb) || $json${
        "pages": {"source": "viewer.current_page"}
    }$json$::jsonb
WHERE capability_id IN
      ('cap-read-document', 'cap-pdf-extract-text', 'cap-pdf-table-to-text');

COMMIT;
