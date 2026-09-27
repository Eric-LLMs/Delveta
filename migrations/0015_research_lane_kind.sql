-- Delveta schema migration 0015 — the 'research' lane kind (lane ruling 2026-09-27).
--
-- The Registry's kind column is the lane-ownership axis: action/private/web
-- are chat/files funnel candidates (per-kind rollout gates), while 'research'
-- rows belong to the RESEARCH lane — visible in the table, corpus curated for
-- future research-side retrieval, but excluded from every chat/files funnel
-- plane in CODE (entry.chat_plane_candidate: Matcher index, Recall corpus,
-- entries_by_id; executor kind gate stays the certification backstop). The
-- research lane itself executes via Plugin mount + handoff and never consults
-- the row, so no runtime permission system is introduced.
--
-- The enum lives in code (registry.entry.VALID_KINDS, enforced by the write
-- gate); this CHECK is the DB-side backstop against hand-edited rows only —
-- widened to mirror the code roster.

BEGIN;

ALTER TABLE public.capabilities
    DROP CONSTRAINT IF EXISTS capabilities_intent_kind_check;

ALTER TABLE public.capabilities
    ADD CONSTRAINT capabilities_intent_kind_check
    CHECK (intent_kind IN ('action', 'private', 'web', 'research'));

COMMIT;
