-- Delveta data migration 0031 — sessions gain an additive metadata bag.
--
-- WHY: the WorkSpace assistant needs to remember which workspace (and which
-- folder, when the user started the session inside one) a session belongs to.
-- Today `sessions` carries only user_id/title/type, so a WorkSpace conversation
-- has no durable scope and dies with the app's in-memory per-mode pointer.
--
-- WHAT: one nullable-free JSONB column, default '{}'. It stores descriptive
-- scope metadata written by the client through PATCH /sessions/{id}:
--   { "workspace_id": "<uuid>|null", "workspace_name": "<name>",
--     "folder_path": "<path>|null", "folder_name": "<name>|null" }
--
-- SCOPE NOTE: this column is descriptive in this phase. Retrieval (`rag_search`
-- / the recall SQL) still filters by user visibility only, so a stored folder
-- path does NOT restrict what the agent can retrieve yet. The UI must not claim
-- otherwise until the recall filter lands.
--
-- Idempotent: ADD COLUMN IF NOT EXISTS; existing rows take the empty object.
-- APPEND-ONLY: any later correction must be a new numbered file.

BEGIN;

ALTER TABLE public.sessions
    ADD COLUMN IF NOT EXISTS meta jsonb NOT NULL DEFAULT '{}'::jsonb;

COMMIT;
