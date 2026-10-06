-- Delveta schema migration 0018 — cap-read-file dual-plane description (Phase 3-B).
--
-- WHY APPEND-ONLY: the runner (``core.infrastructure.db.init_db``) records each
-- applied file by its filename stem in ``schema_migrations`` and NEVER re-runs an
-- applied version — there is no content checksum. So editing 0011 (which seeded
-- the action_catalog text) or 0014 (which copied it into ``capabilities``) is a
-- silent no-op on every DB that already applied them: the running Registry would
-- keep describing ``read_file`` as "workspace-local" even though the runtime
-- resolves BOTH a workspace-relative path and a ``My Drive/<path>`` drive path
-- (tool_binding ``read_file`` → ``fs_tools`` plane dispatch). The correction must
-- therefore travel forward as a new version.
--
-- Description text ONLY: no parameter, arg_slot, permission or schema-structure
-- change. Both surfaces are aligned in one place — the live Registry truth
-- (``capabilities``, which the funnel reads) and the static inventory
-- (``action_catalog``, which a fresh bootstrap copies from in 0014).

UPDATE public.capabilities
   SET description = 'Read the contents of an existing file from the workspace or My Drive using its path. It does not apply to attachments or viewer documents, or to image analysis.',
       updated_at  = now()
 WHERE capability_id = 'cap-read-file';

UPDATE public.action_catalog
   SET description = 'Read the contents of an existing file from the workspace or My Drive using its path. It does not apply to attachments or viewer documents, or to image analysis.',
       updated_at  = now()
 WHERE action_key = 'read_file';
