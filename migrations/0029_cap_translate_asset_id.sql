-- Delveta data migration 0029 — cap-translate gains the deferred-capture asset id.
--
-- WHY: the translate tool no longer asks Qwen for a payload. The intent-funnel
-- handler certifies the user's sentence VERBATIM as ``text`` and, for a VIDEO
-- viewer, reuses the vision lane's deferred capture: the PC client uploads the
-- current frame and ``resume_plan`` patches the uploaded asset id into the
-- action as ``asset_id``. That slot must therefore exist in the Registry
-- parameters, exactly matching the runtime ``translate`` schema
-- (``snapshot._parameter_issues`` rejects any divergence in either direction).
--
-- WHAT: parameters only. ``text`` stays REQUIRED (the handler always supplies
-- the user's sentence). ``asset_id`` is OPTIONAL and is never a model output:
-- the Binder sources it from turn facts, and the capture resume injects the
-- client's real upload. No ``max_len`` is set, exactly like the tool schema.
--
-- Idempotent: the ``||`` merge re-applies the same ``asset_id`` object.
-- APPEND-ONLY: any later correction must be a new numbered file.

BEGIN;

UPDATE public.capabilities
SET parameters = coalesce(parameters, '{}'::jsonb) || $json${
        "asset_id": {
            "type": "string",
            "required": false,
            "description": "captured video-frame asset id injected by the funnel after a deferred capture; never a model output"
        }
    }$json$::jsonb
WHERE capability_id = 'cap-translate';

COMMIT;
