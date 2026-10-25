-- Delveta data migration 0030 — cap-summary gains the explicit ``auto`` marker.
--
-- WHY: summary_gen no longer runs a page-scope model or a shared argument
-- extractor. The handler certifies ``auto`` when the user's sentence does not
-- fully specify a source, and the tool's FIRST LLM call decides the target
-- (previous message / open document and pages / whole session), whether it can
-- already answer, and the save intent. The marker must exist in the Registry
-- parameters so the Binder's schema and the runtime ``summary_gen`` schema carry
-- the same slot set (``snapshot._parameter_issues``).
--
-- WHAT: parameters only; every slot stays optional (a summary has exactly ONE
-- source, chosen by the turn). ``auto`` is a trusted, funnel-set marker — never
-- a user payload — so its presence is all the tool reads.
--
-- Idempotent: the ``||`` merge re-applies the same ``auto`` object.
-- APPEND-ONLY: any later correction must be a new numbered file.

BEGIN;

UPDATE public.capabilities
SET parameters = coalesce(parameters, '{}'::jsonb) || $json${
        "auto": {
            "type": "string",
            "required": false,
            "description": "internal marker set by the intent funnel: the tool's first LLM call decides the source and the save intent"
        }
    }$json$::jsonb
WHERE capability_id = 'cap-summary';

COMMIT;
