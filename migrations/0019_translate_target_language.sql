-- Delveta data migration 0019 — cap-translate gains an optional target language.
--
-- Slot-extraction wiring: the unified local-Qwen acquisition lane can now resolve
-- a semantic ``target_language`` for translation. The translate executor has always
-- hard-coded an into-Chinese render; the confirmed contract retires that and makes
-- the target an OPTIONAL slot — when the user names one it is extracted and used,
-- and when none is named the executor applies a deterministic ENGLISH default (the
-- default is the executor's constant, never a model output, so the model is only
-- asked to fill a language the sentence actually states).
--
-- PARAMETER-SCHEMA sync only: this mirrors the tool schema into the live Registry
-- ``parameters`` so the 0014 write gate (registry slot-name set == runtime tool
-- schema set) stays green — ``translate_tool`` declares the matching ``target_
-- language`` string slot. The REQUIRED ``text`` slot is untouched; ``target_
-- language`` is optional, so an absent value never blocks certification and the
-- executor default owns it. No ``max_len`` is set here, exactly like the tool's
-- own ``target_language`` (no maxLength), so the runtime max_len cross-check cannot
-- disagree. ``arg_slots`` / declaration are deliberately NOT changed: the slot is
-- added to the schema; per-capability acquisition ownership is the handler's
-- ``slot_plan``, not this table's declaration.
--
-- Idempotent: the ``||`` merge re-applies the same ``target_language`` object.

BEGIN;

UPDATE public.capabilities
SET parameters = coalesce(parameters, '{}'::jsonb) || $json${
        "target_language": {
            "type": "string",
            "required": false,
            "description": "optional target language name (e.g. \"English\", \"Chinese\"); omitted -> executor default (English)"
        }
    }$json$::jsonb
WHERE capability_id = 'cap-translate';

COMMIT;
