-- Delveta schema migration 0012a — Registry-lane capability seed (fresh-install chain fix).
--
-- Ordering matters: this file sorts AFTER 0012 (it needs the ``action_query_drafts``
-- configuration columns) and BEFORE 0013. It is deliberately numbered between them
-- because it must run before 0013's corpus UPDATEs — see below.
--
-- Why this file exists. 0013 is the canonical corpus seed for all 18 user-facing
-- Actions; it writes ``capabilities.standard_example`` / ``synonym_examples`` through
-- UPDATEs. 0014's hard acceptance gate then requires exactly 36 Standard rows: 30 from
-- the 15 draft-lane Actions (15 zh + 15 promoted en) plus 6 from the three Registry-lane
-- capabilities (cap-add-term, cap-create-folder, cap-pdf-extract-text — 3 zh + 3
-- promoted en).
--
-- On the curated database those three capabilities were admitted through the admin plane
-- (POST /admin/registry/catalog/{action_key}), so they already existed when 0013 and 0014
-- ran. No schema migration ever created them. A fresh database therefore reached 0013
-- with an empty ``capabilities`` table, its UPDATEs became no-ops, and 0014 aborted with
-- "0014: Standard must be exactly 36, got 30" — the whole fresh-install chain failed.
--
-- This migration materialises exactly those three rows so the incremental chain is
-- self-contained and a fresh install reproduces the same corpus as the curated database.
-- It only creates the capability shell; 0013 fills in the corpus and 0014 performs its
-- normal seed/gate on top.
--
-- Idempotent (ON CONFLICT (capability_id) DO NOTHING): on every already-migrated
-- database — including the curated one, where the rows exist — it is a no-op.
--
-- The parameter / arg_slots shape follows 0007's canonical
-- ``{slot: {type, description, required, max_len?}}`` contract. The three Registry-lane
-- capabilities carry the same configuration the curated database has; the 15 draft-lane
-- capabilities are still created by 0014 from ``action_query_drafts``.

-- Column list is deliberately limited to columns that exist in BOTH the pre-0014
-- schema (this migration's position in the fresh chain) and the post-0014 schema
-- (where a grown database -- e.g. the curated one -- applies this file standalone).
-- ``examples`` and ``negatives`` are DROPPED by 0014, so naming them here would make
-- this INSERT fail with ``column "examples" does not exist`` on any database that has
-- already reached 0014. Both default to '[]'::jsonb (0002), which is the value the
-- curated rows carry anyway, so omitting them is value-preserving.
INSERT INTO public.capabilities (
    capability_id, tool_binding, description,
    patterns, aliases,
    parameters, arg_slots,
    permissions, execution_policy, intent_kind,
    enabled, status, row_version
) VALUES (
    'cap-add-term', 'add_term',
    'Add a word, term, or expression to a named vocabulary domain.',
    '[]'::jsonb, '[]'::jsonb,
    '{"term": {"type": "string", "max_len": 120, "required": true, "description": "the word/term to add, quotes removed"}, "domain": {"type": "string", "max_len": 60, "required": true, "description": "the vocabulary domain to add it into"}}'::jsonb,
    '{"term": {"source": "user_input"}, "domain": {"source": "user_input"}}'::jsonb,
    '', 'auto', 'action',
    true, 'active', 0
), (
    'cap-create-folder', 'create_folder',
    'Create a folder with an explicit name in the workspace.',
    '[]'::jsonb, '[]'::jsonb,
    '{"name": {"type": "string", "max_len": 120, "required": true, "description": "the folder name, quotes removed"}}'::jsonb,
    '{"name": {"source": "user_input"}}'::jsonb,
    '', 'auto', 'action',
    true, 'active', 0
), (
    'cap-pdf-extract-text', 'pdf_extract_text',
    'Extract the complete raw text from an attached or resolved document.',
    '[]'::jsonb, '[]'::jsonb,
    '{"pages": {"type": "string", "required": false, "description": "optional page spec like \"2\" or \"1-3\" from the viewer context; omitted = whole document"}, "asset_id": {"type": "string", "max_len": 64, "required": true, "description": "asset id of the attached/opened document (from turn facts, not the sentence)"}}'::jsonb,
    '{"pages": {"source": "viewer.current_page"}, "asset_id": {"source": "attachment"}}'::jsonb,
    '', 'auto', 'action',
    true, 'active', 0
)
ON CONFLICT (capability_id) DO NOTHING;
