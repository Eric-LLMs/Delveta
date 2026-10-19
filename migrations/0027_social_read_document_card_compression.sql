-- Delveta data migration 0027 — compress the social-search and read-document
-- capability cards to fit the Laya option budget.
--
-- WHY APPEND-ONLY: the runner (``core.infrastructure.db.init_db``) records each
-- applied file by filename stem in ``schema_migrations`` and never re-runs an
-- applied version; there is no content checksum. The correction must therefore
-- travel forward as a new version.
--
-- ``cap-social-search`` measured 250 tokens and ``cap-read-document`` 231 before
-- this change; both fit the 256-token per-option cap but left the three-card-plus-
-- REJECT combination above the shared 768-token head budget. The compressed text
-- below measures 223 and 228 tokens under the real V2 option encoding, so both
-- cards stay complete and the worst legal four-option combination no longer
-- reaches the ``opt_budget < 16`` fallback.
--
-- Scope: registry TEXT only — the parameter descriptions of these two
-- capabilities. No parameter name, type, requiredness, ``max_len``, arg slot,
-- permission or tool schema is changed; the capability descriptions and the
-- REJECT card are untouched. ``action_catalog`` carries no parameter column, so
-- only ``capabilities`` is updated. Full slot objects are supplied (not a
-- per-slot ``||`` merge) because the top-level merge replaces nested objects
-- wholesale.
--
-- Idempotent: re-applying writes the same strings and the same ``||`` merge.

UPDATE public.capabilities
   SET parameters = coalesce(parameters, '{}'::jsonb) || $json${
           "limit": {
               "type": "integer",
               "required": false,
               "description": "max results per platform; only if stated"
           },
           "query": {
               "type": "string",
               "required": true,
               "description": "search query copied verbatim from the user's sentence; do not rewrite, translate, summarize, expand or paraphrase"
           },
           "platform": {
               "type": "string",
               "required": false,
               "description": "platform reddit / x / zhihu; omit unless named"
           },
           "subreddit": {
               "type": "string",
               "required": false,
               "description": "reddit-only subreddit; omit unless named"
           }
       }$json$::jsonb,
       updated_at = now()
 WHERE capability_id = 'cap-social-search';

UPDATE public.capabilities
   SET parameters = coalesce(parameters, '{}'::jsonb) || $json${
           "pages": {
               "type": "string",
               "required": false,
               "description": "page/slide spec \"2\" or \"1-3\"; only if stated"
           },
           "asset_id": {
               "type": "string",
               "max_len": 64,
               "required": true,
               "description": "ID of the attached or opened document, obtained from turn facts, not the user's sentence"
           }
       }$json$::jsonb,
       updated_at = now()
 WHERE capability_id = 'cap-read-document';
