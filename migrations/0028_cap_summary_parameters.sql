-- Delveta data migration 0028 — declare the real ``summary_gen`` parameter
-- schema for ``cap-summary`` so the funnel can acquire its arguments.
--
-- WHY: ``cap-summary`` was DEFERRED_AGENT because the tool required
-- ``paths: string[]`` and the chat path had no producer for a path array. The
-- tool now also accepts the string slots the funnel CAN legally produce (an
-- authorized ``asset_id`` from turn facts, ``pages`` from the page-scope
-- resolver, a literal workspace ``path``, the ``context`` transcript, and an
-- explicit ``output_dir``), so the Registry must carry exactly the runtime
-- slot set — ``snapshot._parameter_issues`` rejects any divergence in either
-- direction.
--
-- WHAT: parameters only. The capability description, intent kind, tool binding,
-- permissions and the ``action_catalog`` row are untouched (that table carries
-- no parameter column). Every slot is optional: a summary has exactly ONE
-- content source, chosen by the turn, and the tool body rejects an empty or
-- multi-source call with a readable error.
--
-- Card size: the six short descriptions keep the rendered ``B_noprov`` card
-- well under the per-option budget used by the choice question.
--
-- Idempotent: re-applying writes the same objects through the same ``||``
-- merge. APPEND-ONLY: the runner records the filename stem and never re-runs an
-- applied version, so any later correction must be a new numbered file.

UPDATE public.capabilities
   SET parameters = coalesce(parameters, '{}'::jsonb) || $json${
           "paths": {
               "type": "array",
               "required": false,
               "description": "workspace file paths to summarize (one or more)"
           },
           "path": {
               "type": "string",
               "required": false,
               "description": "one workspace file path"
           },
           "asset_id": {
               "type": "string",
               "required": false,
               "description": "attached or opened document id, from turn facts, not the sentence"
           },
           "pages": {
               "type": "string",
               "required": false,
               "description": "page/slide spec \"2\" or \"1-3\" for an asset_id document; omit for all"
           },
           "context": {
               "type": "string",
               "required": false,
               "description": "conversation transcript to summarize instead of a file"
           },
           "output_dir": {
               "type": "string",
               "required": false,
               "description": "explicit save directory; omit to return the summary in chat"
           }
       }$json$::jsonb,
       updated_at = now()
 WHERE capability_id = 'cap-summary';
