-- U6 + U1 registry-description alignment (append-only; description text only).
--
-- U6: the search `query` contract is a CLEAN TOPIC the model understands, not a
--     verbatim copy of the user's sentence. The per-slot description feeds the
--     extractor's slot block, so it must agree with the system prompt
--     (argument_acquisition/prompts.SEARCH_QUERY_PROMPT) — the old wording
--     "exactly as the user words it" directly contradicted it.
-- U1: rag `domain` is a plain domain NAME the business layer resolves to a real
--     domain id; it is never a model-fabricated UUID/id.
--
-- No schema change, no row removed, no behaviour change beyond the wording the
-- model reads. Reversible by restoring the prior description strings.

UPDATE capabilities
SET parameters = jsonb_set(
        parameters, '{query,description}',
        '"the search query — a clean, self-contained TOPIC distilled from the '
        'user sentence (instruction frame removed); the model understands the '
        'topic, it is NOT copied verbatim"')
WHERE capability_id IN ('cap-web-search', 'cap-rag-search', 'cap-social-search');

UPDATE capabilities
SET parameters = jsonb_set(
        parameters, '{domain,description}',
        '"optional domain NAME to scope the search to (resolved to a real '
        'domain id by the tool); only when the sentence names one"')
WHERE capability_id = 'cap-rag-search';
