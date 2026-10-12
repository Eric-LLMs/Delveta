-- PE-v1 alignment: the search `query` is a VERBATIM topic, not a distilled one.
--
-- 0020 made the search `query` "a clean, self-contained TOPIC distilled from the
-- user sentence ... NOT copied verbatim". That wording told the model to
-- summarize/rephrase, which contradicted the training target (the topic copied
-- verbatim with only the execution frame stripped) and taught a lossy query.
-- This migration restores the verbatim-topic contract the model is trained and
-- evaluated against, and drops the `auto` platform hint so a generic social
-- request OMITS the slot instead of inventing a platform.
--
-- Description text only — no schema change, no row removed, no slot added or
-- dropped, no behaviour change beyond the wording the model reads. Reversible by
-- restoring the 0020 strings.
--
-- Kept in lockstep with the frozen snapshot
-- ``scripts/qwen_parameter_extraction_finetune/pe_contract.py``
-- (``_SEARCH_QUERY_DESC`` / the ``platform`` description) and the pin
-- ``tests/test_registry_contract_alignment.py::FROZEN_PARAMETERS`` — all three
-- must carry the identical strings.

UPDATE capabilities
SET parameters = jsonb_set(
        parameters, '{query,description}',
        $q$"the search query — the user's search TOPIC, copied verbatim from the sentence (only the execution-instruction frame is removed); keep the user's own wording and language and any qualifiers (time range, domain, material type) as stated; never translate, summarize, expand or paraphrase"$q$::jsonb)
WHERE capability_id IN ('cap-web-search', 'cap-rag-search', 'cap-social-search');

UPDATE capabilities
SET parameters = jsonb_set(
        parameters, '{platform,description}',
        $q$"optional platform (reddit / x / zhihu); only when the sentence names one; OMIT when no specific platform is named"$q$::jsonb)
WHERE capability_id = 'cap-social-search';
