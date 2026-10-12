-- PE-v1 alignment: the search `query` also strips an EXPLICIT result-count control.
--
-- 0021 restored the verbatim-topic contract but still left a gap: the `query`
-- description named only the execution-instruction frame as removable, so a model
-- reading the Registry was never told what to do with an explicit result-count
-- control ("top 3 results", "10 results"). The training target and the production
-- system prompt (``prompts.SEARCH_QUERY_PROMPT``) already remove such a control AND
-- KEEP a number that belongs to the topic itself ("top 3 retrieval methods"). This
-- migration brings the Registry wording into the SAME rule.
--
-- Description text only — no schema change, no row removed, no slot added or
-- dropped, no behaviour change beyond the wording the model reads. Reversible by
-- restoring the 0021 string.
--
-- Kept in lockstep with the frozen snapshot
-- ``scripts/qwen_parameter_extraction_finetune/pe_contract.py``
-- (``_SEARCH_QUERY_DESC``) and the pin
-- ``tests/test_registry_contract_alignment.py::FROZEN_PARAMETERS`` — all three
-- must carry the identical string.
--
-- Relationship to 0021 (both unexecuted as of this change; applied in order):
--   0020 distilled-topic  ->  0021 verbatim-topic  ->  0022 verbatim + count rule.
-- The final applied value equals ``pe_contract._SEARCH_QUERY_DESC``.

UPDATE capabilities
SET parameters = jsonb_set(
        parameters, '{query,description}',
        $q$"the search query — the user's search TOPIC, copied verbatim from the sentence: remove ONLY the execution-instruction frame AND an explicit result-count control (a phrase that says how many RESULTS to return, e.g. \"top 3 results\", \"10 results\"); KEEP a number that is part of the TOPIC itself (e.g. \"top 3 retrieval methods\"); if you cannot tell, keep it; keep the user's own wording and language and any qualifiers (time range, domain, material type) as stated; never translate, summarize, expand or paraphrase"$q$::jsonb)
WHERE capability_id IN ('cap-web-search', 'cap-rag-search', 'cap-social-search');
