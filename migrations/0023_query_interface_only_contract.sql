-- PE-v1 alignment: the search `query` description is INTERFACE-ONLY.
--
-- 0021 made the query a verbatim topic; 0022 added an explicit result-count rule.
-- Both taught the model a SEMANTIC transformation (which phrase to strip) in the
-- slot description. The unified contract keeps descriptions to the INTERFACE fact
-- only: WHICH text to extract (the search topic) and that it is kept VERBATIM; the
-- extraction behaviour is learned from the SFT data, never from a prose rule.
--
-- This is the append-only step (0021/0022 are already applied, so their files must
-- not be edited): it rewrites ONLY the `query` description of the three search
-- capabilities to the interface-only string.
--
-- Description text only — no schema change, no row removed, no slot added or
-- dropped, no behaviour change beyond the wording the model reads. Reversible by
-- restoring the 0022 string.
--
-- Kept in lockstep with the frozen snapshot
-- ``scripts/qwen_parameter_extraction_finetune/pe_contract.py``
-- (``_SEARCH_QUERY_DESC``). Applying this migration also moves the pin
-- ``tests/test_registry_contract_alignment.py::FROZEN_PARAMETERS`` — update it in
-- the same change that applies the migration.

UPDATE capabilities
SET parameters = jsonb_set(
        parameters, '{query,description}',
        $q$"the user's search TOPIC, extracted from the sentence and kept VERBATIM — do not rewrite, translate, summarize, expand or paraphrase"$q$::jsonb)
WHERE capability_id IN ('cap-web-search', 'cap-rag-search', 'cap-social-search');
