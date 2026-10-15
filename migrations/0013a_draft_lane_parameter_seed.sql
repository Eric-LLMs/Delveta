-- Delveta schema migration 0013a — Draft-lane parameter seed (fresh-install data fidelity).
--
-- Ordering matters: this file sorts AFTER 0013 (its rows come from 0013's INSERTs)
-- and BEFORE 0014 (which copies action_query_drafts.parameters into ``capabilities``
-- and then drops the draft tables). See 0012a for the registry-lane half of the
-- same fidelity fix.
--
-- Why this file exists. 0014 seeds the 15 draft-lane capabilities' ``parameters`` /
-- ``arg_slots`` from ``action_query_drafts`` — but nothing ever WROTE the parameter
-- bodies there. On the curated database those bodies were entered through the admin
-- plane; no migration carries them. A fresh database therefore reaches 0014 with
-- ``action_query_drafts.parameters = '{}'`` and records empty slots. Worse, the later
-- parameter-sync migrations do not repair it: 0016/0017/0019 use ``||`` (which merges
-- into '{}' and so lands only the slot they add), and 0020–0023 use ``jsonb_set`` on a
-- nested path ``{query,description}`` (which is a SILENT no-op when ``query`` does not
-- yet exist). The result is a Registry that mis-declares the funnel's argument
-- contract: required slots such as ``translate.text`` and ``pdf_table_to_text.asset_id``
-- missing, ``cap-vision`` / ``cap-read-document`` / the three search caps entirely empty.
--
-- This migration materialises the canonical parameter configuration for the 11
-- draft-lane Actions that carry one, so the incremental chain is self-contained and a
-- fresh install reproduces the same Registry as the curated database. The 4 remaining
-- draft-lane Actions (summary, mindmap, slides, research) are canonically EMPTY on the
-- curated database and are deliberately left so: their arguments are resolved from the
-- workspace/turn context by the executing toolkit, never extracted from the sentence,
-- so declaring slots for them would wrong-route the funnel.
--
-- The values are the FULL final bodies (the post-0023 search ``query`` description,
-- the ``pages``/``target_language`` slots, the raw JSON-Schema form the two-shape intake
-- adapter in ``registry/parameter_schema.py`` normalizes). Because they already equal
-- what 0016–0023 would produce, those later syncs re-apply identical values and the
-- chain converges on the same Registry.
--
-- Idempotent by construction: it only runs while the pre-Registry draft table still
-- exists. On every already-migrated database — the curated one included — 0014 has
-- already dropped ``action_query_drafts``, so the guard returns immediately and no
-- existing ``capabilities`` row is touched.

DO $$
BEGIN
    IF to_regclass('public.action_query_drafts') IS NULL THEN
        -- Post-0014: the draft table is retired; the live Registry is already seeded.
        RETURN;
    END IF;

    -- ── search lane: verbatim-topic query + optional controls ──────────────────────
    UPDATE public.action_query_drafts SET parameters = $json${
        "query": {"type": "string", "required": true, "description": "the user's search TOPIC, extracted from the sentence and kept VERBATIM — do not rewrite, translate, summarize, expand or paraphrase"},
        "top_k": {"type": "integer", "required": false, "description": "optional result count; only when the sentence states one"},
        "domain": {"type": "string", "required": false, "description": "optional domain NAME to scope the search to (resolved to a real domain id by the tool); only when the sentence names one"}
    }$json$::jsonb WHERE action_key = 'rag_search';

    UPDATE public.action_query_drafts SET parameters = $json${
        "query": {"type": "string", "required": true, "description": "the user's search TOPIC, extracted from the sentence and kept VERBATIM — do not rewrite, translate, summarize, expand or paraphrase"},
        "top_k": {"type": "integer", "required": false, "description": "optional number of results; only when the sentence states one"}
    }$json$::jsonb WHERE action_key = 'web_search';

    UPDATE public.action_query_drafts SET parameters = $json${
        "limit": {"type": "integer", "required": false, "description": "optional max results per platform; only when the sentence states one"},
        "query": {"type": "string", "required": true, "description": "the user's search TOPIC, extracted from the sentence and kept VERBATIM — do not rewrite, translate, summarize, expand or paraphrase"},
        "platform": {"type": "string", "required": false, "description": "optional platform (reddit / x / zhihu); only when the sentence names one; OMIT when no specific platform is named"},
        "subreddit": {"type": "string", "required": false, "description": "optional reddit-only scope; only when the sentence names one"}
    }$json$::jsonb WHERE action_key = 'social_search';

    -- ── content-reading lane: attachment/id + viewer page window ───────────────────
    UPDATE public.action_query_drafts SET parameters = $json${
        "asset_id": {"type": "string", "max_len": 64, "required": true, "description": "asset id of the attached/opened image (from turn facts, not the sentence)"},
        "question": {"type": "string", "required": false, "description": "optional question about the image, copied from the sentence"}
    }$json$::jsonb WHERE action_key = 'vision';

    UPDATE public.action_query_drafts SET
        parameters = $json${
            "pages": {"type": "string", "required": false, "description": "optional page/slide spec like \"2\" or \"1-3\"; only when the sentence states one"},
            "asset_id": {"type": "string", "max_len": 64, "required": true, "description": "asset id of the attached/opened document (from turn facts, not the sentence)"}
        }$json$::jsonb,
        arg_slots = $json${"pages": {"source": "viewer.current_page"}}$json$::jsonb
     WHERE action_key = 'read_document';

    UPDATE public.action_query_drafts SET
        parameters = $json${
            "pages": {"type": "string", "required": false, "description": "optional page spec like \"2\" or \"1-3\" from the viewer context; omitted = whole document"},
            "asset_id": {"type": "string", "max_len": 64, "required": true, "description": "asset id of the attached/opened document (from turn facts, not the sentence)"}
        }$json$::jsonb,
        arg_slots = $json${"pages": {"source": "viewer.current_page"}}$json$::jsonb
     WHERE action_key = 'pdf_table_to_text';

    -- ── translate: text + optional target language ─────────────────────────────────
    UPDATE public.action_query_drafts SET parameters = $json${
        "text": {"type": "string", "required": true, "description": "the text to translate, copied from the sentence"},
        "target_language": {"type": "string", "required": false, "description": "optional target language name (e.g. \"English\", \"Chinese\"); omitted -> executor default (English)"}
    }$json$::jsonb WHERE action_key = 'translate';

    -- ── fs / plugin lane: raw JSON-Schema bodies (normalized at read time) ─────────
    -- These four carry the tool's own JSON-Schema shape, which the two-shape intake
    -- adapter (``registry/parameter_schema.py``) folds to the canonical flat form.
    UPDATE public.action_query_drafts SET parameters = $json${
        "type": "object",
        "required": ["path"],
        "properties": {
            "path": {"type": "string", "description": "Workspace-relative file path, or a drive path \"My Drive/<path>\"."},
            "max_chars": {"type": "integer", "description": "Optional read cap."}
        }
    }$json$::jsonb WHERE action_key = 'read_file';

    UPDATE public.action_query_drafts SET parameters = $json${
        "type": "object",
        "required": ["path"],
        "properties": {
            "path": {"type": "string", "description": "Workspace-relative file path, or a drive path \"My Drive/<path>\"."},
            "new_text": {"type": "string", "description": "Replacement / new file content."},
            "old_text": {"type": "string", "description": "Exact text to replace (optional)."}
        }
    }$json$::jsonb WHERE action_key = 'edit_file';

    UPDATE public.action_query_drafts SET parameters = $json${
        "type": "object",
        "required": ["command"],
        "properties": {
            "command": {"type": "string", "description": "The shell command to run."},
            "timeout": {"type": "integer", "description": "Timeout in seconds."}
        }
    }$json$::jsonb WHERE action_key = 'bash';

    UPDATE public.action_query_drafts SET parameters = $json${
        "type": "object",
        "required": ["action"],
        "properties": {
            "action": {"enum": ["compile_pdf", "status"], "type": "string", "description": "Which action to run. One of: compile_pdf, status."},
            "run_id": {"type": "string", "description": "status: the compile run id; compile_pdf: optional replay id override."},
            "project_id": {"type": "string", "description": "compile_pdf: research project whose primary manuscript is compiled."}
        }
    }$json$::jsonb WHERE action_key = 'artifact';
END $$;
