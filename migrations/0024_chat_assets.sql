-- Chat Assets — chat-only attachments, hidden from the drive plane.
--
-- Three additive structures plus one retirement:
--   1. ``assets.scope``             — the visibility dimension ('drive' | 'chat').
--   2. ``global_objects.state``     — durable retirement state so physical GC is
--                                     crash-safe (''live'' | ''pending_delete'').
--   3. ``chat_attachments``         — the authority for "this asset belongs to this
--                                     chat" (owned / reference, draft / bound).
--   4. drop ``messages.attach_asset_id`` — single-valued; superseded by chat_attachments.
--
-- Additive except the column drop: every existing asset defaults to scope='drive',
-- every existing object stays state='live', and no row is rewritten or deleted.
-- Depends on 0001 only; no other migration in the chain touches these tables.

BEGIN;

-- 1. Visibility dimension: drive-visible vs chat-only. Not a UI category — it is the
--    discriminator the shared visibility predicate keys on.
ALTER TABLE public.assets ADD COLUMN IF NOT EXISTS scope text NOT NULL DEFAULT 'drive';
ALTER TABLE public.assets DROP CONSTRAINT IF EXISTS assets_scope_check;
ALTER TABLE public.assets ADD CONSTRAINT assets_scope_check CHECK (scope IN ('drive', 'chat'));
CREATE INDEX IF NOT EXISTS assets_chat_owner_idx ON public.assets (user_id) WHERE scope = 'chat';

-- 2. CAS object-retirement state. A freed object flips to 'pending_delete'; physical GC
--    (file delete then row delete) only ever runs against this state, under a per-sha
--    advisory lock. A concurrent upload re-increments and flips it back to 'live'.
ALTER TABLE public.global_objects ADD COLUMN IF NOT EXISTS state text NOT NULL DEFAULT 'live';
ALTER TABLE public.global_objects DROP CONSTRAINT IF EXISTS global_objects_state_check;
ALTER TABLE public.global_objects
    ADD CONSTRAINT global_objects_state_check CHECK (state IN ('live', 'pending_delete'));
ALTER TABLE public.global_objects ADD COLUMN IF NOT EXISTS pending_since timestamptz;
CREATE INDEX IF NOT EXISTS global_objects_pending_idx
    ON public.global_objects (pending_since) WHERE state = 'pending_delete';

-- 3. Chat attachment authority. One row per (asset, chat) link. ``owned`` assets are
--    scope='chat' and die with the chat; ``reference`` assets are normal drive files
--    (scope='drive') that the chat merely points at and must never delete.
CREATE TABLE IF NOT EXISTS public.chat_attachments (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    asset_id    uuid NOT NULL REFERENCES public.assets(id)   ON DELETE CASCADE,
    user_id     uuid NOT NULL REFERENCES public.users(id)    ON DELETE CASCADE,
    session_id  uuid          REFERENCES public.sessions(id) ON DELETE CASCADE,
    message_id  uuid          REFERENCES public.messages(id) ON DELETE CASCADE,
    ownership   text NOT NULL DEFAULT 'owned'
                CHECK (ownership IN ('owned', 'reference')),
    source      text NOT NULL DEFAULT 'upload'
                CHECK (source IN ('upload', 'screenshot', 'frame', 'roi', 'viewer')),
    status      text NOT NULL DEFAULT 'draft'
                CHECK (status IN ('draft', 'bound')),
    locator     jsonb NOT NULL DEFAULT '{}'::jsonb,   -- frame/roi/page coordinates
    created_at  timestamptz NOT NULL DEFAULT now(),
    bound_at    timestamptz,
    expires_at  timestamptz,                            -- set while draft
    CONSTRAINT chat_attach_bound_needs_session
        CHECK (status = 'draft' OR session_id IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS chat_attach_session_idx
    ON public.chat_attachments (session_id) WHERE session_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS chat_attach_message_idx
    ON public.chat_attachments (message_id) WHERE message_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS chat_attach_asset_idx ON public.chat_attachments (asset_id);
CREATE INDEX IF NOT EXISTS chat_attach_draft_idx
    ON public.chat_attachments (expires_at) WHERE status = 'draft';

-- 4. Retire messages.attach_asset_id. It is single-valued, carries no ownership/source/
--    locator, and is superseded by chat_attachments. All four of its duties move there.
--
-- Guarded, fail-closed, in the SAME transaction:
--   * no non-null rows  -> straight drop (the ordinary case);
--   * non-null rows that are ALL classifiable -> carry the relations into
--     chat_attachments first, verify every row landed, then drop;
--   * any row whose asset is missing / soft-deleted / cross-user -> RAISE and roll
--     the whole migration back (never guess, never silently drop).
-- No DROP ... CASCADE is used; only this column, its index and its FK are removed.
DO $$
DECLARE
    legacy int;
    unclassifiable int;
    copied int;
BEGIN
    SELECT count(*) INTO legacy FROM public.messages WHERE attach_asset_id IS NOT NULL;
    IF legacy = 0 THEN
        RETURN;
    END IF;

    SELECT count(*) INTO unclassifiable
    FROM public.messages m LEFT JOIN public.assets a ON a.id = m.attach_asset_id
    WHERE m.attach_asset_id IS NOT NULL
      AND (a.id IS NULL OR a.deleted_at IS NOT NULL OR a.user_id <> m.user_id);
    IF unclassifiable > 0 THEN
        RAISE EXCEPTION
            '0024 aborted: % messages.attach_asset_id rows are unclassifiable; resolve before migrating',
            unclassifiable;
    END IF;

    INSERT INTO public.chat_attachments
        (asset_id, user_id, session_id, message_id, ownership, source, status, bound_at)
    SELECT m.attach_asset_id, m.user_id, m.session_id, m.id,
           CASE WHEN a.scope = 'chat' THEN 'owned' ELSE 'reference' END,
           'viewer', 'bound', now()
    FROM public.messages m JOIN public.assets a ON a.id = m.attach_asset_id
    WHERE m.attach_asset_id IS NOT NULL;

    GET DIAGNOSTICS copied = ROW_COUNT;
    IF copied <> legacy THEN
        RAISE EXCEPTION '0024 aborted: carried % of % legacy attachments', copied, legacy;
    END IF;
END $$;

ALTER TABLE public.messages DROP CONSTRAINT IF EXISTS messages_attach_asset_id_fkey;
DROP INDEX IF EXISTS public.idx_messages_attach_asset;
ALTER TABLE public.messages DROP COLUMN IF EXISTS attach_asset_id;

COMMIT;
