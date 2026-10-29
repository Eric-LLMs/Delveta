-- History pagination: keyset scans read a session's messages newest-first and stop after
-- the page size, so the (session_id, created_at, id) prefix is the access path for both the
-- page itself and the "is there anything older" probe. Before this, messages had only its
-- primary key, so every session-history read was a sequential scan over the whole table.
--
-- The trailing id keeps the index usable for the composite cursor (created_at, id) — plain
-- created_at is not unique (a turn's rows share one timestamp in data written before
-- per-turn timestamps), and the cursor must not be created_at alone.
CREATE INDEX IF NOT EXISTS idx_messages_session_order
    ON public.messages USING btree (session_id, created_at, id);
