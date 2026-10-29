-- Session-list pagination: the sidebar pages one lane with
--   WHERE user_id = ? AND type = ? ORDER BY created_at DESC, id DESC LIMIT n
-- (plus a keyset predicate on (created_at, id) for older pages). Before this, `sessions`
-- had only its primary key, so every list read was a sequential scan of the whole table
-- followed by a sort — with a large account that is the whole session history.
--
-- type sits in the key because every list query filters on the lane (0 = chat,
-- 2 = workspace); leaving it out would make a rare lane scan past every row of the busy
-- one. The trailing id breaks created_at ties: sessions created in one transaction share
-- a timestamp, so the timestamp alone is not a cursor.
CREATE INDEX IF NOT EXISTS idx_sessions_user_type_created
    ON public.sessions USING btree (user_id, type, created_at DESC, id DESC);
