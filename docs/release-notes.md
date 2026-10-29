# Release Notes

Operational notes for changes that alter a deployed interface or need a coordinated
rollout. Product/design detail lives in [architecture.md](architecture.md).

## Sessions API pagination

`GET /sessions` and `GET /sessions/{id}` return one page by default.

| Endpoint | Default page | Maximum | Older pages | Response fields |
|---|---|---|---|---|
| `GET /sessions` | 50 | 200 | `?before=<next_before>` | `has_more`, `next_before` |
| `GET /sessions/{id}` | 50 | 200 | `?before=<next_before>` | `has_more`, `next_before`, `first_message` |

- Ordering is `(created_at DESC, id DESC)` for the session list and `(created_at, turn_rank,
  id)` for the transcript. Cursors are opaque; clients must pass `next_before` back verbatim.
- `GET /sessions?ids=a,b,c` is the pinned-session batch: at most 50 ids, owner-scoped, and it
  ignores paging (`has_more: false`, `next_before: null`).
- `GET /sessions/{id}` still carries the session's title, type, meta and compaction checkpoint
  on every page. `first_message` is filled only for untitled sessions, so a paged open can
  still name the conversation after its opening turn.
- The server-side `load_session_detail(..., limit=None)` keeps returning every message; the
  worker import path and the agent kernel are unaffected.

**Callers.** The desktop renderer is the only in-repo client and every call site was updated:
the Chat sidebar, the WorkSpace picker, the pinned batch, the live refresh, the title probe and
the resume/history path all pass an explicit page size and page with `before`. No service,
worker or script calls these routes.

**Compatibility for external callers.** There is no legacy mode: the default response is the
paged shape. A caller that previously expected the whole list now receives the newest 50 rows
and must page while `has_more` is true, or request a larger `limit` (capped at 200). A caller
that ignores the new fields silently shows a truncated list or transcript.

## Index migrations 0032 / 0033

Both are additive indexes; no column, constraint or data change.

| Migration | Index | Table |
|---|---|---|
| 0032 | `idx_messages_session_order (session_id, created_at, id)` | `public.messages` |
| 0033 | `idx_sessions_user_type_created (user_id, type, created_at DESC, id DESC)` | `public.sessions` |

The migration runner applies each file in one transaction, so both statements are plain
`CREATE INDEX IF NOT EXISTS`. Measured on PostgreSQL 16.14 with 500,000 message rows (149 MB)
and 200,000 session rows (26 MB), while a second connection inserted continuously:

| Build | Build time | Longest concurrent write |
|---|---|---|
| messages, blocking | 1.2 s | 1.2 s |
| messages, `CONCURRENTLY` | 1.2 s | 0.3 s |
| sessions, blocking | 0.3 s | 0.3 s |
| sessions, `CONCURRENTLY` | 0.3 s | 0.01 s |

A blocking build holds a `SHARE` lock: reads continue, but every `INSERT` / `UPDATE` /
`DELETE` on that table waits for the whole build. Build time scales with table size, so on a
large deployment choose one of:

1. **Pre-build online before deploying** (recommended once `messages` is above roughly one to
   two million rows):

   ```sql
   CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_messages_session_order
       ON public.messages (session_id, created_at, id);
   CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sessions_user_type_created
       ON public.sessions (user_id, type, created_at DESC, id DESC);
   ```

   The migration's `IF NOT EXISTS` then makes both statements no-ops.

   **Verify the build actually completed before deploying.** An interrupted
   `CREATE INDEX CONCURRENTLY` leaves the index behind under the same name with
   `indisvalid = false`. The planner ignores an invalid index, and the migration's
   `IF NOT EXISTS` only checks the name — so it skips the build and the deployment looks
   successful while the index does nothing. Check both indexes first:

   ```sql
   SELECT c.relname, i.indisvalid, i.indisready
   FROM pg_index i
   JOIN pg_class c ON c.oid = i.indexrelid
   WHERE c.relname IN ('idx_messages_session_order', 'idx_sessions_user_type_created');
   ```

   A missing row is fine (the migration will build it). A row with `indisvalid = false`
   **must be dropped and rebuilt before deploying**; `IF NOT EXISTS` will not repair it:

   ```sql
   DROP INDEX CONCURRENTLY IF EXISTS public.idx_messages_session_order;
   CREATE INDEX CONCURRENTLY idx_messages_session_order
       ON public.messages (session_id, created_at, id);

   DROP INDEX CONCURRENTLY IF EXISTS public.idx_sessions_user_type_created;
   CREATE INDEX CONCURRENTLY idx_sessions_user_type_created
       ON public.sessions (user_id, type, created_at DESC, id DESC);
   ```

   Re-run the check until every existing row reports `indisvalid = true` (`indisready =
   false` also means the build has not finished). `DROP INDEX CONCURRENTLY` and
   `CREATE INDEX CONCURRENTLY` cannot run inside a transaction block, so run them on a
   plain connection. If the rebuild is interrupted again, drop and repeat.

2. **Deploy in a maintenance window** and accept that writes to each table wait for the build.

Migrations are append-only: 0032 and 0033 must not be edited once applied.
