"""``rag_search``: search the learning-material corpus for relevant chunks."""
from __future__ import annotations

import json
import logging

from agent import Context, ToolExecution, ToolOutput, ToolRuntime, define_tool, text_block
from agent.tools import search_reuse
from agent.tools.embedding_source import embedding_source
from core.infrastructure.repositories import SqlDomainRepository
from core.infrastructure.request_context import get_request_user_id, get_rag_fast_lane

logger = logging.getLogger(__name__)

# Returned as a *successful* tool result when the retrieval stack is down (e.g. the
# embedding service returns 503). A normal result lets the model answer from knowledge
# in this round; a raised error would surface as a tool error and burn another round
# on a pointless retry.
_UNAVAILABLE = [{"notice": "学习资料库检索服务暂时不可用,请直接基于你的知识回答,不要重试检索。"}]


def register(runtime: ToolRuntime, ctx: Context, llm) -> None:
    async def _resolve_domain_id(name: str, user_id) -> str:
        """Resolve a domain NAME to its real ``domains.id`` within the caller's
        visible set (public + own), the same set ``list_domains`` exposes. The
        model supplies a plain NAME; it can NEVER be a fabricated UUID.

        * exactly 1 match -> its id;
        * 0 matches       -> ``preflight:`` not found (provably no scope filter);
        * >1 matches      -> ``preflight:`` ambiguous — never pick one, the
          executor escalates so the Agent can clarify.
        """
        key = name.strip().lower()
        async with ctx.resolve("session_factory")() as session:
            domains = await SqlDomainRepository(session).list_all(user_id)
        matches = [d for d in domains if (d.name or "").strip().lower() == key]
        if not matches:
            raise ValueError(f"preflight: rag domain {name!r} not found")
        if len(matches) > 1:
            raise ValueError(
                f"preflight: {len(matches)} domains are named {name!r}; "
                "ask the user which one"
            )
        return str(matches[0].id)

    async def rag_search(args: dict, exec: ToolExecution) -> list[dict]:
        retriever = ctx.resolve("retrieval")
        # Tenant isolation: bind the current request's user so retrieval only sees their
        # assets (owner / workspace / ACL). A guest (None) sees public-link assets only.
        user_id = get_request_user_id()
        filters = {"user_id": user_id}
        # Optional domain scoping: the model names a domain; the BUSINESS layer resolves
        # that name to a real assets.domain_id (never a model-fabricated UUID).
        if args.get("domain"):
            filters["domain_id"] = await _resolve_domain_id(str(args["domain"]), user_id)
        query = args.get("query", "")
        top_k = args.get("top_k", 5)
        # Same-turn reuse: an identical call after a successful one returns the earlier
        # chunks without touching the retrieval stack. Different query/top_k/domain is a
        # different search and runs normally. A degraded notice is never cached below.
        cache_key = search_reuse.key("rag_search", query, top_k, filters.get("domain_id"))
        cached = search_reuse.lookup(cache_key)
        if cached is not None:
            exec.trace["reused"] = True
            exec.trace["top_k"] = top_k
            return cached
        try:
            # Lane fork (compile-time: /chat pins it at request entry, worker/admin
            # never do). Chat = fast lane (no LLM rewrite/CRAG, ~0.5s); research agent
            # & every other context = the original full retrieve(). hasattr keeps
            # test doubles / the gRPC client (no chat lane yet) on the safe full path.
            with embedding_source(
                "rag_search", call_id=exec.call_id, call_index=exec.call_index
            ):
                if get_rag_fast_lane() and hasattr(retriever, "retrieve_chat"):
                    hits = await retriever.retrieve_chat(query, top_k, filters)
                else:
                    hits = await retriever.retrieve(query, top_k, filters)
        except Exception as exc:  # noqa: BLE001 - degrade to knowledge-only, don't retry
            logger.warning("rag_search degraded (retrieval unavailable): %s", exc)
            return _UNAVAILABLE
        search_reuse.record(cache_key, hits)
        exec.trace["top_k"] = top_k
        if filters.get("domain_id"):
            exec.trace["scope"] = "domain"
        return hits

    runtime.register(
        define_tool(
            name="rag_search",
            description="Search the user's OWN knowledge base — their uploaded documents, "
            "cloud drive files and internal material — for chunks relevant to the query. "
            "Returns matching chunks. It does not hold general knowledge and does not "
            "reach the web: do not use it for general explanations, and do not call it "
            "before every answer.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "top_k": {"type": "integer", "description": "Number of results."},
                    "domain": {
                        "type": "string",
                        "description": "Optional domain NAME to scope the search to "
                        "(resolved to a domain id by the tool).",
                    },
                },
                "required": ["query"],
            },
            output=ToolOutput(
                schema={"type": "array"},
                render=lambda args, value: [
                    text_block(json.dumps(value, ensure_ascii=False, default=str))
                ],
            ),
            execute=rag_search,
        )
    )
