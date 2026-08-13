"""Web search, as a LangChain tool.

The benchmark scenarios read a fixed corpus so runs are reproducible. The
playground reads the live web instead — same memory layer, unscripted input,
which is the point: it shows the middleware working on material nobody
authored for it.

DuckDuckGo needs no API key, so the playground runs with nothing but Ollama
and Postgres, exactly like the benchmarks.
"""

from __future__ import annotations

from langchain_core.tools import tool

_MAX_RESULTS = 5
_SNIPPET = 400


@tool
def web_search(query: str) -> str:
    """Search the web. Use this to look up facts you do not already know."""
    try:
        from ddgs import DDGS

        hits = list(DDGS().text(query, max_results=_MAX_RESULTS))
    except Exception as e:  # noqa: BLE001 — a dead network must not kill the turn
        return f"Search failed: {type(e).__name__}: {e}"
    if not hits:
        return f"No results for {query!r}."
    return "\n\n".join(
        f"[{i}] {h.get('title', '')}\n{h.get('href', '')}\n{(h.get('body') or '')[:_SNIPPET]}"
        for i, h in enumerate(hits, 1)
    )
