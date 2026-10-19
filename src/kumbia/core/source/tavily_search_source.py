"""
      Point at Tavily's search API and produce a DataTable of results (title,
      url, content snippet). Fully generic — any persona wanting web search
      reuses this, same as JsonTableSource is reused for any generic JSON
      endpoint. query.address is unused since Tavily's endpoint is fixed;
      query.statement carries the actual search text.
"""
from __future__ import annotations

import httpx

from kumbia.core.domain.data_table import DataColumn, DataRow, DataTable
from kumbia.core.source.source_query import SourceQuery

TAVILY_SEARCH_URL = "https://api.tavily.com/search"

_COLUMNS = [DataColumn(key="title"), DataColumn(key="url"), DataColumn(key="content")]


class TavilySearchSource:
    def __init__(self, client: httpx.AsyncClient, api_key: str) -> None:
        self._client = client
        self._api_key = api_key

    async def fetch(self, query: SourceQuery) -> DataTable:
        response = await self._client.post(
            TAVILY_SEARCH_URL,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"query": query.statement, "max_results": 5},
        )
        if response.is_error:
            raise RuntimeError(
                f"Tavily search request failed ({response.status_code}) — "
                "check that TAVILY_API_KEY is set correctly."
            )
        results = response.json().get("results", [])
        rows = [
            DataRow({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
            })
            for r in results
        ]
        return DataTable(columns=_COLUMNS, rows=rows)