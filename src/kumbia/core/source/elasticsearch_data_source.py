"""Point at an Elasticsearch _search endpoint and produce a DataTable. Fully
 generic — handles both aggregation responses (buckets) and plain hit
 responses, with no knowledge of any specific index or persona.
 """
from __future__ import annotations

from typing import Any

import httpx

from kumbia.core.domain.data_table import DataColumn, DataRow, DataTable
from kumbia.core.source.source_query import SourceQuery


class ElasticsearchDataSource:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def fetch(self, query: SourceQuery) -> DataTable:
        auth = None
        if query.credentials is not None:
            auth = (query.credentials.username, query.credentials.password)

        response = await self._client.post(
            query.address,
            content=query.statement or "{}",
            headers={"Content-Type": "application/json"},
            auth=auth,
        )
        response.raise_for_status()
        root = response.json()

        if not isinstance(root, dict):
            return DataTable.EMPTY

        aggregations = root.get("aggregations")
        if isinstance(aggregations, dict):
            return _table_from_aggregations(aggregations)
        return _table_from_hits(root)


def _table_from_aggregations(aggregations: dict) -> DataTable:
    bucket_agg = next(
        ((name, value) for name, value in aggregations.items()
         if isinstance(value, dict) and isinstance(value.get("buckets"), list)),
        None,
    )

    if bucket_agg is None:
        cells = {name: _stringify(value.get("value", value) if isinstance(value, dict) else value)
                 for name, value in aggregations.items()}
        return DataTable(columns=[DataColumn(key=k) for k in cells], rows=[DataRow(cells)])

    _, bucket_value = bucket_agg
    buckets = bucket_value["buckets"]

    column_keys: dict[str, None] = {"key": None, "doc_count": None}
    rows: list[DataRow] = []
    for bucket in buckets:
        if not isinstance(bucket, dict):
            continue
        cells = {
            "key": _stringify(bucket.get("key")),
            "doc_count": _stringify(bucket.get("doc_count")),
        }
        for name, value in bucket.items():
            if name in ("key", "doc_count", "key_as_string"):
                continue
            metric_value = value.get("value") if isinstance(value, dict) else None
            cells[name] = _stringify(metric_value) if metric_value is not None else ""
            column_keys[name] = None
        rows.append(DataRow(cells))

    return DataTable(columns=[DataColumn(key=k) for k in column_keys], rows=rows)


def _table_from_hits(root: dict) -> DataTable:
    hits = ((root.get("hits") or {}).get("hits")) or []
    sources = [h["_source"] for h in hits if isinstance(h, dict) and isinstance(h.get("_source"), dict)]
    if not sources:
        return DataTable.EMPTY

    column_keys: dict[str, None] = {}
    for source in sources:
        for key in source:
            column_keys[key] = None

    rows = [
        DataRow({k: _stringify(source.get(k)) for k in column_keys})
        for source in sources
    ]
    return DataTable(columns=[DataColumn(key=k) for k in column_keys], rows=rows)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    import json
    return json.dumps(value)