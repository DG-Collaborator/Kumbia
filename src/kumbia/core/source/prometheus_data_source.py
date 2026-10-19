"""
   Point at a Prometheus HTTP API endpoint and produce a DataTable. Selects
   /api/v1/query_range (matrix — multiple samples over time) when the
   statement includes start/end, otherwise /api/v1/query (vector — one point
   in time). No knowledge of any specific metric, persona, or telemetry
   domain — same generality as ElasticsearchDataSource handling both
   aggregation- and hit-shaped responses.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import httpx

from kumbia.core.domain.data_table import DataColumn, DataRow, DataTable
from kumbia.core.source.source_query import SourceQuery


class PrometheusDataSource:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def fetch(self, query: SourceQuery) -> DataTable:
        assert query.statement is not None, "PrometheusDataSource requires query.statement"
        params: dict[str, Any] = json.loads(query.statement)

        auth = None
        if query.credentials is not None:
            auth = (query.credentials.username, query.credentials.password)

        endpoint = "query_range" if "start" in params and "end" in params else "query"
        url = f"{query.address.rstrip('/')}/api/v1/{endpoint}"

        response = await self._client.get(url, params=params, auth=auth)
        response.raise_for_status()
        root = response.json()

        if not isinstance(root, dict) or root.get("status") != "success":
            return DataTable.EMPTY

        data = root.get("data") or {}
        result = data.get("result")
        if not isinstance(result, list) or not result:
            return DataTable.EMPTY

        rows = _matrix_rows(result) if data.get("resultType") == "matrix" else _vector_rows(result)
        if not rows:
            return DataTable.EMPTY

        column_keys: dict[str, None] = {}
        for row in rows:
            for key in row:
                column_keys[key] = None
        columns = [DataColumn(key=k) for k in column_keys]
        return DataTable(
            columns=columns,
            rows=[DataRow({k: row.get(k, "") for k in column_keys}) for row in rows],
        )


def _matrix_rows(result: list[dict]) -> list[dict[str, str]]:
    """One row per (series, sample) pair — a range query returns many samples per series."""
    rows: list[dict[str, str]] = []
    for series in result:
        metric = series.get("metric", {})
        for timestamp, value in series.get("values", []):
            rows.append({**metric, "timestamp": _to_iso(timestamp), "value": str(value)})
    return rows


def _vector_rows(result: list[dict]) -> list[dict[str, str]]:
    """One row per series — an instant query returns exactly one sample each."""
    rows: list[dict[str, str]] = []
    for series in result:
        sample = series.get("value")
        if sample is None:
            continue
        timestamp, value = sample
        rows.append({**series.get("metric", {}), "timestamp": _to_iso(timestamp), "value": str(value)})
    return rows


def _to_iso(epoch_seconds: float) -> str:
    return datetime.fromtimestamp(float(epoch_seconds), tz=timezone.utc).isoformat().replace("+00:00", "Z")