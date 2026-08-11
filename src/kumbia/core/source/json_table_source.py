"""The universal source: point it at any URL that returns JSON and it
  produces a DataTable, inferring columns from the data itself. Models no
  specific schema — whatever objects the endpoint returns become rows, the
  union of their keys becomes the columns.
  """
from __future__ import annotations

from typing import Any

import httpx

from kumbia.core.domain.data_table import DataColumn, DataRow, DataTable
from kumbia.core.source.source_query import SourceQuery

_PREFERRED_KEYS = ("features", "data", "results", "items", "records", "rows")


class JsonTableSource:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def fetch(self, query: SourceQuery) -> DataTable:
        response = await self._client.get(query.address)
        response.raise_for_status()
        root = response.json()

        records = [_unwrap(r) for r in _extract_records(root)]
        if not records:
            return DataTable.EMPTY

        # columns = union of keys across all records, first-seen order
        column_keys: dict[str, None] = {}
        for record in records:
            for key in record:
                column_keys[key] = None
        columns = [DataColumn(key=k) for k in column_keys]

        rows = [
            DataRow({k: _stringify(record.get(k)) for k in column_keys})
            for record in records
        ]
        return DataTable(columns=columns, rows=rows)


def _extract_records(root: Any) -> list[dict]:
    """Handles: a bare array, an object wrapping a list under a well-known
    key, the largest array of objects as a fallback, or a single object
    treated as one row."""
    if isinstance(root, list):
        return [r for r in root if isinstance(r, dict)]

    if isinstance(root, dict):
        for key in _PREFERRED_KEYS:
            candidate = root.get(key)
            if isinstance(candidate, list) and any(isinstance(x, dict) for x in candidate):
                return [r for r in candidate if isinstance(r, dict)]

        # fallback: the largest array of objects — record arrays are almost
        # always bigger than incidental metadata arrays
        arrays = [v for v in root.values() if isinstance(v, list) and any(isinstance(x, dict) for x in v)]
        if arrays:
            largest = max(arrays, key=len)
            return [r for r in largest if isinstance(r, dict)]

        return [root]

    return []


def _unwrap(record: dict) -> dict:
    """
       Promotes ArcGIS 'attributes' or GeoJSON 'properties' wrappers so their
       real fields surface as columns instead of one blob column.
    """
    attributes = record.get("attributes")
    if isinstance(attributes, dict):
        return _merge_siblings(record, "attributes", attributes)

    looks_like_geojson = record.get("type") == "Feature" or "geometry" in record
    if looks_like_geojson:
        properties = record.get("properties")
        if isinstance(properties, dict):
            return _merge_siblings(record, "properties", properties)

    return record


def _merge_siblings(record: dict, wrapper_key: str, inner: dict) -> dict:
    siblings = {
        k: v for k, v in record.items()
        if k != wrapper_key and (isinstance(v, (str, int, float, bool)) or v is None)
    }
    return {**siblings, **inner}  # inner fields win on collisions


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    import json
    return json.dumps(value)