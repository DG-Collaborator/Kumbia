"""
     Point at a local JSONL file (one JSON object per line) and produce a
     DataTable, inferring columns from the records themselves — the file-based
     sibling of JsonTableSource (JSON over HTTP) and SqlDataSource (sqlite).
     Models no specific schema: the union of every record's keys becomes the
     columns, in first-seen order.

     query.address is the file path. query.statement is optional JSON options:
     {"last": N} keeps only the final N records, which for an append-only file
     means the N most recent. A missing file is an empty table, not an error —
     an append-only log that hasn't been written yet has simply recorded nothing.
  """
from __future__ import annotations

import asyncio
import json
from collections import deque
from pathlib import Path
from typing import Any

from kumbia.core.domain.data_table import DataColumn, DataRow, DataTable
from kumbia.core.source.source_query import SourceQuery


class JsonlTableSource:
    async def fetch(self, query: SourceQuery) -> DataTable:
        return await asyncio.to_thread(self._fetch_sync, query)

    def _fetch_sync(self, query: SourceQuery) -> DataTable:
        path = Path(query.address)
        if not path.exists():
            return DataTable.EMPTY

        last = _parse_last(query.statement)
        records: deque[dict] = deque(maxlen=last)  # maxlen=None keeps everything
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if isinstance(record, dict):
                    records.append(record)

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


def _parse_last(statement: str | None) -> int | None:
    if not statement:
        return None
    options = json.loads(statement)
    last = options.get("last")
    if last is None:
        return None
    if not isinstance(last, int) or isinstance(last, bool) or last < 1:
        raise ValueError(f'JsonlTableSource "last" must be a positive integer, got: {last!r}')
    return last


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value)