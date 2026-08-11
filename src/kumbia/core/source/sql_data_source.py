"""
   The SQL counterpart to JsonTableSource: point it at a sqlite database and
   run a SELECT, and it produces a DataTable, inferring columns from the
   cursor's own description. sqlite3 only, on purpose — stdlib, zero extra
   dependency, matching the only driver the Kotlin app actually shipped.
"""
from __future__ import annotations

import asyncio
import sqlite3

from kumbia.core.domain.data_table import DataColumn, DataRow, DataTable
from kumbia.core.source.source_query import SourceQuery


class SqlDataSource:
    async def fetch(self, query: SourceQuery) -> DataTable:
        return await asyncio.to_thread(self._fetch_sync, query)

    def _fetch_sync(self, query: SourceQuery) -> DataTable:
        statement = query.statement
        assert statement is not None, "SqlDataSource requires query.statement"

        with sqlite3.connect(query.address) as connection:
            cursor = connection.execute(statement)
            keys = [description[0] for description in cursor.description]
            columns = [DataColumn(key=k) for k in keys]

            rows = [
                DataRow({k: ("" if v is None else str(v)) for k, v in zip(keys, record)})
                for record in cursor.fetchall()
            ]
            return DataTable(columns=columns, rows=rows)