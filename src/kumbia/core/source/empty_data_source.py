"""
     A DataSource that is always empty: fetch() ignores the query and returns
     DataTable.EMPTY. For a slot that must hold a DataSource but has no real
     system behind it yet (e.g. a telemetry stream whose backing platform isn't
     reachable), so the caller can still be constructed and run end to end —
     a stream wired to this simply contributes no rows. Holds no client and no
     configuration, since it never talks to anything.
"""
from __future__ import annotations

from kumbia.core.domain.data_table import DataTable
from kumbia.core.source.source_query import SourceQuery


class EmptyDataSource:
    async def fetch(self, query: SourceQuery) -> DataTable:
        return DataTable.EMPTY