"""
   The data-access seam. Callers depend only on this protocol, never on a
   concrete source — swap JSON-over-HTTP for SQL, Elasticsearch, or anything
   else and nothing above this line changes. Takes a SourceQuery, not a raw URL,
   so non-HTTP sources fit the same seam.
"""
from __future__ import annotations

from typing import Protocol

from kumbia.core.domain.data_table import DataTable
from kumbia.core.source.source_query import SourceQuery

class DataSource(Protocol):
    async def fetch(self, query: SourceQuery) -> DataTable: ...

