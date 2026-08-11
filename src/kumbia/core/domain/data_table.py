"""
   Universal tabular domain model. Every DataSource produces a DataTable, and
   everything downstream (RAG ingestion, MCP tool rendering) consumes whatever
   columns it finds — this is what makes the backend pluggable.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DataColumn:
    """
       key is the raw field name used to look up cell values; label is what a
       UI shows. Same by default so a header can be improved without breaking
       lookups.
    """
    key: str
    label: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            object.__setattr__(self, "label", self.key)


@dataclass(frozen=True)
class DataRow:
    """
       Addressed by column key. Values are pre-stringified: the source layer
       flattens numbers/booleans/None into display text so nothing downstream
       reasons about raw types. A missing cell resolves to ''.
    """
    cells: dict[str, str]

    def value(self, key: str) -> str:
        return self.cells.get(key, "")


@dataclass(frozen=True)
class DataTable:
    columns: list[DataColumn]
    rows: list[DataRow]

    @property
    def is_empty(self) -> bool:
        return len(self.rows) == 0


DataTable.EMPTY = DataTable(columns=[], rows=[])  # typed "nothing yet"