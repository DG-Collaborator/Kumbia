"""
       Reads/writes a persona's on-disk memory file: one JSON object per line
       ({"text", "created_at", "source"}). Kept as plain functions rather than a
       DataSource, since this is local-file glue any persona's ConversationPipeline
       uses directly for its own memory_path/cache_path, not a generic HTTP/DB
       source fetched through the DataSource seam.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from kumbia.core.domain.data_table import DataColumn, DataRow, DataTable

_COLUMNS = [DataColumn(key="text"), DataColumn(key="created_at"), DataColumn(key="source")]


def load_memory_table(path: Path) -> DataTable:
    if not path.exists():
        return DataTable.EMPTY

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            rows.append(DataRow({
                "text": record.get("text", ""),
                "created_at": record.get("created_at", ""),
                "source": record.get("source", ""),
            }))
    return DataTable(columns=_COLUMNS, rows=rows)


def append_memory(path: Path, text: str, source: str = "user") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "text": text,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")