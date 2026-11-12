"""
     Reads/writes the collector's on-disk store of correlated telemetry: one
     CanonicalRecord per line as JSON. Plain functions over a Path, same shape
     as rag/memory_store.py — no knowledge of who owns the file or why.

     Only fields that have a value are written, so a line holds exactly what
     was measured in that bucket (no "cpu_utilization": null padding for a
     stream that had nothing); load_records restores omitted fields to their
     None defaults, so readers of CanonicalRecord never notice the difference.
     Extra readings are written flat, beside the named fields (the engine
     guarantees their names never collide), so a line stays one flat object
     and any generic reader sees every reading as its own column.

     A stand-in for the time-series store the FBA architecture calls for:
     fine for one process with one writer (the collector), which is today's
     setup. A second writer process, or concurrent writers per tenant, is the
     point where this graduates to a real store (Phase 6), not something to
     stretch JSONL to cover.
"""
from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path

from kumbia.core.domain.data_table import DataColumn, DataRow, DataTable
from kumbia.telemetry.canonical_record import CanonicalRecord

_FIELD_NAMES = frozenset(f.name for f in fields(CanonicalRecord)) - {"extra_readings"}


def append_records(path: Path, records: list[CanonicalRecord]) -> None:
    """
       Appends every record in one write call, so a batch from one collection
       cycle lands on disk together rather than line by line.
    """
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    batch = "".join(json.dumps(flatten_record(record)) + "\n" for record in records)
    with path.open("a", encoding="utf-8") as f:
        f.write(batch)


def load_records(path: Path) -> list[CanonicalRecord]:
    """
       Every stored record, oldest first. Unknown keys holding a number are
       extra readings; other unknown keys are ignored and missing ones take
       their defaults, so records written before a field was added (or after
       one was dropped) still load.
    """
    if not path.exists():
        return []

    records: list[CanonicalRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            known = {k: v for k, v in data.items() if k in _FIELD_NAMES}
            extras = {
                k: float(v) for k, v in data.items()
                if k not in _FIELD_NAMES and isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            records.append(CanonicalRecord(**known, extra_readings=extras))
    return records


def latest_timestamp(path: Path) -> str | None:
    """
       The newest bucket timestamp stored, or None for an empty/missing
       store. ISO-8601 UTC strings sort in time order, so max() on the text
       is max() in time. Reads the whole file — the collector calls this once
       at start and tracks progress in memory after that.
    """
    return max((record.timestamp for record in load_records(path)), default=None)


def flatten_record(record: CanonicalRecord) -> dict:
    """
       The record as one flat dict: named fields that have a value, then its
       extra readings. timestamp/asset_id/source_system are never None.
    """
    flat = {name: value for name, value in asdict(record).items()
            if name != "extra_readings" and value is not None}
    flat.update(record.extra_readings)
    return flat


def records_to_table(records: list[CanonicalRecord]) -> DataTable:
    """
       Records as a generic DataTable, one row per record and one column per
       field or extra reading that any record has — the same shape reading
       the store through JsonlTableSource gives, without the file.
    """
    if not records:
        return DataTable.EMPTY
    flat = [flatten_record(record) for record in records]
    keys: dict[str, None] = {}
    for row in flat:
        keys.update(dict.fromkeys(row))
    return DataTable(
        columns=[DataColumn(key=key) for key in keys],
        rows=[DataRow({key: ("" if row.get(key) is None else str(row[key])) for key in keys}) for row in flat],
    )
