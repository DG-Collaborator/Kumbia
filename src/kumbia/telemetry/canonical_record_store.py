"""
     Reads/writes the collector's on-disk store of correlated telemetry: one
     CanonicalRecord per line as JSON. Plain functions over a Path, same shape
     as rag/memory_store.py — no knowledge of who owns the file or why.

     Only fields that have a value are written, so a line holds exactly what
     was measured in that bucket (no "cpu_utilization": null padding for a
     stream that had nothing); load_records restores omitted fields to their
     None defaults, so readers of CanonicalRecord never notice the difference.

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

from kumbia.telemetry.canonical_record import CanonicalRecord

_FIELD_NAMES = frozenset(f.name for f in fields(CanonicalRecord))


def append_records(path: Path, records: list[CanonicalRecord]) -> None:
    """
       Appends every record in one write call, so a batch from one collection
       cycle lands on disk together rather than line by line.
    """
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    batch = "".join(json.dumps(_present_fields(record)) + "\n" for record in records)
    with path.open("a", encoding="utf-8") as f:
        f.write(batch)


def load_records(path: Path) -> list[CanonicalRecord]:
    """
       Every stored record, oldest first. Keys CanonicalRecord doesn't know
       are ignored and missing ones take their defaults, so records written
       before a field was added (or after one was dropped) still load.
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
            records.append(CanonicalRecord(**{k: v for k, v in data.items() if k in _FIELD_NAMES}))
    return records


def latest_timestamp(path: Path) -> str | None:
    """
       The newest bucket timestamp stored, or None for an empty/missing
       store. ISO-8601 UTC strings sort in time order, so max() on the text
       is max() in time. Reads the whole file — the collector calls this once
       at start and tracks progress in memory after that.
    """
    return max((record.timestamp for record in load_records(path)), default=None)


def _present_fields(record: CanonicalRecord) -> dict:
    """The record as a dict without its None fields; timestamp/asset_id/source_system are never None."""
    return {name: value for name, value in asdict(record).items() if value is not None}
