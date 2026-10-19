"""
      Deterministic join of any number of telemetry streams (power, compute,
      network, ...) into CanonicalRecords, keyed by asset (via AssetRegistry)
      and a rounded time bucket. No LLM involved, no derived ratios yet — this
      proves the correlation itself: pairing readings from several systems that
      each name the same physical asset differently. An asset+bucket with a
      reading from only some streams still produces a record, with the other
      streams' fields left None.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime, timezone

from kumbia.core.domain.data_table import DataTable
from kumbia.telemetry.asset_registry import AssetRegistry
from kumbia.telemetry.canonical_record import CanonicalRecord

"""
    CanonicalRecord's numeric reading fields — the only valid targets for a
    SourceMapping's metric_columns. Derived from the record itself (its
    annotations are strings under `from __future__ import annotations`), so a
    reading field added there is automatically joinable here with no edit.
"""
_READING_FIELDS = frozenset(f.name for f in fields(CanonicalRecord) if f.type == "float | None")


@dataclass(frozen=True)
class SourceMapping:
    """
       Describes how to read one telemetry source's generic DataTable:
       which column holds the raw identity (and what identity_type that is,
       for AssetRegistry.resolve), which column holds the timestamp, and
       which columns carry which CanonicalRecord reading fields.
    """
    identity_type: str
    identity_column: str
    timestamp_column: str
    metric_columns: dict[str, str]  # DataTable column key -> CanonicalRecord field name
    source_system: str


def correlate_streams(
        streams: list[tuple[DataTable, SourceMapping]],
        registry: AssetRegistry,
        bucket_minutes: int = 5,
) -> list[CanonicalRecord]:
    """
       Joins every stream's readings by (resolved asset_id, time bucket).
       Every bucket that has a reading from any stream produces exactly one
       CanonicalRecord; fields from a stream with no reading in that bucket
       are left None rather than dropping the record entirely.
    """
    for _, mapping in streams:
        _validate(mapping)

    per_stream = [_extract(table, mapping, registry, bucket_minutes) for table, mapping in streams]

    records: list[CanonicalRecord] = []
    for key in sorted(set().union(*per_stream)):
        asset_id, bucket = key
        source_systems: set[str] = set()
        readings: dict[str, float] = {}
        for entries in per_stream:
            entry = entries.get(key)
            if entry is None:
                continue
            source_systems |= entry["source_systems"]
            readings.update({name: value for name, value in entry.items() if name != "source_systems"})

        records.append(CanonicalRecord(
            timestamp=bucket,
            asset_id=asset_id,
            source_system="+".join(sorted(source_systems)),
            **readings,
        ))
    return records





def _validate(mapping: SourceMapping) -> None:
    """
       Fails loudly on a metric_columns target that isn't a CanonicalRecord
       reading field — a typo'd mapping would otherwise silently drop that
       stream's readings (or crash later inside CanonicalRecord(**readings)).
    """
    unknown = set(mapping.metric_columns.values()) - _READING_FIELDS
    if unknown:
        raise ValueError(
            f"SourceMapping for '{mapping.source_system}' targets unknown reading field(s) "
            f"{sorted(unknown)}; valid fields are {sorted(_READING_FIELDS)}."
        )


def _extract(
        table: DataTable,
        mapping: SourceMapping,
        registry: AssetRegistry,
        bucket_minutes: int,
) -> dict[tuple[str, str], dict]:
    """
       One source's rows, resolved to (asset_id, bucket) and reduced to
       whichever reading fields that row's metric_columns provide. Rows
       whose identity doesn't resolve, or that have no timestamp, are
       dropped — there's no asset to attach them to.
    """
    entries: dict[tuple[str, str], dict] = {}
    for row in table.rows:
        asset_id = registry.resolve(mapping.identity_type, row.value(mapping.identity_column))
        if asset_id is None:
            continue

        raw_timestamp = row.value(mapping.timestamp_column)
        if not raw_timestamp:
            continue
        bucket = _floor_to_bucket(raw_timestamp, bucket_minutes)

        key = (asset_id, bucket)
        entry = entries.setdefault(key, {"source_systems": set()})
        entry["source_systems"].add(mapping.source_system)
        for column_key, field_name in mapping.metric_columns.items():
            value = _to_float(row.value(column_key))
            if value is not None:
                entry[field_name] = value

    return entries


def _floor_to_bucket(timestamp_iso: str, bucket_minutes: int) -> str:
    dt = datetime.fromisoformat(timestamp_iso.replace("Z", "+00:00"))
    epoch_minutes = int(dt.timestamp() // 60)
    bucket_start = (epoch_minutes // bucket_minutes) * bucket_minutes
    bucket_dt = datetime.fromtimestamp(bucket_start * 60, tz=timezone.utc)
    return bucket_dt.isoformat().replace("+00:00", "Z")


def _to_float(value: str) -> float | None:
    """Cell values are pre-stringified text; '' (a missing cell) and anything non-numeric become None."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
