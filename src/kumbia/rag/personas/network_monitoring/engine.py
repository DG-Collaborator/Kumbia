"""
   Deterministic shaping of Elasticsearch aggregation results into typed
   report data — no LLM involved.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from kumbia.core.domain.data_table import DataTable


@dataclass(frozen=True)
class TopTalker:
    host_ip: str
    doc_count: int
    unique_dns_names: int


@dataclass(frozen=True)
class VolumeBucket:
    timestamp: str
    doc_count: int


@dataclass(frozen=True)
class Summary:
    top_talkers: list[TopTalker]
    volume: list[VolumeBucket]
    total_doc_count: int
    busiest_bucket: VolumeBucket | None


def top_talkers(table: DataTable) -> list[TopTalker]:
    talkers = [
        TopTalker(
            host_ip=row.value("key"),
            doc_count=_to_int(row.value("doc_count")),
            unique_dns_names=_to_int(row.value("unique_dns_names")),
        )
        for row in table.rows
    ]
    return sorted(talkers, key=lambda t: t.doc_count, reverse=True)


def volume_over_time(table: DataTable) -> list[VolumeBucket]:
    """
       date_histogram buckets carry their key as raw epoch millis
       (ElasticsearchDataSource intentionally drops key_as_string since it only
       applies to time buckets, not terms buckets), so it's converted back to
       an ISO timestamp here rather than special-casing the generic DataSource.
    """
    return [
        VolumeBucket(
            timestamp=_epoch_millis_to_iso(_to_int(row.value("key"))),
            doc_count=_to_int(row.value("doc_count")),
        )
        for row in table.rows
    ]


def summarize(top_talkers_table: DataTable, volume_table: DataTable) -> Summary:
    talkers = top_talkers(top_talkers_table)
    volume = volume_over_time(volume_table)
    return Summary(
        top_talkers=talkers,
        volume=volume,
        total_doc_count=sum(bucket.doc_count for bucket in volume),
        busiest_bucket=max(volume, key=lambda bucket: bucket.doc_count, default=None),
    )


def _to_int(value: str) -> int:
    """
       Cell values are pre-stringified text (DataTable's contract). Tolerates
       both plain integers ("1420") and stringified floats ("11.0", which
       cardinality aggregations can return), defaulting to 0 on anything else —
       one shared parser instead of the two separate int/float paths the Kotlin
       original used for doc_count vs. unique_dns_names.
    """
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _epoch_millis_to_iso(epoch_millis: int) -> str:
    return (
        datetime.fromtimestamp(epoch_millis / 1000, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )
