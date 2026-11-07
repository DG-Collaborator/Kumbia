"""
   Deterministic shaping of Elasticsearch aggregation results into typed
   report data — no LLM involved.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from statistics import mean, pstdev

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
    anomalies: list[VolumeBucket] = field(default_factory=list)
    interval: str = "1h"
    newest_event: str | None = None  # only looked up when the window is empty


@dataclass(frozen=True)
class FieldValue:
    value: str
    doc_count: int


@dataclass(frozen=True)
class FieldBreakdown:
    alias: str
    field_name: str
    distinct_count: int
    values: list[FieldValue]


ANOMALY_SIGMA = 2.0
_MIN_BUCKETS_FOR_ANOMALIES = 4


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


def volume_anomalies(volume: list[VolumeBucket], sigma: float = ANOMALY_SIGMA) -> list[VolumeBucket]:
    """
       Buckets whose count is more than sigma standard deviations above the
       window's mean — plain statistics, so a "spike" is a computed fact the
       narration can only repeat, never decide. Too few buckets (or a flat
       series) yields none rather than a meaningless flag.
    """
    if len(volume) < _MIN_BUCKETS_FOR_ANOMALIES:
        return []
    counts = [bucket.doc_count for bucket in volume]
    spread = pstdev(counts)
    if spread == 0:
        return []
    threshold = mean(counts) + sigma * spread
    return [bucket for bucket in volume if bucket.doc_count > threshold]


def summarize(top_talkers_table: DataTable, volume_table: DataTable, interval: str = "1h") -> Summary:
    talkers = top_talkers(top_talkers_table)
    volume = volume_over_time(volume_table)
    return Summary(
        top_talkers=talkers,
        volume=volume,
        total_doc_count=sum(bucket.doc_count for bucket in volume),
        busiest_bucket=max(volume, key=lambda bucket: bucket.doc_count, default=None),
        anomalies=volume_anomalies(volume),
        interval=interval,
    )


def field_breakdown(alias: str, field_name: str, values_table: DataTable, distinct_table: DataTable) -> FieldBreakdown:
    values = [
        FieldValue(value=row.value("key"), doc_count=_to_int(row.value("doc_count")))
        for row in values_table.rows
    ]
    distinct = _to_int(distinct_table.rows[0].value("distinct")) if distinct_table.rows else len(values)
    return FieldBreakdown(
        alias=alias,
        field_name=field_name,
        distinct_count=distinct,
        values=sorted(values, key=lambda v: v.doc_count, reverse=True),
    )


def newest_event(latest_table: DataTable) -> str | None:
    """The max-timestamp aggregation's value (epoch millis) as ISO, or None if the scope has no events."""
    if not latest_table.rows:
        return None
    millis = _to_int(latest_table.rows[0].value("newest"))
    return _epoch_millis_to_iso(millis) if millis > 0 else None


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
