"""
   Deterministic shaping of Elasticsearch aggregation results into typed
   report data — no LLM involved.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from statistics import mean, pstdev

from kumbia.core.domain.data_table import DataTable


@dataclass(frozen=True)
class TopTalker:
    host_ip: str
    doc_count: int
    unique_dns_names: int
    unique_destinations: int = 0


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
    other_record_count: int = 0  # records in the window that aren't connections (no host field)


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
            unique_destinations=_to_int(row.value("unique_destinations")),
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
    values = field_values(values_table)
    distinct = single_count(distinct_table, "distinct") if distinct_table.rows else len(values)
    return FieldBreakdown(
        alias=alias,
        field_name=field_name,
        distinct_count=distinct,
        values=values,
    )


@dataclass(frozen=True)
class HostProfile:
    """
       One host's traffic in a window, from both ends: what it sent (as
       source) and what it received (as destination), plus the top values
       of each configured breakdown among the events it sent.
    """
    host_ip: str
    events_as_source: int
    events_as_destination: int
    top_destinations: list[FieldValue]
    top_sources: list[FieldValue]
    breakdowns: dict[str, list[FieldValue]]


def field_values(table: DataTable) -> list[FieldValue]:
    """A terms aggregation's buckets as (value, count), most frequent first."""
    values = [FieldValue(value=row.value("key"), doc_count=_to_int(row.value("doc_count"))) for row in table.rows]
    return sorted(values, key=lambda v: v.doc_count, reverse=True)


def single_count(table: DataTable, column: str = "events") -> int:
    """The value of a single-row, non-bucket aggregation (value_count, cardinality)."""
    return _to_int(table.rows[0].value(column)) if table.rows else 0


@dataclass(frozen=True)
class MetricStats:
    count: int
    total: float | None
    mean: float | None
    low: float | None
    high: float | None
    p95: float | None


@dataclass(frozen=True)
class MetricHost:
    host_ip: str
    records: int
    total: float | None
    mean: float | None


@dataclass(frozen=True)
class MetricBucket:
    timestamp: str
    records: int
    total: float | None
    mean: float | None


@dataclass(frozen=True)
class MetricReport:
    alias: str
    field_name: str
    interval: str
    stats: MetricStats
    top_by_total: list[MetricHost]
    top_by_mean: list[MetricHost]
    buckets: list[MetricBucket]

    @property
    def peak_total_bucket(self) -> MetricBucket | None:
        return max((b for b in self.buckets if b.total is not None), key=lambda b: b.total, default=None)

    @property
    def peak_mean_bucket(self) -> MetricBucket | None:
        return max((b for b in self.buckets if b.mean is not None), key=lambda b: b.mean, default=None)

    @property
    def low_mean_bucket(self) -> MetricBucket | None:
        return min((b for b in self.buckets if b.mean is not None), key=lambda b: b.mean, default=None)


def metric_stats(table: DataTable) -> MetricStats:
    """The single-row stats aggregation; the percentiles cell arrives as JSON ({"values": {"95.0": x}})."""
    row = table.rows[0] if table.rows else None
    cell = (lambda name: row.value(name) if row is not None else "")
    p95 = None
    try:
        p95 = _to_float(next(iter(json.loads(cell("p95")).get("values", {}).values()), None))
    except (ValueError, TypeError, AttributeError):
        p95 = None
    return MetricStats(
        count=_to_int(cell("n")),
        total=_to_float(cell("total")),
        mean=_to_float(cell("mean")),
        low=_to_float(cell("low")),
        high=_to_float(cell("high")),
        p95=p95,
    )


def metric_hosts(table: DataTable) -> list[MetricHost]:
    return [
        MetricHost(
            host_ip=row.value("key"),
            records=_to_int(row.value("doc_count")),
            total=_to_float(row.value("total")),
            mean=_to_float(row.value("mean")),
        )
        for row in table.rows
    ]


def metric_buckets(table: DataTable) -> list[MetricBucket]:
    return [
        MetricBucket(
            timestamp=_epoch_millis_to_iso(_to_int(row.value("key"))),
            records=_to_int(row.value("doc_count")),
            total=_to_float(row.value("total")) if _to_int(row.value("doc_count")) else None,
            mean=_to_float(row.value("mean")),
        )
        for row in table.rows
    ]


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


def _to_float(value: object) -> float | None:
    """A pre-stringified metric cell as a float; '' / 'None' / non-numeric (an empty window's avg) -> None."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _epoch_millis_to_iso(epoch_millis: int) -> str:
    return (
        datetime.fromtimestamp(epoch_millis / 1000, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )
