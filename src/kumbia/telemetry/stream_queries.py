"""
      Builds each telemetry stream's SourceQuery for one time window — the
      piece TelemetryCorrelationPipeline deliberately leaves to its caller.
      Pure functions: no I/O, no persona knowledge, no LLM. Every value that
      ends up inside a query string comes from code (a TimeWindow of real
      datetimes, identifier names validated below, a registry's own IPs) —
      never from user or LLM text, same "structural, not instructional" rule
      as power_telemetry_report's fixed SQL. plan_window() turns a number of
      hours (parsed from a question by regex) into a window of real datetimes
      and a bucket size, so on-demand reports reuse these same builders.

      Windows are half-open, [start, end): the collector asks for exactly the
      buckets that have closed since its last run, so consecutive windows
      never overlap and never re-read a reading twice.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from kumbia.core.source.source_query import Credentials, SourceQuery

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_INSTANCE_RE = re.compile(r"^[A-Za-z0-9.:_\-\[\]]+$")
_SQL_VALUE_RE = re.compile(r"^[A-Za-z0-9.:_\-/]+$")

MAX_COMPOSITE_BUCKETS = 1000  # one page of an Elasticsearch composite aggregation
_BUCKET_CHOICES_MINUTES = (1, 5, 15, 30, 60, 180, 360, 720, 1440)
_NETWORK_AGGREGATIONS = frozenset({"sum", "avg", "max"})


@dataclass(frozen=True)
class TimeWindow:
    """[start, end) in UTC. Built from real datetimes, so no caller-supplied text reaches a query."""
    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("TimeWindow needs timezone-aware datetimes.")
        if self.start >= self.end:
            raise ValueError(f"TimeWindow start must be before end, got {self.start} >= {self.end}.")

    @property
    def start_iso(self) -> str:
        return _iso(self.start)

    @property
    def end_iso(self) -> str:
        return _iso(self.end)

    @property
    def minutes(self) -> float:
        return (self.end - self.start).total_seconds() / 60


@dataclass(frozen=True)
class NetworkMetric:
    """One numeric network field to aggregate per bucket: its field name and sum, avg, or max."""
    field: str
    aggregation: str = "sum"

    def __post_init__(self) -> None:
        if self.aggregation not in _NETWORK_AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {sorted(_NETWORK_AGGREGATIONS)}, got {self.aggregation!r}")


@dataclass(frozen=True)
class WindowPlan:
    """A trailing window of whole, closed buckets, and the bucket size it was cut to."""
    hours_back: int
    bucket_minutes: int
    window: TimeWindow

    @property
    def bucket_count(self) -> int:
        return math.ceil(self.window.minutes / self.bucket_minutes)

    def describe(self) -> str:
        return (
            f"Window: last {self.hours_back} hour(s), {self.window.start_iso} to {self.window.end_iso}, "
            f"in {self.bucket_minutes}-minute buckets ({self.bucket_count} buckets)"
        )


def plan_window(
        hours_back: int,
        now: datetime,
        min_bucket_minutes: int = 1,
        series_count: int = 1,
        max_buckets: int = MAX_COMPOSITE_BUCKETS,
) -> WindowPlan:
    """
       The last hours_back hours as whole buckets that have already closed
       (the still-filling bucket is left out, same as the collector), with
       the finest standard bucket size >= min_bucket_minutes that keeps
       series_count * buckets within max_buckets — so a 1-hour question gets
       minute detail and a 30-day one still fits one aggregation page.
    """
    if hours_back < 1:
        raise ValueError("hours_back must be at least 1.")
    minutes = hours_back * 60
    series = max(1, series_count)
    candidates = [b for b in _BUCKET_CHOICES_MINUTES if b >= min_bucket_minutes] or [min_bucket_minutes]
    bucket = next(
        (b for b in candidates if series * math.ceil(minutes / b) <= max_buckets),
        candidates[-1],
    )
    end = _floor(now, bucket)
    return WindowPlan(hours_back, bucket, TimeWindow(end - timedelta(hours=hours_back), end))


def sql_power_buckets(
        db_path: str,
        window: TimeWindow,
        bucket_minutes: int,
        outlets: list[str],
        table: str = "power_readings",
        identity_column: str = "outlet",
        timestamp_column: str = "ts",
        value_column: str = "watts",
) -> SourceQuery:
    """
       Power readings inside the window, reduced to one row per (outlet,
       time bucket): the bucket's start as ISO text in the timestamp column,
       its average watts in the value column, plus <value>_min, <value>_max,
       and readings (how many readings it averaged). Sources that report
       more often than the bucket size are averaged rather than having one
       reading win, and a long window stays a bounded number of rows.

       Reads a sqlite table whose timestamp column holds ISO-8601 UTC text
       ("2026-09-24T14:05:00Z"). That format sorts lexicographically in time
       order, which is what makes a plain string >= / < comparison a correct
       time filter here. Filtered to outlets (the registry's known power
       identities), so unregistered outlets never come back.
    """
    if not outlets:
        raise ValueError("sql_power_buckets needs at least one outlet to filter on.")
    for identifier in (table, identity_column, timestamp_column, value_column):
        _require_identifier(identifier)
    if bucket_minutes < 1:
        raise ValueError("bucket_minutes must be at least 1.")

    seconds = bucket_minutes * 60
    outlet_list = ", ".join(_sql_string_literal(outlet) for outlet in outlets)
    bucket_start = (
        f"strftime('%Y-%m-%dT%H:%M:%SZ', "
        f"(CAST(strftime('%s', {timestamp_column}) AS INTEGER) / {seconds}) * {seconds}, 'unixepoch')"
    )
    statement = (
        f"SELECT {identity_column} AS {identity_column}, {bucket_start} AS {timestamp_column}, "
        f"AVG({value_column}) AS {value_column}, MIN({value_column}) AS {value_column}_min, "
        f"MAX({value_column}) AS {value_column}_max, COUNT({value_column}) AS readings "
        f"FROM {table} "
        f"WHERE {timestamp_column} >= '{window.start_iso}' AND {timestamp_column} < '{window.end_iso}' "
        f"AND {identity_column} IN ({outlet_list}) AND {value_column} IS NOT NULL "
        f"GROUP BY 1, 2 ORDER BY 2, 1"
    )
    return SourceQuery(address=db_path, statement=statement)


def es_network_window(
        base_url: str,
        index_pattern: str,
        window: TimeWindow,
        host_ips: list[str],
        bucket_minutes: int,
        credentials: Credentials | None = None,
        host_field: str = "host.ip",
        timestamp_field: str = "@timestamp",
        max_buckets: int = MAX_COMPOSITE_BUCKETS,
        metrics: dict[str, NetworkMetric] | None = None,
) -> SourceQuery:
    """
       Event counts per (host, time bucket) inside the window, as one
       Elasticsearch composite aggregation. ElasticsearchDataSource turns
       the response into rows with key.host, key.ts, doc_count columns.

       metrics adds one numeric sub-aggregation per entry (column name ->
       field and sum/avg/max), each its own column, e.g. bytes sent per
       bucket beside the event count. Names and fields come from
       configuration, never from a question.

       Filtered to host_ips (the registry's known network identities), so
       unregistered hosts never cost a bucket. The composite's after_key
       isn't followed, so instead of silently truncating, this refuses a
       window that could produce more buckets than one page holds.
    """
    if not host_ips:
        raise ValueError("es_network_window needs at least one host IP to filter on.")

    worst_case_buckets = len(host_ips) * math.ceil(window.minutes / bucket_minutes)
    if worst_case_buckets > max_buckets:
        raise ValueError(
            f"Window could produce {worst_case_buckets} buckets (> {max_buckets}); "
            "use a shorter window or fewer hosts."
        )

    composite: dict = {
        "composite": {
            "size": max_buckets,
            "sources": [
                {"host": {"terms": {"field": host_field}}},
                {"ts": {"date_histogram": {
                    "field": timestamp_field,
                    "fixed_interval": f"{bucket_minutes}m",
                    "format": "strict_date_time_no_millis",
                }}},
            ],
        }
    }
    if metrics:
        for name, metric in metrics.items():
            _require_identifier(name)
        composite["aggs"] = {name: {metric.aggregation: {"field": metric.field}} for name, metric in metrics.items()}

    body = {
        "size": 0,
        "query": {
            "bool": {
                "filter": [
                    {"terms": {host_field: host_ips}},
                    {"range": {timestamp_field: {"gte": window.start_iso, "lt": window.end_iso}}},
                ]
            }
        },
        "aggs": {"by_host_bucket": composite},
    }
    return SourceQuery(
        address=f"{base_url.rstrip('/')}/{index_pattern}/_search",
        statement=json.dumps(body),
        credentials=credentials,
    )


def prometheus_compute_window(
        base_url: str,
        window: TimeWindow,
        instances: list[str],
        bucket_minutes: int,
        credentials: Credentials | None = None,
        identity_label: str = "instance",
        cpu_metric: str = "node_cpu_seconds_total",
        max_points: int = 11000,
) -> SourceQuery:
    """
       Average CPU utilization (0-100) per (instance, time bucket) inside the
       window, as one Prometheus range query. PrometheusDataSource turns the
       matrix response into rows with the series' labels plus timestamp and
       value columns.

       Evaluated once per bucket (step = bucket size) at each bucket's start,
       with the rate looking forward over that bucket (offset -bucket), so a
       value's timestamp floors into the bucket it actually describes, the
       same bucket the power and network readings for that span land in.
       end is pulled back one second to keep the window half-open.

       Filtered to instances (the registry's known compute identities), so
       unregistered targets never come back. Prometheus caps a range query at
       11,000 points per series, so instead of being rejected there, this
       refuses a window that would exceed it.
    """
    if not instances:
        raise ValueError("prometheus_compute_window needs at least one instance to filter on.")
    for identifier in (identity_label, cpu_metric):
        _require_identifier(identifier)

    points = math.ceil(window.minutes / bucket_minutes)
    if points > max_points:
        raise ValueError(
            f"Window would produce {points} points per series (> {max_points}); use a shorter window."
        )

    instance_pattern = "|".join(_promql_regex_literal(instance) for instance in instances)
    rate_range = f"{bucket_minutes}m"
    promql = (
        f"100 * (1 - avg by ({identity_label}) ("
        f'rate({cpu_metric}{{mode="idle",{identity_label}=~"{instance_pattern}"}}[{rate_range}] offset -{rate_range})'
        f"))"
    )
    statement = {
        "query": promql,
        "start": window.start_iso,
        "end": _iso(window.end - timedelta(seconds=1)),
        "step": f"{bucket_minutes * 60}s",
    }
    return SourceQuery(address=base_url.rstrip("/"), statement=json.dumps(statement), credentials=credentials)


def _require_identifier(name: str) -> None:
    """SQL/PromQL identifiers can't be bound as parameters, so they're restricted to plain names instead."""
    if not _IDENTIFIER_RE.fullmatch(name):
        raise ValueError(f"Not a plain SQL identifier: {name!r}")


def _sql_string_literal(value: str) -> str:
    """A registry value (an outlet name) as a quoted SQL string, restricted to plain name characters first."""
    if not _SQL_VALUE_RE.fullmatch(value):
        raise ValueError(f"Not a plain identity value: {value!r}")
    return f"'{value}'"


def _floor(dt: datetime, bucket_minutes: int) -> datetime:
    epoch_minutes = int(dt.timestamp() // 60)
    return datetime.fromtimestamp((epoch_minutes // bucket_minutes) * bucket_minutes * 60, tz=timezone.utc)


def _promql_regex_literal(value: str) -> str:
    """
       A label value as an exact-match alternative inside a PromQL =~ string.
       Restricted to host:port characters first, so it can't close the quoted
       string or add matchers; then regex-escaped, with backslashes doubled
       because PromQL string literals unescape them once.
    """
    if not _INSTANCE_RE.fullmatch(value):
        raise ValueError(f"Not a plain host:port label value: {value!r}")
    return re.escape(value).replace("\\", "\\\\")


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")