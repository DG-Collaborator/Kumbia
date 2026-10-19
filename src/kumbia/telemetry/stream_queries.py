"""
      Builds each telemetry stream's SourceQuery for one time window — the
      piece TelemetryCorrelationPipeline deliberately leaves to its caller.
      Pure functions: no I/O, no persona knowledge, no LLM. Every value that
      ends up inside a query string comes from code (a TimeWindow of real
      datetimes, identifier names validated below, a registry's own IPs) —
      never from user or LLM text, same "structural, not instructional" rule
      as power_telemetry_report's fixed SQL.

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


def sql_power_window(
        db_path: str,
        window: TimeWindow,
        table: str = "power_readings",
        identity_column: str = "outlet",
        timestamp_column: str = "ts",
        value_column: str = "watts",
) -> SourceQuery:
    """
       Power readings inside the window from a sqlite table whose timestamp
       column holds ISO-8601 UTC text ("2026-09-24T14:05:00Z"). That format
       sorts lexicographically in time order, which is what makes a plain
       string >= / < comparison a correct time filter here.
    """
    for identifier in (table, identity_column, timestamp_column, value_column):
        _require_identifier(identifier)

    statement = (
        f"SELECT {identity_column}, {timestamp_column}, {value_column} FROM {table} "
        f"WHERE {timestamp_column} >= '{window.start_iso}' AND {timestamp_column} < '{window.end_iso}' "
        f"ORDER BY {timestamp_column}"
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
        max_buckets: int = 1000,
) -> SourceQuery:
    """
       Event counts per (host, time bucket) inside the window, as one
       Elasticsearch composite aggregation. ElasticsearchDataSource turns
       the response into rows with key.host, key.ts, doc_count columns.

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
        "aggs": {
            "by_host_bucket": {
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
        },
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