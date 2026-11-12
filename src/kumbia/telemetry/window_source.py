"""
     On-demand correlation over any trailing window, exposed as a DataSource
     so a persona can read it exactly like any other named source.

     The collector correlates continuously into a store, but only while it is
     running and only from the moment it started. This runs the same
     TelemetryCorrelationPipeline with the same per-stream query builders
     straight against the live sources for whatever window is asked about —
     "the past 24 hours" works whether or not collection was ever started —
     and returns the joined records as a generic DataTable (one row per asset
     per bucket, one column per reading), the shape telemetry.analysis reads.

     plan() picks the window and bucket size (coarser buckets for longer
     windows, so the network stream's composite aggregation stays within one
     page); fetch() takes that plan back as its SourceQuery statement. The
     statement only ever holds timestamps and a bucket size produced by
     plan(), never question text.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone

from kumbia.core.domain.data_table import DataTable
from kumbia.core.source.source_query import SourceQuery
from kumbia.telemetry.canonical_record_store import records_to_table
from kumbia.telemetry.collector import QueryBuilder
from kumbia.telemetry.pipeline import TelemetryCorrelationPipeline
from kumbia.telemetry.stream_queries import TimeWindow, WindowPlan, plan_window


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CorrelatedWindowSource:
    def __init__(
            self,
            pipeline: TelemetryCorrelationPipeline,
            query_builders: dict[str, QueryBuilder],
            min_bucket_minutes: int,
            series_count: int = 1,
            clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        """
           query_builders must cover exactly the pipeline's stream names (the
           collector's builders are the intended ones). min_bucket_minutes is
           the finest bucket ever used — the collector's own size, below
           which sources aren't expected to have distinct readings.
           series_count is how many series the largest stream returns per
           bucket (registered network hosts), which bounds the bucket count.
        """
        if set(query_builders) != set(pipeline.stream_names):
            raise ValueError(
                f"query_builders must cover exactly the streams {sorted(pipeline.stream_names)}, "
                f"got {sorted(query_builders)}"
            )
        self._pipeline = pipeline
        self._query_builders = dict(query_builders)
        self._min_bucket_minutes = min_bucket_minutes
        self._series_count = series_count
        self._clock = clock

    def plan(self, hours_back: int) -> WindowPlan:
        return plan_window(hours_back, self._clock(), self._min_bucket_minutes, self._series_count)

    @staticmethod
    def query_for(plan: WindowPlan) -> SourceQuery:
        statement = {
            "hours_back": plan.hours_back,
            "start": plan.window.start_iso,
            "end": plan.window.end_iso,
            "bucket_minutes": plan.bucket_minutes,
        }
        return SourceQuery(address="correlated_window", statement=json.dumps(statement))

    async def fetch(self, query: SourceQuery) -> DataTable:
        plan = _plan_from_statement(query.statement)
        queries = {
            name: build(plan.window, plan.bucket_minutes) for name, build in self._query_builders.items()
        }
        records = await self._pipeline.correlate(queries, bucket_minutes=plan.bucket_minutes)
        return records_to_table(records)


def _plan_from_statement(statement: str | None) -> WindowPlan:
    if not statement:
        raise ValueError("CorrelatedWindowSource needs a statement from query_for(plan).")
    data = json.loads(statement)
    window = TimeWindow(_parse(data["start"]), _parse(data["end"]))
    bucket_minutes = int(data["bucket_minutes"])
    if bucket_minutes < 1:
        raise ValueError("bucket_minutes must be at least 1.")
    return WindowPlan(int(data["hours_back"]), bucket_minutes, window)


def _parse(iso: str) -> datetime:
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))
