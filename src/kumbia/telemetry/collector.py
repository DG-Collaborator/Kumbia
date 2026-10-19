"""
     Continuous collection for the telemetry correlation layer: on a timer,
     asks TelemetryCorrelationPipeline for the next window of completed time
     buckets and appends the resulting CanonicalRecords to the on-disk store.
     Runs only between start() and stop() — nothing collects until someone
     asks it to. No MCP, persona, or LLM knowledge; whoever exposes start/stop
     (today, platform-level MCP tools) is a separate concern.

     Only completed buckets are ever collected: each cycle's window ends at
     the start of the current (still-filling) bucket, and the next window
     starts exactly where the last one ended. So a bucket is fetched once,
     after it has closed — no duplicates, no half-filled records.

     Queries are built per cycle by code-supplied builders (one per stream
     name) from a TimeWindow of real datetimes — never from user or LLM text.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

from kumbia.core.source.source_query import SourceQuery
from kumbia.telemetry.canonical_record_store import append_records, latest_timestamp
from kumbia.telemetry.pipeline import TelemetryCorrelationPipeline
from kumbia.telemetry.stream_queries import TimeWindow

QueryBuilder = Callable[[TimeWindow], SourceQuery]

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TelemetryCollector:
    def __init__(
            self,
            pipeline: TelemetryCorrelationPipeline,
            query_builders: dict[str, QueryBuilder],
            store_path: Path,
            poll_interval_seconds: float = 300.0,
            initial_lookback_minutes: int = 60,
            max_window_minutes: int = 60,
            ingest_delay_seconds: float = 15.0,
            clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        """
           query_builders must cover exactly the pipeline's stream names.
           initial_lookback_minutes: how far back the very first run reaches
           when the store is empty. max_window_minutes caps one cycle's
           window, so catching up after a long stop happens in bounded steps
           instead of one oversized query. ingest_delay_seconds holds a
           bucket back briefly after it closes, so readings that reach a
           source a few seconds late still land in the right cycle. clock is
           injectable for tests.
        """
        if set(query_builders) != set(pipeline.stream_names):
            raise ValueError(
                f"query_builders must cover exactly the streams {sorted(pipeline.stream_names)}, "
                f"got {sorted(query_builders)}"
            )
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive.")

        self._pipeline = pipeline
        self._query_builders = dict(query_builders)
        self._store_path = store_path
        self._poll_interval_seconds = poll_interval_seconds
        self._initial_lookback = timedelta(minutes=initial_lookback_minutes)
        self._max_window = timedelta(minutes=max_window_minutes)
        self._ingest_delay = timedelta(seconds=ingest_delay_seconds)
        self._clock = clock

        self._task: asyncio.Task | None = None
        self._stop_event: asyncio.Event | None = None
        self._watermark: datetime | None = None  # end of the last window collected
        self._records_collected = 0
        self._cycles_completed = 0
        self._last_error: str | None = None

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def poll_interval_seconds(self) -> float:
        return self._poll_interval_seconds

    @property
    def bucket_minutes(self) -> int:
        return self._pipeline.bucket_minutes

    @property
    def store_path(self) -> Path:
        return self._store_path

    @property
    def records_collected(self) -> int:
        """Records appended since this process started (not the store's total)."""
        return self._records_collected

    @property
    def cycles_completed(self) -> int:
        return self._cycles_completed

    @property
    def last_error(self) -> str | None:
        """The most recent cycle's failure, or None if the last cycle succeeded."""
        return self._last_error

    @property
    def collected_through(self) -> str | None:
        """Everything before this timestamp has been collected; None before the first start()."""
        return _iso(self._watermark) if self._watermark is not None else None

    def start(self) -> bool:
        """
           Begins collecting in the background. Returns False (and does
           nothing) if already running — a second start never spawns a
           second loop double-writing the store. Must be called from inside
           a running event loop (e.g. an MCP tool call).
        """
        if self.is_running:
            return False
        if self._watermark is None:
            self._watermark = self._resume_point()
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._run())
        return True

    async def stop(self) -> bool:
        """
           Signals the loop to stop and waits for it to finish. A cycle
           already in progress completes (fetch, correlate, append) rather
           than being cancelled mid-way. Returns False if it wasn't running.
        """
        if not self.is_running:
            return False
        self._stop_event.set()
        await self._task
        self._task = None
        return True

    async def collect_once(self) -> int:
        """
           One cycle: fetch the next window of completed buckets, correlate,
           append. Returns how many records were stored. The watermark only
           advances after a successful append, so a failed cycle is retried
           next time instead of leaving a gap.
        """
        if self._watermark is None:
            self._watermark = self._resume_point()

        end = self._latest_closed_bucket_end()
        start = self._watermark
        if start >= end:
            return 0
        end = min(end, start + self._max_window)

        window = TimeWindow(start, end)
        queries = {name: build(window) for name, build in self._query_builders.items()}
        records = await self._pipeline.correlate(queries)
        append_records(self._store_path, records)

        self._watermark = end
        self._records_collected += len(records)
        return len(records)

    async def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self.collect_once()
                self._last_error = None
            except Exception as error:  # keep collecting through a transient source failure
                self._last_error = f"{type(error).__name__}: {error}"
                logger.warning("Telemetry collection cycle failed: %s", self._last_error)
            self._cycles_completed += 1

            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._poll_interval_seconds)
            except TimeoutError:
                pass

    def _resume_point(self) -> datetime:
        """
           Continue right after the newest stored bucket, or, for an empty
           store, start initial_lookback_minutes ago (aligned to a bucket).
        """
        latest = latest_timestamp(self._store_path)
        if latest is not None:
            last_bucket = datetime.fromisoformat(latest.replace("Z", "+00:00"))
            return last_bucket + timedelta(minutes=self.bucket_minutes)
        return _floor(self._clock() - self._initial_lookback, self.bucket_minutes)

    def _latest_closed_bucket_end(self) -> datetime:
        return _floor(self._clock() - self._ingest_delay, self.bucket_minutes)


def _floor(dt: datetime, bucket_minutes: int) -> datetime:
    epoch_minutes = int(dt.timestamp() // 60)
    bucket_start = (epoch_minutes // bucket_minutes) * bucket_minutes
    return datetime.fromtimestamp(bucket_start * 60, tz=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
