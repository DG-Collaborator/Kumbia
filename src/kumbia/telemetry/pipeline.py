"""
     Orchestration for the telemetry correlation layer: holds any number of
     named TelemetryStreams (a DataSource plus the SourceMapping that says how
     to read it) and a shared AssetRegistry, fetches every stream on demand,
     and delegates the actual join to correlation_engine. Deliberately LLM-free
     — narration/reasoning over these records is a persona concern, not this
     layer's. Deliberately doesn't build its own SourceQuery objects either:
     how to query a given telemetry system (time window, filters) is the
     caller's concern, supplied per stream name at correlate() time.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from kumbia.core.source.data_source import DataSource
from kumbia.core.source.source_query import SourceQuery
from kumbia.telemetry.asset_registry import AssetRegistry
from kumbia.telemetry.canonical_record import CanonicalRecord
from kumbia.telemetry.correlation_engine import SourceMapping, correlate_streams


@dataclass(frozen=True)
class TelemetryStream:
    """
       One telemetry domain feeding the correlation layer (e.g. "power",
       "compute", "network"). name is how callers address this stream's
       query in correlate(); it has no meaning beyond that.
    """
    name: str
    source: DataSource
    mapping: SourceMapping


class TelemetryCorrelationPipeline:
    def __init__(
            self,
            streams: list[TelemetryStream],
            registry: AssetRegistry,
            bucket_minutes: int = 5,
    ) -> None:
        names = [stream.name for stream in streams]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"TelemetryStream names must be unique, duplicated: {duplicates}")

        self._streams = list(streams)
        self._registry = registry
        self._bucket_minutes = bucket_minutes

    @property
    def stream_names(self) -> list[str]:
        return [stream.name for stream in self._streams]

    @property
    def bucket_minutes(self) -> int:
        return self._bucket_minutes

    async def correlate(self, queries: dict[str, SourceQuery]) -> list[CanonicalRecord]:
        """
           Fetches every stream with its own query (keyed by stream name),
           concurrently, and joins the results. queries must name exactly
           this pipeline's streams — a missing or unknown name is a caller
           bug, so it fails loudly instead of silently skipping a domain.
        """
        expected = set(self.stream_names)
        if set(queries) != expected:
            raise ValueError(
                f"queries must cover exactly the streams {sorted(expected)}, got {sorted(queries)}"
            )

        tables = await asyncio.gather(
            *(stream.source.fetch(queries[stream.name]) for stream in self._streams)
        )
        return correlate_streams(
            [(table, stream.mapping) for table, stream in zip(tables, self._streams)],
            self._registry,
            self._bucket_minutes,
        )